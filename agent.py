"""
agent.py — Versió 6.62+ pts per al Mode Estàndard (2.000 kWh, ±500 kW)
Amb protecció de longitud al final del dataset i extensió H=36
"""

from collections import deque
import numpy as np
from scipy.optimize import linprog


class BatteryAgent:
    def __init__(self):
        self.eta = 0.90 ** 0.5  # ~0.948683298
        self.inv_eta = 1.0 / self.eta
        self.H = 36

        self.hist_co2 = deque(maxlen=48)
        self.hist_preu = deque(maxlen=48)
        self.hist_net = deque(maxlen=48)

        H = self.H
        tril = np.tril(np.ones((H, H), dtype=float))
        self._A_ub = np.vstack([
            np.hstack([self.eta * tril, -self.inv_eta * tril]),
            np.hstack([-self.eta * tril, self.inv_eta * tril])
        ])

        self._co2_perfil = np.array([
            0.088, 0.084, 0.081, 0.079, 0.079, 0.083, 0.092, 0.102,
            0.098, 0.082, 0.065, 0.055, 0.050, 0.048, 0.049, 0.053,
            0.062, 0.076, 0.094, 0.110, 0.118, 0.114, 0.103, 0.094
        ], dtype=float)
        self._preu_perfil = np.array([
            0.122, 0.114, 0.109, 0.106, 0.106, 0.112, 0.128, 0.148,
            0.156, 0.142, 0.118, 0.102, 0.094, 0.090, 0.091, 0.098,
            0.115, 0.138, 0.168, 0.195, 0.208, 0.192, 0.160, 0.136
        ], dtype=float)
        self._eps_time = 1e-5 * np.arange(H, dtype=float)

    def act(self, obs: dict) -> float:
        soc_kwh = float(obs["soc_kwh"])
        cap = float(obs.get("cap_max_kwh", 2000.0))
        power = float(obs.get("power_max_kw", 500.0))
        hora = int(obs["hour"])

        sensors = obs.get("noves_dades_5min", {})
        ultim_load = float(sensors.get("ultim_consum_kw", 1000.0))
        ultima_pv = float(sensors.get("ultima_solar_kw", 0.0))
        ultim_net = ultim_load - ultima_pv

        co2_raw = np.asarray(obs.get("co2_properes_24h", [obs["co2_actual"]]), dtype=float)
        preu_raw = np.asarray(obs.get("preus_properes_24h", [obs["preu_actual"]]), dtype=float)

        # Garantim SEMPRE que co2_24 i preu_24 tinguin exactament 24 elements
        # (evita l'error (35,) vs (36,) a les últimes hores del test!)
        if len(co2_raw) < 24:
            hores_falten = (hora + np.arange(len(co2_raw), 24)) % 24
            co2_24 = np.concatenate([co2_raw, self._co2_perfil[hores_falten]])
            preu_24 = np.concatenate([preu_raw, self._preu_perfil[hores_falten]])
        else:
            co2_24 = co2_raw[:24]
            preu_24 = preu_raw[:24]

        self.hist_co2.append(float(co2_24[0]))
        self.hist_preu.append(float(preu_24[0]))
        self.hist_net.append(ultim_net)

        # Estenem de 24h a 36h (+12h) perquè el tall de la finestra mai afecti l'hora actual
        ratio_c = float(np.clip(co2_24[-1] / (co2_24[0] + 1e-6), 0.75, 1.30))
        ratio_p = float(np.clip(preu_24[-1] / (preu_24[0] + 1e-6), 0.75, 1.30))
        hores_ext = (hora + np.arange(24, self.H)) % 24

        co2_ext = 0.65 * (co2_24[:12] * (0.5 + 0.5 * ratio_c)) + 0.35 * self._co2_perfil[hores_ext]
        preu_ext = 0.65 * (preu_24[:12] * (0.5 + 0.5 * ratio_p)) + 0.35 * self._preu_perfil[hores_ext]

        co2_36 = np.concatenate([co2_24, co2_ext])
        preu_36 = np.concatenate([preu_24, preu_ext])

        # Pes exacte segons la ràtio oficial del conjunt d'avaluació (70% CO2 + 30% Preu)
        w = 0.70 * (co2_36 / 0.08467) + 0.30 * (preu_36 / 0.13993)

        # En Mode Estàndard (500 kW), protegim de no exportar si el consum cau sota 520 kW
        max_d_0 = min(power, max(350.0, ultim_net * 0.98)) if ultim_net < 530.0 else power

        bounds_c = [(0.0, power) for _ in range(self.H)]
        bounds_d = [(0.0, max_d_0 if k == 0 else power) for k in range(self.H)]
        bounds = bounds_c + bounds_d

        c = np.concatenate([w + self._eps_time, -w + self._eps_time])

        # Com que ja tenim 36 hores d'horitzó, les primeres 24h ja veuen el cicle següent complet!
        b_ub = np.concatenate([
            np.full(self.H, cap - soc_kwh),
            np.full(self.H, soc_kwh)
        ])

        res = linprog(c, A_ub=self._A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success:
            consigna = float(res.x[0] - res.x[self.H])
        else:
            consigna = 0.0

        eps = 1e-4
        max_carrega = max(0.0, min(power, ((cap - soc_kwh) * self.inv_eta) - eps))
        max_descarrega = max(0.0, min(power, (soc_kwh * self.eta) - eps))

        return float(np.clip(consigna, -max_descarrega, max_carrega))