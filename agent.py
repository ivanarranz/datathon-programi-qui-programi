"""
agent.py
Agent de control predictiu (MPC + Machine Learning) per al Repte UPC Sostenible.
Datathon 2026.
"""

from collections import deque
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class BatteryAgent:
    def __init__(self):
        self.cap = 2000.0
        self.power = 500.0
        self.eta = 0.90 ** 0.5  # ~0.948683
        self.historial = deque(maxlen=48)

        directori = Path(__file__).resolve().parent
        ruta_model = directori / "model_demanda_neta.pkl"
        self.model = joblib.load(ruta_model) if ruta_model.exists() else None

        # Perfil horari mitjà de seguretat si encara no tenim 24h d'historial
        self.perfil_base = [
            850, 820, 800, 790, 810, 880, 1020, 1180,
            1320, 1410, 1450, 1460, 1430, 1380, 1350, 1320,
            1280, 1220, 1150, 1080, 1010, 950, 900, 870
        ]

    def _predir_demanda_24h(self, obs: dict, horitzó: int) -> np.ndarray:
        ts_actual = pd.Timestamp(obs["timestamp"])
        sensors = obs.get("noves_dades_5min", {})
        prev_load = float(sensors.get("ultim_consum_kw", 1050.0))
        prev_pv = float(sensors.get("ultima_solar_kw", 0.0))
        prev_neta = max(0.0, prev_load - prev_pv)

        self.historial.append({
            "hour": int(obs["hour"]),
            "neta": prev_neta
        })

        # Obtenim la demanda neta d'ahir a la mateixa hora si tenim historial suficient
        neta_24h = self.historial[-24]["neta"] if len(self.historial) >= 24 else prev_neta

        # 1. Predim l'hora actual (k=0) amb el model entrenat
        if self.model is not None:
            X_0 = pd.DataFrame([{
                "hour": int(obs["hour"]),
                "dayofweek": int(obs["day_of_week"]),
                "month": int(ts_actual.month),
                "is_weekend": 1 if int(obs["day_of_week"]) >= 5 else 0,
                "preu_actual": float(obs["preu_actual"]),
                "co2_actual": float(obs["co2_actual"]),
                "prev_load_1h": prev_load,
                "prev_pv_1h": prev_pv,
                "prev_neta_1h": prev_neta,
                "prev_neta_24h": neta_24h
            }])
            pred_0 = max(50.0, float(self.model.predict(X_0)[0]))
        else:
            pred_0 = max(50.0, prev_neta)

        # 2. Estimem les properes hores (k=1..H-1) combinant el perfil horari amb el nivell recent
        demandes = np.zeros(horitzó, dtype=float)
        demandes[0] = pred_0

        factor_escala = pred_0 / max(100.0, self.perfil_base[int(obs["hour"])])
        factor_escala = float(np.clip(factor_escala, 0.5, 1.8))

        for k in range(1, horitzó):
            ts_k = ts_actual + pd.Timedelta(hours=k)
            h_k = ts_k.hour
            # Si tenim historial de fa 24h per aquella hora, el ponderem amb el perfil escalat
            idx_hist = len(self.historial) - 24 + k
            if 0 <= idx_hist < len(self.historial):
                hist_val = self.historial[idx_hist]["neta"]
                demandes[k] = 0.65 * hist_val + 0.35 * (self.perfil_base[h_k] * factor_escala)
            else:
                demandes[k] = self.perfil_base[h_k] * factor_escala

        return np.maximum(50.0, demandes)

    def act(self, obs: dict) -> float:
        soc_kwh = float(obs["soc_kwh"])
        co2_24 = np.array(obs.get("co2_properes_24h", [obs["co2_actual"]]), dtype=float)
        preus_24 = np.array(obs.get("preus_properes_24h", [obs["preu_actual"]]), dtype=float)
        H = len(co2_24)

        demandes_24 = self._predir_demanda_24h(obs, H)

        # Cost marginal ponderat oficial: 70% CO2 + 30% Preu (normalitzats per la mitjana)
        co2_norm = co2_24 / (np.mean(co2_24) + 1e-9)
        preu_norm = preus_24 / (np.mean(preus_24) + 1e-9)
        w = 0.70 * co2_norm + 0.30 * preu_norm

        # Optimització lineal (MPC) sobre l'horitzó de H hores:
        # Variables: [P_carrega_0..H-1, P_descarrega_0..H-1] (2*H variables)
        # Volem minimitzar: sum(w_k * P_c_k - w_k * P_d_k)
        c = np.concatenate([w, -w])

        # Fites de cada variable:
        # 0 <= P_c_k <= 500
        # 0 <= P_d_k <= min(500, demanda_neta_k) (per no exportar energia a preu 0)
        bounds_c = [(0.0, self.power) for _ in range(H)]
        bounds_d = [(0.0, float(min(self.power, demandes_24[k]))) for k in range(H)]
        bounds = bounds_c + bounds_d

        # Restriccions d'estat de càrrega (SOC_k en [0, 2000] per a tot k=1..H):
        # SOC_k = soc_kwh + sum_{i=0..k-1} (eta * P_c_i - (1/eta) * P_d_i)
        A_ub = []
        b_ub = []

        for k in range(1, H + 1):
            # 1) SOC_k <= 2000  =>  sum(eta * P_c_i - (1/eta) * P_d_i) <= 2000 - soc_kwh
            row_max = np.zeros(2 * H)
            row_max[:k] = self.eta
            row_max[H:H + k] = -1.0 / self.eta
            A_ub.append(row_max)
            b_ub.append(self.cap - soc_kwh)

            # 2) SOC_k >= 0  =>  sum(-eta * P_c_i + (1/eta) * P_d_i) <= soc_kwh
            row_min = np.zeros(2 * H)
            row_min[:k] = -self.eta
            row_min[H:H + k] = 1.0 / self.eta
            A_ub.append(row_min)
            b_ub.append(soc_kwh)

        # Afegim una reserva mínima suau al final de les 24h (si H >= 12) per no buidar del tot la bateria
        if H >= 12:
            row_end = np.zeros(2 * H)
            row_end[:H] = -self.eta
            row_end[H:2 * H] = 1.0 / self.eta
            A_ub.append(row_end)
            b_ub.append(max(0.0, soc_kwh - 600.0))

        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")

        if res.success:
            consigna = float(res.x[0] - res.x[H])
        else:
            # Fallback segur si l'optimitzador no convergeix
            consigna = 300.0 if w[0] < np.percentile(w, 30) else (-300.0 if w[0] > np.percentile(w, 70) else 0.0)

        # Clamping físic estricte per garantir 0 infraccions
        eps = 1e-4
        max_carrega = max(0.0, min(self.power, ((self.cap - soc_kwh) / self.eta) - eps))
        max_descarrega = max(0.0, min(self.power, (soc_kwh * self.eta) - eps))

        return float(np.clip(consigna, -max_descarrega, max_carrega))
