import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import config


@dataclass
class AnomalyResult:
    step: int
    similarity: float
    anomaly_score: float
    is_anomaly: bool
    spike: bool
    confidence: float
    reason: str


class AnomalyDetector:

    def __init__(self, engine):
        self._engine = engine
        self._step = 0
        self._history = deque(maxlen=config.BASELINE_WINDOW)

    def _spike_check(self, score: float):
        if len(self._history) < 10:
            return False, 0.0

        arr = np.array(self._history)
        mean = arr.mean()
        std = arr.std()

        if std < 1e-6:
            return False, 0.0

        z = (score - mean) / std
        return z > config.SPIKE_Z_SCORE, min(1.0, z / 4)

    def process(self, vector: np.ndarray, reading=None):

        self._step += 1

        similarity = self._engine.search(vector)
        anomaly_score = 1.0 - similarity

        # ✅ WARMUP
        if self._step <= config.WARMUP_STEPS:
            self._engine.store(vector)
            return AnomalyResult(
                self._step, similarity, anomaly_score,
                False, False, 0.0,
                f"WARMUP {self._step}"
            )

        spike, spike_conf = self._spike_check(anomaly_score)
        self._history.append(anomaly_score)

        low_similarity = similarity < config.ANOMALY_THRESHOLD
        is_anomaly = low_similarity or spike

        if is_anomaly:
            reason = "Anomaly detected"
            confidence = max(
                (config.ANOMALY_THRESHOLD - similarity),
                spike_conf
            )
        else:
            self._engine.store(vector)
            reason = "Normal"
            confidence = 0.0

        return AnomalyResult(
            self._step,
            similarity,
            anomaly_score,
            is_anomaly,
            spike,
            round(confidence, 3),
            reason
        )
