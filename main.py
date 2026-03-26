import time
import math
import random
import streamlit as st
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from core.qdrant_engine import QdrantEdgeEngine
from intelligence.anomaly_engine import AnomalyDetector

# ----------------------------
# UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("Real-Time Sensor Anomaly Detection (Qdrant Edge)")

# ----------------------------
# Sensor object
# ----------------------------
class SensorReading:
    def __init__(self, temp, hum, vib):
        self.temperature = temp
        self.humidity = hum
        self.vibration = vib
        self.ground_truth_anomaly = False
        self.anomaly_label = None


# ----------------------------
# INIT
# ----------------------------
engine = QdrantEdgeEngine()
detector = AnomalyDetector(engine)

left_col, right_col = st.columns([1, 2])

table_placeholder = left_col.empty()
chart_placeholder = right_col.empty()

# ----------------------------
# DATA STORAGE
# ----------------------------
rows = []
similarity = []
anomaly_x = []
anomaly_y = []

t = 0
step = 0

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:

    step += 1
    t += 0.1

    # normal signal
    temperature = 25 + 3 * math.sin(t) + random.uniform(-0.5, 0.5)
    humidity = 60 + 8 * math.sin(t / 2) + random.uniform(-1, 1)
    vibration = 0.02 + 0.02 * math.sin(t) + random.uniform(-0.01, 0.01)

    #  anomalies injection
    if random.random() < 0.1:
        temperature += random.uniform(40, 70)
        vibration += random.uniform(3, 6)

    if random.random() < 0.1:
        humidity -= random.uniform(20, 40)

    reading = SensorReading(temperature, humidity, vibration)

    vector = np.array([temperature, humidity, vibration])
    result = detector.process(vector, reading)

    similarity.append(result.similarity)

    # ----------------------------
    # STORE ROW
    # ----------------------------
    rows.append({
        "Step": step,
        "Similarity": round(result.similarity, 3),
        "Temp": round(temperature, 2),
        "Humidity": round(humidity, 2),
        "Vibration": round(vibration, 3),
        "Anomaly": "YES" if result.is_anomaly else "NO"
    })

    rows = rows[-50:]  # keep last 50 rows

    # ----------------------------
    # STORE ANOMALY POINTS
    # ----------------------------
    if result.is_anomaly:
        anomaly_x.append(len(similarity))
        anomaly_y.append(result.similarity)

    # ----------------------------
    # UI UPDATE (LESS FREQUENT)
    # ----------------------------
    if step % 3 == 0:

        # LEFT TABLE WITH HIGHLIGHT
        df = pd.DataFrame(rows)

        def highlight(val):
            return "background-color: red" if val == "YES" else ""

        styled_df = df.style.applymap(highlight, subset=["Anomaly"])

        table_placeholder.dataframe(styled_df, width="stretch")

        #  RIGHT GRAPH
        fig = go.Figure()

        # last 50 similarity values
        last_sim = similarity[-50:]
        start_index = len(similarity) - len(last_sim)
        last_indices = list(range(start_index, len(similarity)))

        #  BLUE LINE
        fig.add_trace(go.Scatter(
            x=last_indices,
            y=last_sim,
            mode='lines',
            name='Similarity'
        ))

        #  FILTER ANOMALIES (ONLY VISIBLE RANGE)
        filtered_x = []
        filtered_y = []

        for i in range(len(anomaly_x)):
            if anomaly_x[i] >= start_index:
                filtered_x.append(anomaly_x[i])
                filtered_y.append(anomaly_y[i])

        #  RED DOTS
        fig.add_trace(go.Scatter(
            x=filtered_x,
            y=filtered_y,
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomaly'
        ))

        fig.update_layout(
            yaxis=dict(range=[0.95, 1.0])
        )

        chart_placeholder.plotly_chart(fig, width="stretch")

        time.sleep(1.5)
