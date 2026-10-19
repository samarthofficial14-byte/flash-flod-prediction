# 🌊 FlashFlood AI: Multi-Source Flash Flood Early Warning System for Hilly Regions

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An intelligent, real-time decision support system that predicts flash flood hazards in steep mountainous terrains (Western Ghats, Himalayan Foothills) by fusing live meteorological API telemetry, digital elevation data, hydrological river gauges, and soil saturation indices.**

---

## 📌 Problem Statement

Mountainous and hilly terrains (e.g., Wayanad, Chamoli/Kedarnath, Kullu Valley) face catastrophic flash floods and debris flows characterized by **extremely short lag times (< 2 hours)** between intense precipitation and peak river discharge. 

Traditional flood forecasting systems rely almost solely on downstream river gauges or broad synoptic rain forecasts, failing to account for **topographic slope acceleration**, **antecedent catchment saturation**, and **localized cloudburst dynamics**. 

**FlashFlood AI** bridges this gap by unifying multi-source environmental telemetry into an ensemble machine learning classifier that delivers actionable early warning alerts.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Sources ["Multi-Source Data Ingestion"]
        A["📡 Live Meteorological Telemetry (Open-Meteo REST API)<br>• Hourly Rainfall Rate (mm/h)<br>• 24h Cumulative Precipitation (mm)"]
        B["⛰️ Digital Elevation Model (SRTM DEM)<br>• Terrain Slope Gradient (°)<br>• Catchment Elevation (m ASL)"]
        C["🌊 Hydrological Sensors (CWC / IoT Stream Gauges)<br>• River Discharge % of Danger Mark"]
        D["🌱 Surface & Soil Telemetry (NASA SMAP / MODIS NDVI)<br>• Soil Moisture Saturation (%)<br>• Vegetation Index (NDVI)"]
    end

    subgraph ML_Engine ["Predictive Intelligence Engine"]
        E["Feature Extraction & Normalization"]
        F["Balanced Random Forest Classifier (150 Trees)<br>• Cost-sensitive class weighting for rare disaster events"]
        G["Explainable AI (Feature Attribution)"]
    end

    subgraph Decision_Support ["Early Warning & Decision Support (EWS)"]
        H["🟢 Green (Normal Baseflow)"]
        I["🟡 Yellow (Watch Advisory)"]
        J["🟠 Orange (Flash Flood Warning)"]
        K["🔴 Red (Imminent Debris Flow - 60-90min Evacuation)"]
        L["Interactive Streamlit Geospatial Dashboard"]
    end

    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
    F --> J
    F --> K
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
```

---

## 🔬 Machine Learning Model & Evaluation

The system leverages a **Random Forest Classifier** trained on 5,500 hydrology-modeled observations governing runoff physics (Rational Runoff equation and USDA SCS Curve Number methodology):

- **Overall Accuracy**: `84.7%`
- **Critical Flash Flood Precision**: `100.0%` (Zero false alarms during extreme cloudburst events)
- **High-Risk Surge Recall**: `91.5%` (Captures over 91% of hazardous surges before peak inundation)
- **Weighted F1-Score**: `84.7%`

### Top Hydrological Drivers (Feature Importance):
1. **Rainfall Intensity**: `32.31%`
2. **24-Hour Antecedent Rainfall**: `31.01%`
3. **River Discharge Stress**: `15.18%`
4. **Soil Moisture Saturation**: `11.37%`
5. **Terrain Slope Gradient**: `4.45%`
6. **Vegetation Retention (NDVI)**: `2.93%`
7. **Catchment Elevation**: `2.75%`

---

## 🚀 Key Features

- **🌐 Live REST API Integration**: Pulls real-time precipitation and 24h accumulated rainfall directly from Open-Meteo for monitored mountain stations.
- **🗺️ Geospatial Vulnerability Mapping**: Interactive station overview covering critical Indian mountain basins (Wayanad, Chamoli/Kedarnath, Idukki, Kullu, Darjeeling).
- **⚡ Scenario Simulation Engine**: Test what-if events including normal monsoon showers, prolonged downpours, or extreme cloudbursts (>80 mm/h).
- **🧠 Explainable AI (XAI)**: Visualizes real-time factor stress and model feature importances to explain why an alert was triggered.
- **🚨 Actionable Decision Support**: NDMA/SDRF-aligned crisis alert bulletins with recommended evacuation lead times.

---

## 🛠️ Project Structure

```text
FlashFlood-AI/
├── .venv/                   # Python Virtual Environment
├── data/
│   ├── generate_dataset.py  # Hydrology data simulation script
│   └── flash_flood_data.csv # 5,500 observation records
├── models/
│   ├── train_model.py       # Model training & benchmarking pipeline
│   ├── flood_model.pkl      # Serialized Random Forest model
│   └── model_metadata.json  # Exported metrics & feature importances
├── app.py                   # Streamlit interactive early warning dashboard
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & architecture
```

---

## 💻 Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/flash-flood-prediction.git
cd flash-flood-prediction
```

### 2. Set up virtual environment and install dependencies
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Re-generate data and re-train model
```bash
python data/generate_dataset.py
python models/train_model.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📝 Resume Bullet Points (Copy & Paste)

- **Flash Flood Early Warning System for Hilly Regions** | *Python, Streamlit, Scikit-Learn, Plotly, Open-Meteo REST API*
  - Engineered an end-to-end multi-source early warning pipeline fusing live meteorological API telemetry, digital elevation slope metrics, soil saturation, and river gauge levels.
  - Trained an ensemble Random Forest classifier on 5,500+ hydrology-modeled observation points, achieving **84.7% accuracy**, **91.5% high-risk recall**, and **100% precision on critical flood events**.
  - Built an interactive Streamlit crisis dashboard with live cloudburst simulations, geospatial vulnerability mapping across 5 mountain basins, and explainable AI factor attribution.

