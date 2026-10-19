import numpy as np
import pandas as pd
import os

def generate_flash_flood_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    # 1. Meteorological: Rainfall Intensity (mm/hr) - skewed towards low/moderate with heavy tails for cloudbursts
    # In hilly regions, 0-15 mm/h is light/mod, 15-50 is heavy, 50-120 is cloudburst/extreme
    rainfall_intensity = np.random.exponential(scale=18.0, size=n_samples)
    rainfall_intensity = np.clip(rainfall_intensity, 0.0, 130.0)

    # 2. Meteorological: Cumulative 24-Hour Antecedent Rainfall (mm)
    # Prolonged rain primes the catchment for catastrophic runoff
    rainfall_24h = rainfall_intensity * np.random.uniform(2.5, 4.5, size=n_samples) + np.random.gamma(shape=3.0, scale=20.0, size=n_samples)
    rainfall_24h = np.clip(rainfall_24h, 0.0, 420.0)

    # 3. Hydrological: River / Stream Gauge Level (% of Danger Mark)
    # Normal is 20-50%, 80%+ is near bankfull, >100% is overflowing
    base_river = np.random.beta(a=2.0, b=3.5, size=n_samples) * 100.0
    river_level = base_river + (rainfall_24h / 420.0) * 45.0 + (rainfall_intensity / 130.0) * 20.0
    river_level = np.clip(river_level + np.random.normal(0, 4, size=n_samples), 5.0, 150.0)

    # 4. Soil / Surface: Soil Moisture Saturation (%)
    # In hilly terrain, saturated soil cannot absorb more rain (infiltration excess runoff)
    base_soil = np.random.uniform(20.0, 65.0, size=n_samples)
    soil_moisture = base_soil + (rainfall_24h / 420.0) * 40.0
    soil_moisture = np.clip(soil_moisture + np.random.normal(0, 3, size=n_samples), 10.0, 100.0)

    # 5. Topographical: Terrain Slope (Degrees)
    # Hilly / mountainous terrain: 10 to 55 degrees
    slope_deg = np.random.triangular(left=8.0, mode=28.0, right=55.0, size=n_samples)

    # 6. Environmental: Vegetation Index (NDVI: -0.1 to 0.85)
    # Dense vegetation absorbs water and slows runoff; degraded / deforested slopes cause immediate flash floods
    vegetation_index = np.random.uniform(0.10, 0.82, size=n_samples)

    # 7. Elevation (Meters above sea level) - e.g. Western Ghats / Himalayan Foothills
    elevation_m = np.random.uniform(400.0, 3200.0, size=n_samples)

    # Hydrological Runoff Susceptibility Index (Domain Physics-guided):
    # Normalized components (0 to 1 scale)
    norm_rain_rate = (rainfall_intensity / 100.0) ** 1.2
    norm_rain_24h = (rainfall_24h / 350.0)
    norm_river = (river_level / 120.0)
    norm_soil = (soil_moisture / 100.0) ** 1.3
    norm_slope = np.sin(np.radians(slope_deg))  # Slope gradient drives gravitational velocity
    vegetation_retention = vegetation_index * 0.35  # Vegetation reduces flash flood velocity

    # Composite Flash Flood Severity Score (0 to 100)
    composite_index = (
        0.30 * norm_rain_rate +
        0.20 * norm_rain_24h +
        0.22 * norm_river +
        0.18 * norm_soil +
        0.15 * norm_slope -
        0.05 * vegetation_retention
    ) * 100.0

    # Add realistic environmental sensor noise and localized microclimate variance
    noise = np.random.normal(0, 4.0, size=n_samples)
    composite_score = np.clip(composite_index + noise, 0.0, 100.0)

    # Categorize into 4 Standard EWS (Early Warning System) Alert Tiers:
    # 0: Low (Normal)
    # 1: Moderate (Watch - Advisory)
    # 2: High (Warning - Local Inundation)
    # 3: Critical (Emergency - Flash Flood / Debris Flow Imminent)
    risk_tier = np.zeros(n_samples, dtype=int)
    risk_tier[composite_score >= 32.0] = 1
    risk_tier[composite_score >= 55.0] = 2
    risk_tier[composite_score >= 76.0] = 3

    tier_labels = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}

    df = pd.DataFrame({
        "rainfall_intensity_mmh": np.round(rainfall_intensity, 2),
        "rainfall_24h_mm": np.round(rainfall_24h, 2),
        "river_water_level_pct": np.round(river_level, 2),
        "soil_moisture_pct": np.round(soil_moisture, 2),
        "slope_degrees": np.round(slope_deg, 2),
        "vegetation_index": np.round(vegetation_index, 3),
        "elevation_m": np.round(elevation_m, 1),
        "susceptibility_score": np.round(composite_score, 2),
        "risk_tier": risk_tier,
        "risk_label": [tier_labels[t] for t in risk_tier]
    })

    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_flash_flood_dataset(n_samples=5500)
    output_path = os.path.join("data", "flash_flood_data.csv")
    dataset.to_csv(output_path, index=False)
    
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total Records: {len(dataset)}")
    print("\nClass Distribution:")
    print(dataset["risk_label"].value_counts())
    print("\nFeature Summary:")
    print(dataset.describe().T[["mean", "std", "min", "max"]])

