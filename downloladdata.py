import requests
import pandas as pd
from datetime import datetime

LATITUDE = 12.9141
LONGITUDE = 74.8560


START_DATE = "20150101"
END_DATE   = "20251231"


PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",   # Solar Radiation
    "T2M",                 # Temperature
    "RH2M",                # Relative Humidity
    "WS2M",                # Wind Speed
    "CLOUD_AMT",           # Cloud Amount
    "PRECTOTCORR"          # Precipitation
]

param_string = ",".join(PARAMETERS)

# ------------------------------------------------
# NASA POWER API URL
# ------------------------------------------------
url = (
    f"https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"parameters={param_string}"
    f"&community=RE"
    f"&longitude={LONGITUDE}"
    f"&latitude={LATITUDE}"
    f"&start={START_DATE}"
    f"&end={END_DATE}"
    f"&format=JSON"
)

print("Downloading dataset...")
response = requests.get(url)

# Check request
if response.status_code != 200:
    raise Exception(f"API Error: {response.status_code}")

data = response.json()

print("Dataset downloaded successfully!")

# ------------------------------------------------
# Convert JSON to DataFrame
# ------------------------------------------------
parameters_data = data["properties"]["parameter"]

# Create DataFrame
df = pd.DataFrame(parameters_data)

# Convert index to datetime
df.index = pd.to_datetime(df.index, format="%Y%m%d")

# Rename index column
df.index.name = "DATE"

# Reset index
df.reset_index(inplace=True)

# ------------------------------------------------
# Handle Missing Values
# NASA uses -999 sometimes
# ------------------------------------------------
df = df[~(df == -999).any(axis=1)]
df['date_column'] = pd.to_datetime(df['DATE'])
# Split into separate columns
df['day'] = df['date_column'].dt.day
df['month'] = df['date_column'].dt.month
df['year'] = df['date_column'].dt.year

df.drop(columns=['date_column','DATE'], inplace=True)
# ------------------------------------------------
# Save CSV
# ------------------------------------------------
csv_filename = "dakshina_kannada_solar_radiation_dataset.csv"

df.to_csv(csv_filename, index=False)

print(f"\nCSV saved successfully as: {csv_filename}")

# ------------------------------------------------
# Show sample data
# ------------------------------------------------
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)