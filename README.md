# Data-Cleaning
https://www.kaggle.com/datasets/rafsunahmad/plane-price-prediction/code
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("airplane.zip")

# Replace string "None" with np.nan before any float conversion
df.replace("None", np.nan, inplace=True)

df.info()

# Convert categorical 'Engine Type' with one-hot encoding properly
enginetypes = pd.get_dummies(df["Engine Type"], drop_first=True)
df = df.drop("Engine Type", axis=1)
df = pd.concat([df, enginetypes], axis=1)

# Clean columns with commas and convert to float-friendly format
df["Landing over 50ft"] = df["Landing over 50ft"].str.replace(",", ".")
df["Empty weight lbs"] = df["Empty weight lbs"].str.replace(",", ".")
df["Length ft/in"] = df["Length ft/in"].str.replace(",", ".")

# Replace known invalid values with zero or median later
df["Length ft/in"] = df["Length ft/in"].str.replace('Orig', '0')
df["Length ft/in"] = df["Length ft/in"].str.replace('N/C', '0')

# Split length into feet/inches and convert to decimal feet
length_split = df["Length ft/in"].str.split("/", expand=True)
df["Length ft/in"] = length_split[0].astype(float) + length_split[1].astype(float) / 12

# Replace zero or missing lengths with median value
median_length = df["Length ft/in"].replace(0, np.nan).median()
df["Length ft/in"].replace(0, median_length, inplace=True)
df["Length ft/in"].fillna(median_length, inplace=True)

# Same process for wing span
df["Wing span ft/in"] = df["Wing span ft/in"].str.replace("Orig", "0")
df["Wing span ft/in"] = df["Wing span ft/in"].str.replace("N/C", "0")
df["Wing span ft/in"] = df["Wing span ft/in"].str.replace(".", "/")

wing_split = df["Wing span ft/in"].str.split("/", expand=True)
df["Wing span ft/in"] = wing_split[0].astype(float) + wing_split[1].astype(float) / 12

median_wing = df["Wing span ft/in"].replace(0, np.nan).median()
df["Wing span ft/in"].replace(0, median_wing, inplace=True)
df["Wing span ft/in"].fillna(median_wing, inplace=True)

# Replace commas and convert other columns to float-friendly format
df["All eng rate of climb"] = df["All eng rate of climb"].str.replace(",", ".")
df["Range N.M."] = df["Range N.M."].str.replace(",", ".")

# Clean and convert HP or lbs thr ea engine column
df["HP or lbs thr ea engine"] = df["HP or lbs thr ea engine"].str.replace(",", ".")
df["HP or lbs thr ea engine"] = df["HP or lbs thr ea engine"].str.replace("dry", "")
df["HP or lbs thr ea engine"] = df["HP or lbs thr ea engine"].str.replace("wet", "")
df["HP or lbs thr ea engine"] = df["HP or lbs thr ea engine"].replace("", np.nan)

# Clean Max speed Knots column with "Mach" values converted
df["Max speed Knots"] = df["Max speed Knots"].str.replace("Mach", "")
df["Max speed Knots"] = df["Max speed Knots"].str.replace(".", "*")

values = df["Max speed Knots"].str.split("*", expand=True).astype(str)
values.columns = ["value1", "value2"]

values["value1"] = pd.to_numeric(values["value1"], errors='coerce')
values["value2"] = values["value2"].replace("nan", "1")
values["value2"] = pd.to_numeric(values["value2"], errors='coerce').fillna(1)

df["Max speed Knots"] = values["value1"] * values["value2"]

# Extract numeric climb rate from string
df["All eng rate of climb"] = df["All eng rate of climb"].str.extract(r"([\d.]+)").astype(float)

# Convert all columns to float, ignoring errors on non-numeric automatically replaced by NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Drop 'Model Name' column
df.drop("Model Name", axis=1, inplace=True)

df.info()

# Check for any remaining object columns and their unique values
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Unique values in column {col}:")
        print(df[col].unique())
