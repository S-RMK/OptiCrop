import os
import requests
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download dataset if not present
dataset_filename = "Crop_recommendation.csv"
dataset_url = "https://raw.githubusercontent.com/arzzahid66/Optimizing_Agricultural_Production/master/Crop_recommendation.csv"

if not os.path.exists(dataset_filename):
    print(f"Downloading {dataset_filename} from public repository...")
    try:
        response = requests.get(dataset_url)
        response.raise_for_status()
        with open(dataset_filename, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit(1)

# Load Dataset
print("Loading dataset...")
df = pd.read_csv(dataset_filename)

print(f"Dataset shape: {df.shape}")

# Handling Missing Values
print("Handling missing values...")
for col in df.select_dtypes(include=np.number):
    # fillna inline without inplace=True to support newer pandas versions safely
    df[col] = df[col].fillna(df[col].mean())

# Outlier Removal using IQR
print("Removing outliers...")
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print(f"Shape after outlier removal: {df.shape}")

# Feature and Target Separation
X = df.drop('label', axis=1)
y = df['label']

# Label Encoding
print("Encoding target labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature Scaling
print("Scaling input features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Train Logistic Regression
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save outputs
print("Saving artifacts...")
with open("best_crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved best_crop_model.pkl")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler.pkl")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Saved label_encoder.pkl")

print("Training process finished successfully.")
