# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1️⃣ Load dataset
df = pd.read_csv("heart.csv")

# 2️⃣ Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# 3️⃣ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 6️⃣ Make predictions
y_pred = model.predict(X_test_scaled)

# 7️⃣ Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Save model and scaler
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("💾 Model and Scaler saved successfully!")
