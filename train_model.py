import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("dataset/irrigation_labeled.csv")
print("Dataset Columns:",df.columns.tolist())

# 2. Clean column names
df.columns = df.columns.str.strip().str.lower()

# 3. Select input features
print("\nAVAILABLE COLUMNS:")
print(df.columns.tolist())

X = df[[
    "temperature [_ c]",
    "humidity [%]",
    "soil moisture",
    "reference evapotranspiration",
    "evapotranspiration"
]]

y = df["irrigation_needed"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 7. Train model
model.fit(X_train, y_train)

# 8. Predict test data
y_pred = model.predict(X_test)

# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:", accuracy)

# 10. Display classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# 11. Save model
joblib.dump(model, "irrigation_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")
