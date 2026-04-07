import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
data = pd.read_csv("telco_realistic_2000.csv")

# Convert target
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

# Encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model with imbalance handling
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model + columns
with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("Model saved successfully!")