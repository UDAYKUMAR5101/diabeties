import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ------------------------
# Create folder to save models
# ------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'diabetes.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------
# Load dataset
# ------------------------
def train():
    df = pd.read_csv(DATASET_PATH)

# ------------------------
# Define features (X) and target (y)
# ------------------------
    features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    X = df[features]
    y = df['Outcome']  # Target column

# ------------------------
# Handle missing values (important for numeric dataset)
# ------------------------
    num_imputer = SimpleImputer(strategy='median')
    X = num_imputer.fit_transform(X)

# ------------------------
# Train-test split
# ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# ------------------------
# Balance classes with SMOTE
# ------------------------
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ------------------------
# Train Gradient Boosting Model
# ------------------------
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

# ------------------------
# Evaluate metrics
# ------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("Model Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# ------------------------
# Save model + imputer
# ------------------------
    joblib.dump(model, os.path.join(MODELS_DIR, 'diabetes_gb_model.pkl'))
    joblib.dump(num_imputer, os.path.join(MODELS_DIR, 'num_imputer.pkl'))

    print("Gradient Boosting model trained and saved successfully.")


if __name__ == "__main__":
    train()
