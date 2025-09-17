import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve

# Create folder to save models
os.makedirs('models', exist_ok=True)
# Load dataset using a direct absolute path
df = pd.read_csv(r'C:\Users\harir\Desktop\Diabeties\dataset\diabetes_dataset_with_prediction.csv')

# Define features (X) and target (y)
features = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
    'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity'
]
X = df[features]
y = df['class']  # Target column: Positive / Negative

# Label encode categorical features
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predictions on test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of positive class

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("✅ Model Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save model and encoders
joblib.dump(model, 'app/diabetes_gb_model.pkl')
joblib.dump(label_encoders, 'app/label_encoders.pkl')
joblib.dump(target_encoder, 'app/target_encoder.pkl')

print("✅ Gradient Boosting model trained and saved successfully.")























'''
-------------------------------------------------------------------------------

import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # To handle class imbalance

# Create folder to save models
os.makedirs('models', exist_ok=True)

# Load dataset
df = pd.read_csv('dataset/diabetes_dataset_with_prediction.csv')

# Define features (X) and target (y)
features = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
    'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity'
]
X = df[features]
y = df['class']  # Target column: Positive / Negative

# Convert binary categorical features to 0/1
binary_cols = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
               'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
               'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']

for col in binary_cols:
    X[col] = X[col].map({'yes':1, 'no':0, 'Male':1, 'Female':0})

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split dataset with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_res, y_train_res)

# Predictions on test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of positive class

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("✅ Model Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Save model and encoders
joblib.dump(model, 'models/diabetes_xgb_model.pkl')
joblib.dump(target_encoder, 'models/target_encoder.pkl')

print("✅ XGBoost model trained and saved successfully.")

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ------------------------
# Create folder to save models
# ------------------------
os.makedirs('models', exist_ok=True)

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv('dataset/diabetes_dataset_with_prediction.csv')

# ------------------------
# Define features (X) and target (y)
# ------------------------
features = [
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
    'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity'
]
X = df[features]
y = df['class']  # Target column: Positive / Negative

# ------------------------
# Separate numeric and categorical columns
# ------------------------
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# ------------------------
# Impute missing values
# ------------------------
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# ------------------------
# Encode categorical features
# ------------------------
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ------------------------
# Split dataset
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

print("✅ Model Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# ------------------------
# Save model and encoders
# ------------------------
joblib.dump(model, 'app/diabetes_gb_model.pkl')
joblib.dump(label_encoders, 'app/label_encoders.pkl')
joblib.dump(target_encoder, 'app/target_encoder.pkl')
joblib.dump(num_imputer, 'app/num_imputer.pkl')
joblib.dump(cat_imputer, 'app/cat_imputer.pkl')

print("✅ Gradient Boosting model trained and saved successfully.")
'''