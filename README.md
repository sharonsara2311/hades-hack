# hades-hack
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Load and clean the data
file_path = '/mnt/data/general_disease_diagnosis.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=["Patient_Name"]).dropna(subset=["Disease"])

# Feature engineering: calculate BMI
data["BMI"] = data["Weight_kg"] / ((data["Height_cm"] / 100) ** 2)

# Encode the target variable
label_encoder = LabelEncoder()
data["Disease"] = label_encoder.fit_transform(data["Disease"])

# Define features and target
X = data.drop(columns=["Disease"])
y = data["Disease"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance by calculating class weights
class_weights = compute_class_weight(class_weight='balanced', classes=label_encoder.classes_, y=y)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Initialize and train the XGBoost model with class weights
model = XGBClassifier(scale_pos_weight=class_weights_dict, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", confusion_mat)
