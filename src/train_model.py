import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Constants
DATA_PATH = 'data/Sleep_health_and_lifestyle_dataset.csv'
MODEL_PATH = 'models/sleep_disorder_model.pkl'
LABEL_ENCODERS_PATH = 'models/label_encoders.pkl'
SCALER_PATH = 'models/scaler.pkl'
TARGET_ENCODER_PATH = 'models/encoder.pkl'

def load_data():
    """Load dataset and return dataframe."""
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except FileNotFoundError:
        print("Error: Data file not found.")
        exit()

def encode_categorical_features(data, columns):
    """Encode categorical features and return updated dataframe and label encoders."""
    label_encoders = {}
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

def handle_class_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE and return resampled data."""
    y_train = y_train.fillna("None").astype(str)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled, le

def train_and_save_model(X_train, y_train):
    """Train the model and save it along with encoders and scalers."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    print("Model trained and saved!")

def main():
    data = load_data()
    
    # Check for missing values
    if data.isnull().sum().any():
        print("Warning: Missing values detected. Consider preprocessing the data to handle them.")

    features = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps']
    data, label_encoders = encode_categorical_features(data, ['Gender', 'Occupation', 'BMI Category'])
    
    X = data[features]
    y = data['Sleep Disorder']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_resampled, y_train_resampled, target_encoder = handle_class_imbalance(X_train, y_train)
    
    train_and_save_model(X_train_resampled, y_train_resampled)

    # Save additional resources
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)

if __name__ == "__main__":
    main()
