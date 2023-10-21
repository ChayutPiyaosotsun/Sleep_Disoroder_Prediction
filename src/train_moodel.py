import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
features = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps']

# Encoding categorical variables
label_encoders = {}
for column in ['Gender', 'Occupation', 'BMI Category']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
    
# Train-test split
X = data[features]
y = data['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance
# Drop or fill any missing values
y_train = y_train.fillna("None")
# Convert y_train to string if it's mixed
y_train = y_train.astype(str)
# Encode the target variable if it's categorical
le = LabelEncoder()
y_train = le.fit_transform(y_train)
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Model training
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train_resampled, y_train_resampled)
# Save the trained model
joblib.dump(clf, 'models/sleep_disorder_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/encoder.pkl')

print("Model trained and saved!")