import pandas as pd
import joblib

# Constants
MODEL_PATH = 'models/sleep_disorder_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
LABEL_ENCODERS_PATH = 'models/label_encoders.pkl'
TARGET_ENCODER_PATH = 'models/encoder.pkl'

def load_resources():
    """Load and return models and encoders."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    
    return model, scaler, label_encoders, target_encoder

def prepare_data(data, scaler, label_encoders):
    """Encode categorical features and scale numerical features."""
    for column, encoder in label_encoders.items():
        data[column] = encoder.transform(data[column])
    
    data_scaled = scaler.transform(data)
    return data_scaled

def make_prediction(data, model, scaler, label_encoders, target_encoder):
    """Predict the sleep disorder based on user input."""
    preprocessed_data = prepare_data(data, scaler, label_encoders)
    predictions = model.predict(preprocessed_data)
    
    decoded_predictions = target_encoder.inverse_transform(predictions)
    return decoded_predictions

def get_input(prompt, dtype=str, valid_values=None):
    """Prompt user for input and validate it."""
    while True:
        try:
            # Get input from user
            value = dtype(input(prompt))

            # If valid_values is provided, check if the input is one of the valid values
            if valid_values and value not in valid_values:
                raise ValueError(f"Invalid input. Allowed values are: {', '.join(map(str, valid_values))}")

            return value

        except ValueError as e:
            print(f"Error: {e}. Please try again.")

def get_user_input():
    """Prompt the user for required input data."""
    gender = get_input("Gender (Male/Female): ", valid_values=['Male', 'Female'])
    age = get_input("Age (in years): ", dtype=int)
    occupation = get_input("Occupation: ", valid_values=['Scientist', 'Software Engineer', 'Engineer', 'Doctor', 'Nurse', 'Sales Representative', 'Salesperson', 'Accountant', 'Teacher', 'Lawyer'])
    sleep_duration = get_input("Sleep Duration (hours): ", dtype=float)
    quality_of_sleep = get_input("Quality of Sleep (scale: 1-10): ", dtype=int, valid_values=list(range(1,11)))
    physical_activity_level = get_input("Physical Activity Level (minutes/day): ", dtype=int)
    stress_level = get_input("Stress Level (scale: 1-10): ", dtype=int, valid_values=list(range(1,11)))
    bmi_category = get_input("BMI Category (Underweight, Normal, Overweight): ", valid_values=['Underweight', 'Normal', 'Overweight'])
    heart_rate = get_input("Heart Rate (bpm): ", dtype=int)
    daily_steps = get_input("Daily Steps: ", dtype=int)

    # Construct the DataFrame using the input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity_level],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_category],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps]
    })
    return input_data

if __name__ == "__main__":
    model, scaler, label_encoders, target_encoder = load_resources()

    print("Enter the details to predict Sleep Disorder:")
    input_data = get_user_input()

    prediction = make_prediction(input_data, model, scaler, label_encoders, target_encoder)
    print("-------------------------------\n")
    print("Predicted Sleep Disorder:", prediction[0])
    print("\n-------------------------------")
