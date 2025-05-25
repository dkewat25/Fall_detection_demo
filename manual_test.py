import pandas as pd
import numpy as np
import joblib
import os
import re

# --- Configuration (MUST MATCH your training script) ---
WINDOW_SIZE = 100
OVERLAP = 50
STEP_SIZE = WINDOW_SIZE - OVERLAP
FALL_BUFFER_FRAMES_BEFORE = 20
FALL_BUFFER_FRAMES_AFTER = 50

# --- File Paths for Saved Model and Scaler ---
MODEL_PATH = 'svm_fall_detection_model.joblib'
SCALER_PATH = 'scaler_for_fall_detection.joblib'

# --- Load the Trained Model and Scaler ---
try:
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or scaler files not found. Make sure '{MODEL_PATH}' and '{SCALER_PATH}' exist in the same directory.")
    print("Please run full_kfall_analysis.py first to train and save them.")
    exit() # Exit if files are not found

# --- Function to preprocess a single sensor file for prediction ---
def preprocess_for_prediction(sensor_filepath):
    """
    Loads a single sensor CSV, extracts features using the same logic as training.
    Returns a DataFrame of features ready for prediction.
    """
    try:
        sensor_df = pd.read_csv(sensor_filepath)
    except Exception as e:
        print(f"Error loading sensor file {sensor_filepath}: {e}")
        return pd.DataFrame()

    # Derived Basic Features (MUST BE THE SAME AS TRAINING)
    sensor_df['AccMag'] = np.sqrt(sensor_df['AccX']**2 + sensor_df['AccY']**2 + sensor_df['AccZ']**2)
    sensor_df['GyrMag'] = np.sqrt(sensor_df['GyrX']**2 + sensor_df['GyrY']**2 + sensor_df['GyrZ']**2)
    sensor_df['Pitch_est'] = np.degrees(np.arctan2(sensor_df['AccY'], np.sqrt(sensor_df['AccX']**2 + sensor_df['AccZ']**2)))
    sensor_df['Roll_est'] = np.degrees(np.arctan2(-sensor_df['AccX'], np.sqrt(sensor_df['AccY']**2 + sensor_df['AccZ']**2)))

    trial_features = []
    window_start_frames = [] # To keep track of where each window starts

    # Feature Extraction over Sliding Windows (MUST BE THE SAME AS TRAINING)
    for i in range(0, len(sensor_df) - WINDOW_SIZE + 1, STEP_SIZE):
        window_df = sensor_df.iloc[i : i + WINDOW_SIZE]
        window_start_frames.append(window_df['FrameCounter'].iloc[0])

        # Extract features from the current window
        acc_mag_mean = window_df['AccMag'].mean()
        acc_mag_std = window_df['AccMag'].std()
        acc_mag_max = window_df['AccMag'].max()
        acc_mag_min = window_df['AccMag'].min()

        gyr_mag_mean = window_df['GyrMag'].mean()
        gyr_mag_std = window_df['GyrMag'].std()
        gyr_mag_max = window_df['GyrMag'].max()
        gyr_mag_min = window_df['GyrMag'].min()

        impact_feature = window_df['AccMag'].max() - window_df['AccMag'].min()
        velocity_feature_rms = np.sqrt(np.mean(window_df['AccMag']**2))

        posture_pitch_mean = window_df['Pitch_est'].mean()
        posture_roll_mean = window_df['Roll_est'].mean()

        euler_x_range = window_df['EulerX'].max() - window_df['EulerX'].min()
        euler_y_range = window_df['EulerY'].max() - window_df['EulerY'].min()
        euler_z_range = window_df['EulerZ'].max() - window_df['EulerZ'].min()

        trial_features.append([
            acc_mag_mean, acc_mag_std, acc_mag_max, acc_mag_min,
            gyr_mag_mean, gyr_mag_std, gyr_mag_max, gyr_mag_min,
            euler_x_range, euler_y_range, euler_z_range,
            impact_feature, velocity_feature_rms,
            posture_pitch_mean, posture_roll_mean
        ])

    feature_columns = [
        'AccMag_Mean', 'AccMag_Std', 'AccMag_Max', 'AccMag_Min',
        'GyrMag_Mean', 'GyrMag_Std', 'GyrMag_Max', 'GyrMag_Min',
        'EulerX_Range', 'EulerY_Range', 'EulerZ_Range',
        'Impact_Feature', 'Velocity_Feature_RMS',
        'Posture_Pitch_Mean', 'Posture_Roll_Mean'
    ]
    return pd.DataFrame(trial_features, columns=feature_columns), window_start_frames

# --- Manual Test Demonstration ---
if __name__ == "__main__":
    print("\n--- Manual Fall Detection Demonstration ---")

    # Example: Specify a path to a sensor data CSV you want to test
    # You can pick any file from your KFall Dataset/sensor_data/SAxx/ folder
    # For instance, a fall trial (e.g., S06T01R01.csv) or a non-fall trial (e.g., S06T02R01.csv)
    # Ensure this path is correct for your system
    example_sensor_file = 'KFall Dataset/sensor_data/SA06/S06T01R01.csv' # Example: A fall trial
    #example_sensor_file = 'KFall Dataset/sensor_data/SA06/S06T02R01.csv' # Example: A non-fall trial

    if not os.path.exists(example_sensor_file):
        print(f"Error: Example sensor file not found at {example_sensor_file}")
        print("Please adjust 'example_sensor_file' variable to a valid path on your system.")
    else:
        print(f"Analyzing sensor data from: {example_sensor_file}")

        # 1. Preprocess the new data
        X_new_data, window_starts = preprocess_for_prediction(example_sensor_file)

        if X_new_data.empty:
            print("Failed to preprocess data. Exiting.")
        else:
            print(f"Processed {len(X_new_data)} windows from the file.")

            # 2. Scale the new data using the loaded scaler
            X_new_data_scaled = loaded_scaler.transform(X_new_data)

            # 3. Make predictions
            predictions = loaded_model.predict(X_new_data_scaled)

            # 4. Interpret Results
            fall_detected_windows = np.where(predictions == 1)[0]

            print("\n--- Prediction Results ---")
            if len(fall_detected_windows) > 0:
                print(f"Fall event(s) detected in the following windows (based on window index):")
                for window_idx in fall_detected_windows:
                    print(f"  - Window Index: {window_idx}, starting at FrameCounter: {window_starts[window_idx]}")
                print(f"\nOverall, the model PREDICTS a fall for this sequence.")
            else:
                print("No fall event detected in any window.")
                print("Overall, the model PREDICTS NO fall for this sequence.")

            # Optional: Detailed predictions for each window
            # print("\nDetailed window predictions (0=Non-Fall, 1=Fall):")
            # for i, pred in enumerate(predictions):
            #     print(f"Window {i} (Starts at FrameCounter {window_starts[i]}): Prediction = {pred}")

            print("\nDemonstration complete.")