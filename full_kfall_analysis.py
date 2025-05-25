import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re # Regular expression for parsing filenames
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# --- Configuration ---
# Adjust this path if your KFall Dataset folder is not in the same directory as your script
DATASET_ROOT = 'KFall Dataset'
SENSOR_DATA_DIR = os.path.join(DATASET_ROOT, 'sensor_data')
LABEL_DATA_DIR = os.path.join(DATASET_ROOT, 'label_data')

# --- Feature Engineering Parameters ---
WINDOW_SIZE = 100  # Number of frames in each window (e.g., 1 second if 100Hz)
OVERLAP = 50       # Number of overlapping frames between consecutive windows
STEP_SIZE = WINDOW_SIZE - OVERLAP

# Buffer frames around fall events for labeling the window
FALL_BUFFER_FRAMES_BEFORE = 20
FALL_BUFFER_FRAMES_AFTER = 50

# --- Global lists to store all features and labels ---
all_features = []
all_labels = []

# --- 1. Function to process a single sensor file and its labels ---
def process_single_trial(sensor_filepath, trial_label_row):
    """
    Loads a single sensor CSV, extracts features, and labels windows.
    Returns lists of features and labels for this trial.
    """
    try:
        sensor_df = pd.read_csv(sensor_filepath)
    except Exception as e:
        print(f"Error loading sensor file {sensor_filepath}: {e}")
        return [], []

    # Get fall onset and impact frames for this trial
    fall_onset_frame = trial_label_row['Fall_onset_frame']
    fall_impact_frame = trial_label_row['Fall_impact_frame']

    # --- Derived Basic Features ---
    sensor_df['AccMag'] = np.sqrt(sensor_df['AccX']**2 + sensor_df['AccY']**2 + sensor_df['AccZ']**2)
    sensor_df['GyrMag'] = np.sqrt(sensor_df['GyrX']**2 + sensor_df['GyrY']**2 + sensor_df['GyrZ']**2)

    # Estimate Pitch/Roll (simple static approximation)
    sensor_df['Pitch_est'] = np.degrees(np.arctan2(sensor_df['AccY'], np.sqrt(sensor_df['AccX']**2 + sensor_df['AccZ']**2)))
    sensor_df['Roll_est'] = np.degrees(np.arctan2(-sensor_df['AccX'], np.sqrt(sensor_df['AccY']**2 + sensor_df['AccZ']**2)))

    # --- Create 'IsFall' label column for each sensor frame ---
    sensor_df['IsFall'] = 0
    fall_start_window = max(0, fall_onset_frame - FALL_BUFFER_FRAMES_BEFORE)
    fall_end_window = min(len(sensor_df) - 1, fall_impact_frame + FALL_BUFFER_FRAMES_AFTER)

    sensor_df.loc[
        (sensor_df['FrameCounter'] >= fall_start_window) &
        (sensor_df['FrameCounter'] <= fall_end_window),
        'IsFall'
    ] = 1

    trial_features = []
    trial_labels = []

    # --- Feature Extraction over Sliding Windows ---
    for i in range(0, len(sensor_df) - WINDOW_SIZE + 1, STEP_SIZE):
        window_df = sensor_df.iloc[i : i + WINDOW_SIZE]

        if len(window_df) < WINDOW_SIZE: # Should not happen with current range, but good check
            continue

        # Extract features from the current window
        acc_mag_mean = window_df['AccMag'].mean()
        acc_mag_std = window_df['AccMag'].std()
        acc_mag_max = window_df['AccMag'].max()
        acc_mag_min = window_df['AccMag'].min()

        gyr_mag_mean = window_df['GyrMag'].mean()
        gyr_mag_std = window_df['GyrMag'].std()
        gyr_mag_max = window_df['GyrMag'].max()
        gyr_mag_min = window_df['GyrMag'].min()

        # More robust features (e.g., using max-min range for impact, RMS for velocity)
        impact_feature = window_df['AccMag'].max() - window_df['AccMag'].min()
        velocity_feature_rms = np.sqrt(np.mean(window_df['AccMag']**2))

        # Posture features
        posture_pitch_mean = window_df['Pitch_est'].mean()
        posture_roll_mean = window_df['Roll_est'].mean()

        # Orientation ranges (Euler angles indicate rotation/posture change)
        euler_x_range = window_df['EulerX'].max() - window_df['EulerX'].min()
        euler_y_range = window_df['EulerY'].max() - window_df['EulerY'].min()
        euler_z_range = window_df['EulerZ'].max() - window_df['EulerZ'].min()

        # Determine the label for this window
        window_label = 1 if window_df['IsFall'].any() else 0 # If any frame in window is fall, label as fall

        trial_features.append([
            acc_mag_mean, acc_mag_std, acc_mag_max, acc_mag_min,
            gyr_mag_mean, gyr_mag_std, gyr_mag_max, gyr_mag_min,
            euler_x_range, euler_y_range, euler_z_range,
            impact_feature, velocity_feature_rms,
            posture_pitch_mean, posture_roll_mean
        ])
        trial_labels.append(window_label)

    return trial_features, trial_labels

# --- 2. Main Loop: Iterate through Subjects and Trials ---
print("Starting data loading and feature extraction across all subjects...")

# Get list of all subject folders (e.g., 'SA01', 'SA02', ...)
subject_folders = sorted([d for d in os.listdir(SENSOR_DATA_DIR) if os.path.isdir(os.path.join(SENSOR_DATA_DIR, d))])

for subject_folder in subject_folders:
    subject_id_match = re.search(r'SA(\d+)', subject_folder)
    if not subject_id_match:
        print(f"Skipping {subject_folder}: Could not extract subject ID.")
        continue
    subject_num = int(subject_id_match.group(1)) # Extract just the number (e.g., 06)

    # Load corresponding label file for this subject
    label_filepath = os.path.join(LABEL_DATA_DIR, f'SA{subject_num:02d}_label.xlsx')
    if not os.path.exists(label_filepath):
        print(f"Warning: Label file not found for SA{subject_num:02d} at {label_filepath}. Skipping subject.")
        continue

    try:
        subject_label_df = pd.read_excel(label_filepath)
        # Forward fill 'Task Code (Task ID)' and 'Description' if they are NaN,
        # assuming the first row of each task has the full description.
        # This requires careful handling if labels are not perfectly structured.
        # For KFall, typically only the first entry of a series of repeated trials
        # for a specific fall/activity has the description.
        subject_label_df['Task Code (Task ID)'] = subject_label_df['Task Code (Task ID)'].ffill()
        subject_label_df['Description'] = subject_label_df['Description'].ffill()

    except Exception as e:
        print(f"Error loading label file {label_filepath}: {e}. Skipping subject.")
        continue

    print(f"\nProcessing Subject: {subject_folder} (ID: {subject_num})")

    # Get list of sensor CSV files for this subject
    subject_sensor_path = os.path.join(SENSOR_DATA_DIR, subject_folder)
    sensor_files = sorted([f for f in os.listdir(subject_sensor_path) if f.endswith('.csv')])

    for sensor_file in sensor_files:
        # Extract Trial ID from filename (e.g., S06T01R01.csv -> 1)
        trial_id_match = re.search(r'T(\d+)R', sensor_file)
        if not trial_id_match:
            print(f"Skipping {sensor_file}: Could not extract Trial ID.")
            continue
        current_trial_id = int(trial_id_match.group(1))

        # Find the corresponding label row for this trial ID
        trial_label_row = subject_label_df[subject_label_df['Trial ID'] == current_trial_id]

        if trial_label_row.empty:
            # print(f"Warning: No label found for Trial ID {current_trial_id} in {label_filepath}. Skipping sensor file.")
            continue # Skip trials without labels

        # Take the first matching row if multiple exist (shouldn't for trial ID)
        trial_label_row = trial_label_row.iloc[0]

        full_sensor_filepath = os.path.join(subject_sensor_path, sensor_file)
        print(f"  - Processing {sensor_file} (Trial ID: {current_trial_id})")

        features, labels = process_single_trial(full_sensor_filepath, trial_label_row)
        all_features.extend(features)
        all_labels.extend(labels)

print("\n--- Finished Data Loading and Feature Extraction ---")
print(f"Total collected windows: {len(all_features)}")

# --- Combine all features and labels into DataFrames ---
feature_columns = [
    'AccMag_Mean', 'AccMag_Std', 'AccMag_Max', 'AccMag_Min',
    'GyrMag_Mean', 'GyrMag_Std', 'GyrMag_Max', 'GyrMag_Min',
    'EulerX_Range', 'EulerY_Range', 'EulerZ_Range',
    'Impact_Feature', 'Velocity_Feature_RMS',
    'Posture_Pitch_Mean', 'Posture_Roll_Mean'
]
X_all = pd.DataFrame(all_features, columns=feature_columns)
y_all = pd.Series(all_labels, name='IsFall_Window')

print(f"\nShape of combined X_all: {X_all.shape}")
print(f"Shape of combined y_all: {y_all.shape}")
print("\nLabel distribution in y_all (before splitting):")
print(y_all.value_counts())
print(f"Fall ratio: {y_all.sum() / len(y_all):.4f}")

# --- 3. Machine Learning Model Training and Evaluation ---
if not X_all.empty and not y_all.empty:
    print("\n--- Starting Machine Learning Model Training ---")

    # Split Data into Training and Testing Sets
    # stratify=y_all is crucial for imbalanced datasets
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)

    print(f"\nData Split:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"y_train label distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"y_test label distribution:\n{y_test.value_counts(normalize=True)}")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Machine Learning Model (Support Vector Machine)
    # class_weight='balanced' is added to handle class imbalance
    print("\nTraining SVM Model with class_weight='balanced'...")
    model = SVC(random_state=42, kernel='rbf', C=10, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    model_filename = 'svm_fall_detection_model.joblib'
    scaler_filename = 'scaler_for_fall_detection.joblib'

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"\nModel saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

    # Model Evaluation
    y_pred = model.predict(X_test_scaled)

    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0) # zero_division=0 to handle cases where no positive predictions
    recall = recall_score(y_test, y_pred, pos_label=1) # Sensitivity
    specificity = recall_score(y_test, y_pred, pos_label=0) # Specificity
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision (for Fall): {precision:.4f}")
    print(f"Recall/Sensitivity (for Fall): {recall:.4f}")
    print(f"Specificity (for Non-Fall): {specificity:.4f}")
    print(f"F1-Score (for Fall): {f1:.4f}")

else:
    print("\nNo data collected. Please check dataset paths and file structure.")

print("\nProcess finished.")