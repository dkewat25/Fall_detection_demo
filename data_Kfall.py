import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os # For navigating file paths

# --- Configuration ---
# Assuming your KFall Dataset is in a parent directory or known path
# Adjust these paths based on where you extract your zip file
DATASET_ROOT = 'KFall Dataset'
SENSOR_DATA_PATH = os.path.join(DATASET_ROOT, 'sensor_data', 'SA06')
LABEL_DATA_PATH = os.path.join(DATASET_ROOT, 'label_data')

# Example file names
EXAMPLE_SENSOR_FILE = os.path.join(SENSOR_DATA_PATH, 'S06T01R01.csv')
EXAMPLE_LABEL_FILE = os.path.join(LABEL_DATA_PATH, 'SA06_label.xlsx')

print(f"Attempting to load sensor data from: {EXAMPLE_SENSOR_FILE}")
print(f"Attempting to load label data from: {EXAMPLE_LABEL_FILE}")

# --- 1. Load Sensor Data ---
try:
    sensor_df = pd.read_csv(EXAMPLE_SENSOR_FILE)
    print("\n--- Sensor Data (S06T01R01.csv) ---")
    print("Shape:", sensor_df.shape)
    print("Columns:", sensor_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(sensor_df.head())
    print("\nData Types:")
    print(sensor_df.info()) # Provides non-null count and dtypes

except FileNotFoundError:
    print(f"Error: Sensor file not found at {EXAMPLE_SENSOR_FILE}. Please check your path.")
    sensor_df = None
except Exception as e:
    print(f"An error occurred loading sensor data: {e}")
    sensor_df = None


# --- 2. Load Label Data ---
try:
    # When loading an Excel file, you might need to specify the sheet name
    # If the labels are all in one sheet, typically the first one is 'Sheet1' or left blank
    label_df = pd.read_excel(EXAMPLE_LABEL_FILE)
    print("\n--- Label Data (SA06_label.xlsx) ---")
    print("Shape:", label_df.shape)
    print("Columns:", label_df.columns.tolist())
    print("\nFirst 5 rows:")
    print(label_df.head())
    print("\nData Types:")
    print(label_df.info())

    # Find the specific label for Trial ID = 1 (matching S06T01R01.csv)
    current_trial_id = 1 # From S06T01R01.csv filename
    trial_label = label_df[label_df['Trial ID'] == current_trial_id].iloc[0] # .iloc[0] to get the first matching row

    if not trial_label.empty:
        print(f"\nLabels for Trial ID {current_trial_id}:")
        print(f"Task: {trial_label['Task Code (Task ID)']} - {trial_label['Description']}")
        print(f"Fall Onset Frame: {trial_label['Fall_onset_frame']}")
        print(f"Fall Impact Frame: {trial_label['Fall_impact_frame']}")
    else:
        print(f"No label found for Trial ID {current_trial_id}")


except FileNotFoundError:
    print(f"Error: Label file not found at {EXAMPLE_LABEL_FILE}. Please check your path.")
    label_df = None
except Exception as e:
    print(f"An error occurred loading label data: {e}")
    label_df = None

# --- Initial Data Quality Check (if data loaded successfully) ---
if sensor_df is not None:
    print("\n--- Sensor Data Missing Values ---")
    print(sensor_df.isnull().sum())

if label_df is not None:
    print("\n--- Label Data Missing Values ---")
    print(label_df.isnull().sum())


if sensor_df is not None and label_df is not None:
    # Get the fall onset and impact frames for Trial ID 1
    current_trial_id = 1
    try:
        trial_label_row = label_df[label_df['Trial ID'] == current_trial_id].iloc[0]
        fall_onset_frame = trial_label_row['Fall_onset_frame']
        fall_impact_frame = trial_label_row['Fall_impact_frame']
        description = trial_label_row['Description']

        print(f"\nVisualizing Sensor Data for Trial {current_trial_id} ('{description}')")
        print(f"Fall Onset: Frame {fall_onset_frame}, Fall Impact: Frame {fall_impact_frame}")

        # Plot Accelerometer Data
        plt.figure(figsize=(15, 6))
        plt.plot(sensor_df['FrameCounter'], sensor_df['AccX'], label='AccX')
        plt.plot(sensor_df['FrameCounter'], sensor_df['AccY'], label='AccY')
        plt.plot(sensor_df['FrameCounter'], sensor_df['AccZ'], label='AccZ')

        # Mark Fall Onset and Impact
        plt.axvline(x=fall_onset_frame, color='red', linestyle='--', label='Fall Onset')
        plt.axvline(x=fall_impact_frame, color='green', linestyle='--', label='Fall Impact')

        plt.title(f'Accelerometer Data for Trial {current_trial_id}: {description}')
        plt.xlabel('Frame Counter')
        plt.ylabel('Acceleration (units)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Gyroscope Data
        plt.figure(figsize=(15, 6))
        plt.plot(sensor_df['FrameCounter'], sensor_df['GyrX'], label='GyrX')
        plt.plot(sensor_df['FrameCounter'], sensor_df['GyrY'], label='GyrY')
        plt.plot(sensor_df['FrameCounter'], sensor_df['GyrZ'], label='GyrZ')

        # Mark Fall Onset and Impact
        plt.axvline(x=fall_onset_frame, color='red', linestyle='--', label='Fall Onset')
        plt.axvline(x=fall_impact_frame, color='green', linestyle='--', label='Fall Impact')

        plt.title(f'Gyroscope Data for Trial {current_trial_id}: {description}')
        plt.xlabel('Frame Counter')
        plt.ylabel('Angular Velocity (units)')
        plt.legend()
        plt.grid(True)
        plt.show()

    except IndexError:
        print(f"Warning: No label found for Trial ID {current_trial_id} in the label file for plotting.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")


if sensor_df is not None:
    # --- Basic Feature: Accelerometer Vector Magnitude (SVM or VMag) ---
    sensor_df['AccMag'] = np.sqrt(sensor_df['AccX']**2 + sensor_df['AccY']**2 + sensor_df['AccZ']**2)
    # The '1g' reference is often used to normalize acceleration. If your Acc is in 'g' units, 1g=9.8m/s^2.
    # If not, you might need to determine the unit or a typical baseline.

    # --- Basic Feature: Gyroscope Vector Magnitude ---
    sensor_df['GyrMag'] = np.sqrt(sensor_df['GyrX']**2 + sensor_df['GyrY']**2 + sensor_df['GyrZ']**2)

    # --- Feature: Velocity (Integration of Acceleration) ---
    # This is more complex and typically done over specific intervals.
    # For a simple approximation over a window, you might look at change in position.
    # A common proxy for 'change in velocity' or 'impact' is the change in acceleration.
    # Or, the peak of the derivative of acceleration magnitude.
    # A simple approach for impact: Peak AccMag during a short window.

    # --- Feature: Impact (e.g., Peak Acceleration Magnitude) ---
    # This often refers to the maximum value of the acceleration magnitude during a brief period.
    # We'll calculate it later over a sliding window.

    # --- Feature: Posture (e.g., based on static acceleration component) ---
    # For posture, you need to separate gravity (static acceleration) from body movement (dynamic acceleration).
    # In a static state, the accelerometer measures gravity.
    # The tilt angles can be derived from the orientation of the gravity vector relative to the sensor's axes.
    # Euler angles might already provide some form of posture information.
    # If not, you can calculate angles like pitch and roll from AccX, AccY, AccZ assuming sensor is relatively stable.
    # Example for pitch (rotation around X-axis): atan2(AccY, sqrt(AccX^2 + AccZ^2))
    # Example for roll (rotation around Y-axis): atan2(-AccX, sqrt(AccY^2 + AccZ^2))

    # Let's add simple pitch/roll estimation if Euler angles are unreliable or for comparison
    # These formulas assume sensor is roughly aligned with body axes for static posture
    # and a stationary state for accurate gravity vector isolation.
    sensor_df['Pitch_est'] = np.degrees(np.arctan2(sensor_df['AccY'], np.sqrt(sensor_df['AccX']**2 + sensor_df['AccZ']**2)))
    sensor_df['Roll_est'] = np.degrees(np.arctan2(-sensor_df['AccX'], np.sqrt(sensor_df['AccY']**2 + sensor_df['AccZ']**2)))


    print("\n--- Sensor Data with Basic Derived Features ---")
    print(sensor_df[['AccX', 'AccY', 'AccZ', 'AccMag', 'GyrMag', 'Pitch_est', 'Roll_est']].head())

    # Example: Plot AccMag to see overall motion
    plt.figure(figsize=(15, 4))
    plt.plot(sensor_df['FrameCounter'], sensor_df['AccMag'], label='AccMag')
    if 'fall_onset_frame' in locals() and 'fall_impact_frame' in locals():
        plt.axvline(x=fall_onset_frame, color='red', linestyle='--', label='Fall Onset')
        plt.axvline(x=fall_impact_frame, color='green', linestyle='--', label='Fall Impact')
    plt.title('Accelerometer Magnitude (AccMag)')
    plt.xlabel('Frame Counter')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if sensor_df is not None and 'fall_onset_frame' in locals() and 'fall_impact_frame' in locals():
    # --- Define Windowing Parameters ---
    # These values will need tuning based on your specific dataset characteristics and sampling rate
    WINDOW_SIZE = 100  # e.g., 100 frames, if sensor is 100Hz, this is 1 second
    OVERLAP = 50       # e.g., 50 frames overlap (50% overlap)
    STEP_SIZE = WINDOW_SIZE - OVERLAP

    # --- Create a 'Fall' label column for each sensor frame ---
    # Initialize all frames as non-fall (0)
    sensor_df['IsFall'] = 0

    # Mark frames between onset and impact as 'Fall' (1)
    # Consider a buffer around the fall event for robustness
    # This range might need to be adjusted based on visual inspection of fall patterns
    FALL_BUFFER_FRAMES_BEFORE = 20 # How many frames before onset to consider part of fall
    FALL_BUFFER_FRAMES_AFTER = 50  # How many frames after impact to consider part of fall

    fall_start_window = max(0, fall_onset_frame - FALL_BUFFER_FRAMES_BEFORE)
    fall_end_window = min(len(sensor_df) - 1, fall_impact_frame + FALL_BUFFER_FRAMES_AFTER)

    sensor_df.loc[
        (sensor_df['FrameCounter'] >= fall_start_window) &
        (sensor_df['FrameCounter'] <= fall_end_window),
        'IsFall'
    ] = 1

    print(f"\nTotal frames labeled as 'Fall': {sensor_df['IsFall'].sum()}")
    print(f"Fall region: Frames {fall_start_window} to {fall_end_window}")

    # --- Feature Extraction over Sliding Windows ---
    features_list = []
    labels_list = []

    for i in range(0, len(sensor_df) - WINDOW_SIZE + 1, STEP_SIZE):
        window_df = sensor_df.iloc[i : i + WINDOW_SIZE]

        # Ensure the window has enough data
        if len(window_df) < WINDOW_SIZE:
            continue

        # --- Extract features from the current window ---
        # Example features (you will expand on these)
        # Statistics of acceleration magnitudes
        acc_mag_mean = window_df['AccMag'].mean()
        acc_mag_std = window_df['AccMag'].std()
        acc_mag_max = window_df['AccMag'].max()
        acc_mag_min = window_df['AccMag'].min()

        # Statistics of gyroscope magnitudes
        gyr_mag_mean = window_df['GyrMag'].mean()
        gyr_mag_std = window_df['GyrMag'].std()
        gyr_mag_max = window_df['GyrMag'].max()
        gyr_mag_min = window_df['GyrMag'].min()

        # Range of Euler angles (for posture change)
        euler_x_range = window_df['EulerX'].max() - window_df['EulerX'].min()
        euler_y_range = window_df['EulerY'].max() - window_df['EulerY'].min()
        euler_z_range = window_df['EulerZ'].max() - window_df['EulerZ'].min()

        # "Impact" related features (e.g., peak acceleration, VSR - Vector Sum Resultant)
        # Peak of AccMag, or difference from mean/median
        impact_feature = window_df['AccMag'].max() - window_df['AccMag'].min() # Simple peak-to-peak
        # More advanced: calculate VSR or rate of change of VMag

        # "Velocity" related features
        # Integral of AccMag over the window, or RMS (Root Mean Square)
        velocity_feature_rms = np.sqrt(np.mean(window_df['AccMag']**2))

        # "Posture" related features (mean or range of Euler/Pitch/Roll)
        posture_pitch_mean = window_df['Pitch_est'].mean()
        posture_roll_mean = window_df['Roll_est'].mean()


        # --- Determine the label for this window ---
        # If any part of the window is labeled as 'Fall' (1), then the window is a 'Fall' window
        window_label = 1 if window_df['IsFall'].any() else 0

        features_list.append([
            acc_mag_mean, acc_mag_std, acc_mag_max, acc_mag_min,
            gyr_mag_mean, gyr_mag_std, gyr_mag_max, gyr_mag_min,
            euler_x_range, euler_y_range, euler_z_range,
            impact_feature, velocity_feature_rms,
            posture_pitch_mean, posture_roll_mean
        ])
        labels_list.append(window_label)

    # Create a DataFrame for features and labels
    feature_columns = [
        'AccMag_Mean', 'AccMag_Std', 'AccMag_Max', 'AccMag_Min',
        'GyrMag_Mean', 'GyrMag_Std', 'GyrMag_Max', 'GyrMag_Min',
        'EulerX_Range', 'EulerY_Range', 'EulerZ_Range',
        'Impact_Feature', 'Velocity_Feature_RMS',
        'Posture_Pitch_Mean', 'Posture_Roll_Mean'
    ]
    X = pd.DataFrame(features_list, columns=feature_columns)
    y = pd.Series(labels_list, name='IsFall_Window')

    print(f"\n--- Features (X) and Labels (y) for ML ---")
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("\nFirst 5 rows of X:")
    print(X.head())
    print("\nLabel distribution in y:")
    print(y.value_counts())
    print(f"Fall ratio: {y.sum() / len(y):.2f}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # Support Vector Machine
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

if 'X' in locals() and 'y' in locals() and X is not None and y is not None:
    # --- Split Data into Training and Testing Sets ---
    # It's crucial to split your data to evaluate the model on unseen data
    # stratify=y is important for imbalanced datasets (e.g., fewer fall events)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"\n--- Data Split for ML ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"y_train label distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"y_test label distribution:\n{y_test.value_counts(normalize=True)}")

    # --- Feature Scaling ---
    # Standardize features (mean=0, std=1) for algorithms sensitive to feature scales (like SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train a Machine Learning Model (e.g., Support Vector Machine) ---
    print("\n--- Training SVM Model ---")
    model = SVC(random_state=42, kernel='rbf', C=10) # RBF kernel, C is regularization parameter
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # --- Model Evaluation ---
    y_pred = model.predict(X_test_scaled)

    print("\n--- Model Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fall', 'Fall']))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1) # Sensitivity
    specificity = recall_score(y_test, y_pred, pos_label=0) # Specificity
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision (for Fall): {precision:.4f}")
    print(f"Recall/Sensitivity (for Fall): {recall:.4f}")
    print(f"Specificity (for Non-Fall): {specificity:.4f}")
    print(f"F1-Score (for Fall): {f1:.4f}")

    # You would iterate this process, try different models (Logistic Regression, Decision Tree, etc.)
    # and tune their hyperparameters to improve performance, especially Recall and Specificity.