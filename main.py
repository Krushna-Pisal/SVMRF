import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Global variables
rf_model = None
svm_model = None
scaler = None
ensemble_model = None  # Define ensemble model globally
dataset = None

# GUI Setup
root = tk.Tk()
root.title("UPI Fraud Detection")
root.geometry("600x500")

def load_dataset():
    global dataset
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    dataset = pd.read_csv(file_path)
    messagebox.showinfo("Dataset Loaded", f"Dataset Loaded Successfully\nRows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")

def preprocess_and_train():
    global rf_model, svm_model, scaler, ensemble_model

    if dataset is None:
        messagebox.showerror("Error", "Load the dataset first!")
        return

    # Select features and target
    features = ["Amount", "TransactionFrequency", "UnusualLocation", "UnusualAmount", "NewDevice", "FailedAttempts"]
    target = "FraudFlag"

    X = dataset[features]
    y = dataset[target]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter Tuning for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    rf_model = rf_grid_search.best_estimator_

    # Hyperparameter Tuning for SVM
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_grid_search = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid_search.fit(X_train, y_train)
    svm_model = svm_grid_search.best_estimator_

    # Ensemble Model (Voting Classifier) Combined rf and svm:
    ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model)], voting='soft')
    ensemble_model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    messagebox.showinfo("Training Complete", f"Model Trained Successfully!\nAccuracy: {accuracy:.2f}")

def predict_fraud():
    global ensemble_model

    if ensemble_model is None:
        messagebox.showerror("Error", "Train the model first!")
        return

    # Get user input
    try:
        amount = float(amount_entry.get())
        freq = float(freq_entry.get())
        unusual_loc = int(loc_var.get())
        unusual_amt = int(amt_var.get())
        new_device = int(device_var.get())
        failed_attempts = int(failed_entry.get())

        # Check if failed attempts are more than 10
        if failed_attempts > 10:
            prediction_text = "Fraud"
            root.configure(bg="red")  # Set background to red
        else:
            # Prepare input data for the model prediction
            input_data = np.array([[amount, freq, unusual_loc, unusual_amt, new_device, failed_attempts]])
            input_data_scaled = scaler.transform(input_data)

            # Predict using the ensemble model
            prediction = ensemble_model.predict(input_data_scaled)
            prediction_text = "Fraud" if prediction[0] == 1 else "Legitimate"

            # Change background color based on prediction
            if prediction_text == "Fraud":
                root.configure(bg="red")
            else:
                root.configure(bg="green")

        # Display the result
        result_label.config(text=f"Prediction: {prediction_text}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# GUI Elements
tk.Label(root, text="UPI Fraud Detection", font=("Arial", 16)).pack(pady=10)
tk.Button(root, text="Load Dataset", command=load_dataset).pack(pady=5)
tk.Button(root, text="Train Model", command=preprocess_and_train).pack(pady=5)

# Input Fields
tk.Label(root, text="Transaction Amount:").pack()
amount_entry = tk.Entry(root)
amount_entry.pack()

tk.Label(root, text="Transaction Frequency:").pack()
freq_entry = tk.Entry(root)
freq_entry.pack()

tk.Label(root, text="Unusual Location (0/1):").pack()
loc_var = tk.StringVar(value="0")
tk.OptionMenu(root, loc_var, "0", "1").pack()

tk.Label(root, text="Unusual Amount (0/1):").pack()
amt_var = tk.StringVar(value="0")
tk.OptionMenu(root, amt_var, "0", "1").pack()

tk.Label(root, text="New Device (0/1):").pack()
device_var = tk.StringVar(value="0")
tk.OptionMenu(root, device_var, "0", "1").pack()

tk.Label(root, text="Failed Attempts:").pack()
failed_entry = tk.Entry(root)
failed_entry.pack()

tk.Button(root, text="Predict Fraud", command=predict_fraud).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
