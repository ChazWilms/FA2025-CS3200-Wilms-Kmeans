import pandas as pd
from sklearn.datasets import make_blobs
import numpy as np

# --- Synthetic Data Generation ---
n_samples = 100000
n_features = 10
n_centers = 5

print(f"Generating synthetic data with {n_samples} samples, {n_features} features, and {n_centers} centers...")
X_synthetic, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=42)
df_synthetic = pd.DataFrame(X_synthetic)
df_synthetic.to_csv("synthetic_data.csv", header=False, index=False)
print(f"Synthetic data saved to synthetic_data.csv. Rows generated: {len(df_synthetic)}")

# --- Real Data Preparation ---
real_data_filename = "Customer_Segmentation.csv"
try:
    print(f"\nProcessing real data from {real_data_filename}...")
    df_real = pd.read_csv(real_data_filename)

    # Filter to keep only numerical columns: Age, Income, SpendingScore
    # IMPORTANT: Adjust these column names if they are different in your CSV file
    # You might want to inspect your CSV file first to confirm column names.
    numerical_cols = ['Age', 'Income Level', 'Coverage Amount', 'Premium Amount']
    
    # Check if all specified numerical columns exist
    missing_cols = [col for col in numerical_cols if col not in df_real.columns]
    if missing_cols:
        print(f"Warning: The following specified numerical columns were not found in {real_data_filename}: {missing_cols}")
        print("Attempting to proceed with available columns.")
        numerical_cols = [col for col in numerical_cols if col in df_real.columns]
        if not numerical_cols:
            raise ValueError(f"No specified numerical columns found in {real_data_filename}.")

    df_real_filtered = df_real[numerical_cols]

    # Drop rows with missing values
    df_real_filtered = df_real_filtered.dropna()

    df_real_filtered.to_csv("large_real_data.csv", header=False, index=False)
    print(f"Real data saved to large_real_data.csv. Rows processed: {len(df_real_filtered)}")

except FileNotFoundError:
    print(f"Error: {real_data_filename} not found. Please ensure it's in the same directory as data_setup.py.")
    print("Skipping real data processing.")
except Exception as e:
    print(f"An error occurred during real data processing: {e}")
    print("Skipping real data processing.")
