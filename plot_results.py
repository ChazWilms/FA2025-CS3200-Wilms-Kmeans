import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Read and Parse Benchmark Results ---
try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Error: benchmark_results.csv not found. Please run benchmark_runner.py first.")
    exit()

def plot_data_for_dataset(dataset_name, df_dataset, output_filename_prefix):
    
    # --- Data Extraction ---
    
    # C++ Serial
    serial_cpp_row = df_dataset[(df_dataset['type'] == 'serial_cpp') & (df_dataset['cores'] == 1)]
    if serial_cpp_row.empty:
        print(f"Error: C++ Serial data not found for {dataset_name}.")
        return

    serial_cpp_time = serial_cpp_row['avg_time'].iloc[0]
    serial_cpp_sse = serial_cpp_row['avg_sse'].iloc[0]
    
    # C++ MPI
    mpi_cpp_data = df_dataset[df_dataset['type'] == 'mpi_cpp'].sort_values(by='cores')
    mpi_cpp_cores = np.array([1] + mpi_cpp_data['cores'].tolist()) # Include 1 for serial baseline
    mpi_cpp_times = np.array([serial_cpp_time] + mpi_cpp_data['avg_time'].tolist())
    mpi_cpp_sses = np.array([serial_cpp_sse] + mpi_cpp_data['avg_sse'].tolist())

    # Scikit-learn Python
    sklearn_python_row = df_dataset[df_dataset['type'] == 'sklearn_python']
    if sklearn_python_row.empty:
        print(f"Error: Scikit-learn Python data not found for {dataset_name}.")
        sklearn_python_time = 0.0 # Placeholder
        sklearn_python_sse = 0.0 # Placeholder
    else:
        sklearn_python_time = sklearn_python_row['avg_time'].iloc[0]
        sklearn_python_sse = sklearn_python_row['avg_sse'].iloc[0]
        
    # --- Plotting ---
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7)) # 1 row, 3 columns for three plots
    fig.suptitle(f'K-Means Performance for {dataset_name.capitalize()} Data', fontsize=18)

    # Plot 1: Speedup vs Cores (C++ Only)
    ax1 = axes[0]
    if np.any(mpi_cpp_times <= 0):
        ax1.set_title('Speedup (C++ MPI vs Serial) - No valid times')
        ax1.text(0.5, 0.5, 'Invalid time data', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    else:
        speedup = serial_cpp_time / mpi_cpp_times
        ideal_speedup = mpi_cpp_cores
        ax1.plot(mpi_cpp_cores, speedup, marker='o', linestyle='-', color='blue', label='Actual Speedup')
        ax1.plot(mpi_cpp_cores, ideal_speedup, linestyle='--', color='red', label='Ideal Linear Speedup')
        ax1.set_title('Speedup vs Number of Cores (C++ MPI vs Serial)')
        ax1.set_xlabel('Number of Cores')
        ax1.set_ylabel('Speedup (T_serial / T_parallel)')
        ax1.set_xticks(mpi_cpp_cores)
        ax1.legend()
        ax1.grid(True)
        if len(mpi_cpp_cores) > 1:
            ax1.set_xscale('log', base=2)
        ax1.set_yscale('linear')


    # Plot 2: Absolute Time Comparison (Bar Chart)
    ax2 = axes[1]
    time_labels = ['Serial C++', 'SKLearn Python'] + [f'MPI C++ ({c}c)' for c in mpi_cpp_data['cores']]
    time_values = [serial_cpp_time, sklearn_python_time] + mpi_cpp_data['avg_time'].tolist()

    ax2.bar(time_labels, time_values, color=['skyblue', 'lightgreen'] + ['lightcoral'] * len(mpi_cpp_data))
    ax2.set_title('Absolute Execution Time Comparison')
    ax2.set_xlabel('Implementation')
    ax2.set_ylabel('Average Time (seconds)')
    ax2.grid(axis='y')
    ax2.set_xticklabels(time_labels, rotation=45, ha='right')
    # Optional: Add text labels for time values on bars
    for i, v in enumerate(time_values):
        ax2.text(i, v + (max(time_values) * 0.01), f"{v:.4f}", ha='center', va='bottom', fontsize=8)


    # Plot 3: Final SSE Comparison (Bar Chart)
    ax3 = axes[2]
    sse_labels = ['Serial C++', 'SKLearn Python'] + [f'MPI C++ ({c}c)' for c in mpi_cpp_data['cores']]
    sse_values = [serial_cpp_sse, sklearn_python_sse] + mpi_cpp_data['avg_sse'].tolist()

    ax3.bar(sse_labels, sse_values, color=['skyblue', 'lightgreen'] + ['lightcoral'] * len(mpi_cpp_data))
    ax3.set_title('Final Sum of Squared Errors (SSE) Comparison')
    ax3.set_xlabel('Implementation')
    ax3.set_ylabel('Average Final SSE')
    ax3.grid(axis='y')
    ax3.set_xticklabels(sse_labels, rotation=45, ha='right')
    # Optional: Add text labels for SSE values on bars
    for i, v in enumerate(sse_values):
        # Adjust vertical position for better visibility, especially for large numbers
        if v < 1e6: # For smaller numbers
            ax3.text(i, v + (max(sse_values) * 0.01), f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        else: # For larger numbers, use scientific notation or full number with smaller font
            ax3.text(i, v + (max(sse_values) * 0.01), f"{v:.2e}", ha='center', va='bottom', fontsize=8)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
    output_png_filename = f"{output_filename_prefix}_results.png"
    plt.savefig(output_png_filename)
    print(f"Generated {output_png_filename}")

# Filter data for synthetic and real datasets
df_synthetic = df[df['dataset'] == 'synthetic'].copy()
df_real = df[df['dataset'] == 'real'].copy()

# Generate plots for synthetic data
plot_data_for_dataset("synthetic", df_synthetic, "synthetic")

# Generate plots for real data
plot_data_for_dataset("real", df_real, "real")

print("\nAll plots generated!")

# --- Generate and Save Summary Table ---
summary_table_filename = "benchmark_summary_table.txt"
print(f"\n--- Benchmark Summary Table ---")
# Select and format columns for the table output
table_df = df[['dataset', 'type', 'cores', 'avg_time', 'std_dev_time', 'avg_sse', 'std_dev_sse']].copy()
table_df['avg_time'] = table_df['avg_time'].apply(lambda x: f"{x:.4f}")
table_df['std_dev_time'] = table_df['std_dev_time'].apply(lambda x: f"{x:.4f}")
table_df['avg_sse'] = table_df['avg_sse'].apply(lambda x: f"{x:.4f}")
table_df['std_dev_sse'] = table_df['std_dev_sse'].apply(lambda x: f"{x:.4f}")


table_string = table_df.to_string(index=False)
print(table_string)

with open(summary_table_filename, 'w') as f:
    f.write("K-Means Benchmarking Results\n\n")
    f.write(table_string)
print(f"Summary table saved to {summary_table_filename}")
