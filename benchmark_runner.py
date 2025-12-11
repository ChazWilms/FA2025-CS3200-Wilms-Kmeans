import subprocess
import csv
import numpy as np
import os
import re

def run_kmeans_test(command, expected_output_type):
    """
    Runs a K-Means test command and parses its output for time and SSE.
    expected_output_type: "Serial", "MPI", or "SKLearn" to guide parsing.
    Returns (time, sse) or (None, None) if parsing fails.
    """
    try:
        process = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        output = process.stdout
        
        time_match = re.search(fr'{expected_output_type} Time: ([\d.]+) seconds', output)
        sse_match = re.search(r'Final SSE: ([\d.]+)', output)

        time = float(time_match.group(1)) if time_match else None
        sse = float(sse_match.group(1)) if sse_match else None
        
        if time is None or sse is None:
            print(f"Warning: Could not parse {expected_output_type} time or SSE from command: {command}")
            print(f"Output:\n{output}")
            return None, None
        
        return time, sse
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.cmd}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

def main():
    datasets = {
        "synthetic": "synthetic_data.csv",
        "real": "large_real_data.csv"
    }
    k_clusters = 5  # Number of clusters
    serial_executable = "./serial_kmeans"
    mpi_executable = "./mpi_kmeans"
    sklearn_executable = "python3 sklearn_kmeans.py" # New executable for scikit-learn
    num_trials = 10
    mpi_processes = [2, 4, 6, 8] # MPI processes to test

    results = []

    print("Starting K-Means Benchmarking...")

    for dataset_name, data_file in datasets.items():
        print(f"\n--- Benchmarking {dataset_name} data ({data_file}) ---")

        # --- Serial K-Means (C++) ---
        print(f"Running Serial C++ K-Means ({num_trials} trials)...")
        serial_times = []
        serial_sses = []
        for i in range(num_trials):
            print(f"  Trial {i+1}/{num_trials} (Serial C++)...", end='\r')
            command = f"{serial_executable} {data_file} {k_clusters}"
            time, sse = run_kmeans_test(command, "Serial")
            if time is not None and sse is not None:
                serial_times.append(time)
                serial_sses.append(sse)
        
        if serial_times:
            avg_time = np.mean(serial_times)
            avg_sse = np.mean(serial_sses)
            results.append({
                "dataset": dataset_name,
                "type": "serial_cpp",
                "cores": 1,
                "avg_time": avg_time,
                "std_dev_time": np.std(serial_times),
                "avg_sse": avg_sse,
                "std_dev_sse": np.std(serial_sses),
                "all_times": serial_times,
                "all_sses": serial_sses
            })
            print(f"Serial C++ Avg Time: {avg_time:.4f}s, Avg SSE: {avg_sse:.4f}")
        else:
            print(f"  No successful serial C++ runs for {dataset_name}.")

        # --- MPI K-Means (C++) ---
        for num_proc in mpi_processes:
            print(f"Running MPI C++ K-Means with {num_proc} processes ({num_trials} trials)...")
            mpi_times = []
            mpi_sses = []
            for i in range(num_trials):
                print(f"  Trial {i+1}/{num_trials} (MPI C++ {num_proc} processes)...", end='\r')
                command = f"mpirun -np {num_proc} {mpi_executable} {data_file} {k_clusters}"
                time, sse = run_kmeans_test(command, "MPI")
                if time is not None and sse is not None:
                    mpi_times.append(time)
                    mpi_sses.append(sse)
            
            if mpi_times:
                avg_time = np.mean(mpi_times)
                avg_sse = np.mean(mpi_sses)
                results.append({
                    "dataset": dataset_name,
                    "type": "mpi_cpp",
                    "cores": num_proc,
                    "avg_time": avg_time,
                    "std_dev_time": np.std(mpi_times),
                    "avg_sse": avg_sse,
                    "std_dev_sse": np.std(mpi_sses),
                    "all_times": mpi_times,
                    "all_sses": mpi_sses
                })
                print(f"MPI C++ ({num_proc} cores) Avg Time: {avg_time:.4f}s, Avg SSE: {avg_sse:.4f}")
            else:
                print(f"  No successful MPI C++ runs for {dataset_name} with {num_proc} processes.")
        
        # --- Scikit-learn K-Means (Python) ---
        print(f"Running Scikit-learn K-Means ({num_trials} trials)...")
        sklearn_times = []
        sklearn_sses = []
        for i in range(num_trials):
            print(f"  Trial {i+1}/{num_trials} (SKLearn Python)...", end='\r')
            command = f"{sklearn_executable} {data_file} {k_clusters}"
            time, sse = run_kmeans_test(command, "SKLearn")
            if time is not None and sse is not None:
                sklearn_times.append(time)
                sklearn_sses.append(sse)
        
        if sklearn_times:
            avg_time = np.mean(sklearn_times)
            avg_sse = np.mean(sklearn_sses)
            results.append({
                "dataset": dataset_name,
                "type": "sklearn_python",
                "cores": 1, # Scikit-learn runs on a single process (or uses internal multi-threading if configured)
                "avg_time": avg_time,
                "std_dev_time": np.std(sklearn_times),
                "avg_sse": avg_sse,
                "std_dev_sse": np.std(sklearn_sses),
                "all_times": sklearn_times,
                "all_sses": sklearn_sses
            })
            print(f"Scikit-learn Avg Time: {avg_time:.4f}s, Avg SSE: {avg_sse:.4f}")
        else:
            print(f"  No successful Scikit-learn runs for {dataset_name}.")
        print() # Newline after each dataset

    # --- Save Results to CSV ---
    output_filename = "benchmark_results.csv"
    print(f"Saving results to {output_filename}...")
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ["dataset", "type", "cores", "avg_time", "std_dev_time", "avg_sse", "std_dev_sse"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            # Create a new dictionary with only the fields we want to write
            filtered_row = {key: row[key] for key in fieldnames}
            writer.writerow(filtered_row)
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()