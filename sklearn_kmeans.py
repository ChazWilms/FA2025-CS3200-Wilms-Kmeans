import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import time
import sys

def read_csv_data(filename):
    """Reads a headerless CSV file into a NumPy array."""
    try:
        df = pd.read_csv(filename, header=None)
        return df.to_numpy()
    except FileNotFoundError:
        print(f"Error: File not found at {filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV {filename}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python sklearn_kmeans.py <filename.csv> <k_clusters>", file=sys.stderr)
        sys.exit(1)

    filename = sys.argv[1]
    k = int(sys.argv[2])

    data = read_csv_data(filename)

    if k <= 0 or k > len(data):
        print(f"Error: k ({k}) must be a positive integer and less than or equal to the number of samples ({len(data)}).", file=sys.stderr)
        sys.exit(1)

    # Initialize KMeans with fixed parameters to match C++ versions (max_iter=50)
    # n_init='auto' ensures consistent initialization behavior.
    # random_state for reproducibility of initial centroids
    kmeans = KMeans(n_clusters=k, init='random', n_init=1, max_iter=50, random_state=42, algorithm='lloyd')

    start_time = time.perf_counter()
    kmeans.fit(data)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    final_sse = kmeans.inertia_ # KMeans.inertia_ is the sum of squared distances of samples to their closest cluster center.

    print(f"SKLearn Time: {elapsed_time:.3f} seconds")
    print(f"Final SSE: {final_sse:.3f}")

if __name__ == "__main__":
    main()
