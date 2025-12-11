#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <mpi.h>   // For MPI operations

// Define a Point structure
struct Point {
    std::vector<double> features;
    int cluster_id; // -1 initially, then assigned to a cluster
};

// Function to calculate Euclidean distance between two points
double euclideanDistance(const Point& p1, const Point& p2) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < p1.features.size(); ++i) {
        sum_sq += std::pow(p1.features[i] - p2.features[i], 2);
    }
    return std::sqrt(sum_sq);
}

// Function to read data from a CSV file into a vector of Points
// This function will be called by ALL ranks, but each rank will only process its assigned slice.
std::vector<Point> readCsvData(const std::string& filename) {
    std::vector<Point> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string segment;
        Point p;
        while (std::getline(ss, segment, ',')) {
            try {
                p.features.push_back(std::stod(segment));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid argument: " << e.what() << " in line: " << line << std::endl;
                // Handle error or skip this segment/point
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range: " << e.what() << " in line: " << line << std::endl;
                // Handle error or skip this segment/point
            }
        }
        if (!p.features.empty()) { // Only add if line was successfully parsed
            p.cluster_id = -1; // Initialize cluster_id
            data.push_back(p);
        }
    }
    file.close();
    return data;
}

// Function to calculate Sum of Squared Errors (SSE)
double calculateSSE(const std::vector<Point>& data, const std::vector<Point>& centroids,
                    int start_idx, int end_idx) {
    double sse = 0.0;
    for (int i = start_idx; i < end_idx; ++i) {
        const auto& p = data[i];
        if (p.cluster_id != -1) { // Only points assigned to a cluster contribute to SSE
            sse += std::pow(euclideanDistance(p, centroids[p.cluster_id]), 2);
        }
    }
    return sse;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <filename.csv> <k_clusters>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];
    int k = std::stoi(argv[2]);

    if (k <= 0) {
        if (rank == 0) {
            std::cerr << "Error: Number of clusters (k) must be a positive integer." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // All ranks read the full dataset
    std::vector<Point> data = readCsvData(filename);
    if (data.empty()) {
        if (rank == 0) {
            std::cerr << "Error: No data loaded or file not found." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Ensure that k is not greater than the number of data points
    if (k > data.size()) {
        if (rank == 0) {
            std::cerr << "Error: Number of clusters (k) cannot be greater than the number of data points." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }


    int n_samples = data.size();
    int n_features = data[0].features.size();

    // Master (Rank 0) initializes centroids
    std::vector<Point> centroids(k);
    if (rank == 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, n_samples - 1);

        for (int i = 0; i < k; ++i) {
            centroids[i] = data[distrib(gen)]; // Copy a random data point as initial centroid
        }
    }

    // Define custom MPI_Datatype for Point features
    // This is a simplified approach, a more robust solution might involve
    // sending features as a contiguous array and reconstructing points.
    // For now, let's just broadcast the features directly.

    // Broadcast initial centroids' features from Rank 0 to all ranks
    // Each centroid has 'n_features' doubles. We need to send k * n_features doubles.
    std::vector<double> centroid_features_buffer(k * n_features);

    if (rank == 0) {
        int buffer_idx = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n_features; ++j) {
                centroid_features_buffer[buffer_idx++] = centroids[i].features[j];
            }
        }
    }

    MPI_Bcast(centroid_features_buffer.data(), k * n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct centroids on all ranks
    int buffer_idx = 0;
    for (int i = 0; i < k; ++i) {
        centroids[i].features.resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            centroids[i].features[j] = centroid_features_buffer[buffer_idx++];
        }
    }

    // Start timer on Rank 0
    double start_time = 0.0;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    int max_iterations = 50;

    // Determine data slice for each process
    int chunk_size = n_samples / num_procs;
    int start_idx = rank * chunk_size;
    int end_idx = (rank == num_procs - 1) ? n_samples : start_idx + chunk_size;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // 2. Assignment step: Assign each point in the local slice to the closest centroid
        for (int i = start_idx; i < end_idx; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_centroid = -1;
            for (int j = 0; j < k; ++j) {
                double dist = euclideanDistance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            data[i].cluster_id = closest_centroid;
        }

        // 3. Update step: Recalculate centroids
        std::vector<std::vector<double>> local_new_centroids_sum(k, std::vector<double>(n_features, 0.0));
        std::vector<int> local_cluster_counts(k, 0);

        for (int i = start_idx; i < end_idx; ++i) {
            const auto& p = data[i];
            if (p.cluster_id != -1) {
                local_cluster_counts[p.cluster_id]++;
                for (int f = 0; f < n_features; ++f) {
                    local_new_centroids_sum[p.cluster_id][f] += p.features[f];
                }
            }
        }

        // Global sum for centroids and counts using MPI_Allreduce
        std::vector<double> global_new_centroids_sum_flat(k * n_features);
        std::vector<int> global_cluster_counts(k);

        // Flatten local_new_centroids_sum for Allreduce
        std::vector<double> local_new_centroids_sum_flat(k * n_features);
        int flat_idx = 0;
        for (int c = 0; c < k; ++c) {
            for (int f = 0; f < n_features; ++f) {
                local_new_centroids_sum_flat[flat_idx++] = local_new_centroids_sum[c][f];
            }
        }

        MPI_Allreduce(local_new_centroids_sum_flat.data(), global_new_centroids_sum_flat.data(),
                      k * n_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cluster_counts.data(), global_cluster_counts.data(),
                      k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Update centroids on all ranks
        flat_idx = 0;
        for (int c = 0; c < k; ++c) {
            if (global_cluster_counts[c] > 0) {
                for (int f = 0; f < n_features; ++f) {
                    centroids[c].features[f] = global_new_centroids_sum_flat[flat_idx++] / global_cluster_counts[c];
                }
            } else {
                // Handle empty cluster: reinitialize centroid to a random data point from master (Rank 0) if it's Rank 0
                // For other ranks, it's safer to just keep the old centroid or re-broadcast from Rank 0
                // For simplicity, let's reinitialize from a random point if a cluster becomes empty globally.
                // This reinitialization would ideally be handled more robustly in a full K-Means++.
                // For now, if a cluster is empty globally, we'll assign it to a random data point from the full dataset (data).
                // This might lead to different initializations across ranks if not careful.
                // A safer approach might be to broadcast new random centroids from Rank 0 for empty clusters.
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> distrib(0, n_samples - 1);
                
                // Only Rank 0 reinitializes and then broadcasts, or Allreduce handles it.
                // Given MPI_Allreduce for sums and counts, if a cluster is empty globally,
                // we should probably reinitialize it from a random point.
                // For now, we'll reinitialize locally, but in a real scenario, this would
                // need a collective decision or broadcast from master.
                if (rank == 0) { // Only master makes this decision
                    centroids[c] = data[distrib(gen)];
                }
            }
        }
        // If an empty cluster was reinitialized by rank 0, its new position needs to be broadcasted to all other ranks.
        // This is a subtle point and often handled differently. For this baseline, we assume the Allreduce
        // for sums and counts is sufficient, meaning if a cluster is empty, its sum/count will be zero
        // across all processes, leading to the same result for its centroid on all processes (which would be NaN or 0 if not handled).
        // A full implementation would likely have a check for global_cluster_counts[c] == 0 and then
        // re-broadcast or re-select centroids. For simplicity and fixed iterations, we'll proceed this way.
        
        // Re-broadcast centroids if any were reinitialized, or if there's any chance of divergence.
        // For current logic, we're relying on Allreduce to keep centroids consistent.
        // However, if an empty cluster is reinitialized on rank 0, this needs to be shared.
        // Let's assume that if a cluster becomes empty, its centroid remains in its last known valid position
        // until points are assigned to it again. Re-randomizing locally might diverge.
        // To be safe, if rank 0 reinitializes a centroid, it should broadcast it.
        // This adds complexity, for now we will assume the centroids are consistent across ranks due to Allreduce
        // and identical calculations for the non-empty clusters.
        // For empty clusters, the centroid will remain 0.0, which might not be desired.
        // A simpler strategy for empty clusters in parallel K-Means: don't move them, or pick a random point from global dataset.
        // For now, if global_cluster_counts[c] == 0, the centroid will remain as it was from the previous iteration,
        // or effectively 0 if not previously initialized (which shouldn't happen after first Bcast).
        
        // A more robust approach would be to have Rank 0 determine new centroids for empty clusters
        // and then broadcast ALL centroids again.
        if (rank == 0) {
            int current_buffer_idx = 0;
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < n_features; ++j) {
                    centroid_features_buffer[current_buffer_idx++] = centroids[i].features[j];
                }
            }
        }
        MPI_Bcast(centroid_features_buffer.data(), k * n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        flat_idx = 0;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n_features; ++j) {
                centroids[i].features[j] = centroid_features_buffer[flat_idx++];
            }
        }

    } // End of iterations loop


    // Calculate local SSE
    double local_sse = calculateSSE(data, centroids, start_idx, end_idx);

    // Reduce local SSEs to get total SSE on Rank 0
    double total_sse = 0.0;
    MPI_Reduce(&local_sse, &total_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Stop timer on Rank 0 and print results
    if (rank == 0) {
        double end_time = MPI_Wtime();
        double elapsed_seconds = end_time - start_time;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "MPI Time: " << elapsed_seconds << " seconds" << std::endl;
        std::cout << "Final SSE: " << total_sse << std::endl;
    }

    MPI_Finalize();
    return 0;
}