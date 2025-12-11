#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip> // For std::fixed and std::setprecision

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
double calculateSSE(const std::vector<Point>& data, const std::vector<Point>& centroids) {
    double sse = 0.0;
    for (const auto& p : data) {
        if (p.cluster_id != -1) { // Only points assigned to a cluster contribute to SSE
            sse += std::pow(euclideanDistance(p, centroids[p.cluster_id]), 2);
        }
    }
    return sse;
}

// Main K-Means algorithm
void runKMeans(std::vector<Point>& data, int k, int max_iterations) {
    // 1. Initialize centroids randomly
    std::vector<Point> centroids(k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, data.size() - 1);

    for (int i = 0; i < k; ++i) {
        centroids[i] = data[distrib(gen)]; // Copy a random data point as initial centroid
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        // 2. Assignment step: Assign each point to the closest centroid
        for (auto& p : data) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_centroid = -1;
            for (int i = 0; i < k; ++i) {
                double dist = euclideanDistance(p, centroids[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = i;
                }
            }
            p.cluster_id = closest_centroid;
        }

        // 3. Update step: Recalculate centroids
        std::vector<std::vector<double>> new_centroids_sum(k, std::vector<double>(data[0].features.size(), 0.0));
        std::vector<int> cluster_counts(k, 0);

        for (const auto& p : data) {
            if (p.cluster_id != -1) {
                cluster_counts[p.cluster_id]++;
                for (size_t i = 0; i < p.features.size(); ++i) {
                    new_centroids_sum[p.cluster_id][i] += p.features[i];
                }
            }
        }

        for (int i = 0; i < k; ++i) {
            if (cluster_counts[i] > 0) {
                for (size_t j = 0; j < centroids[i].features.size(); ++j) {
                    centroids[i].features[j] = new_centroids_sum[i][j] / cluster_counts[i];
                }
            } else {
                // Handle empty cluster: reinitialize centroid to a random data point
                centroids[i] = data[distrib(gen)];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <filename.csv> <k_clusters>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int k = std::stoi(argv[2]);

    if (k <= 0) {
        std::cerr << "Error: Number of clusters (k) must be a positive integer." << std::endl;
        return 1;
    }

    // Read data
    std::vector<Point> data = readCsvData(filename);
    if (data.empty()) {
        std::cerr << "Error: No data loaded or file not found." << std::endl;
        return 1;
    }
    
    // Ensure that k is not greater than the number of data points
    if (k > data.size()) {
        std::cerr << "Error: Number of clusters (k) cannot be greater than the number of data points." << std::endl;
        return 1;
    }

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run K-Means
    int max_iterations = 50;
    runKMeans(data, k, max_iterations);

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Calculate SSE
    // Need to pass the final centroids to calculate SSE, but runKMeans doesn't return them directly.
    // A better approach would be to make centroids an output parameter or return it from runKMeans.
    // For now, let's re-run assignment to get final clusters and then calculate SSE.
    // This is not ideal for performance but ensures correct SSE calculation for benchmark.
    // In a real scenario, centroids would be stored or returned by the K-Means function.
    
    // Re-calculating centroids after the runKMeans has finished for SSE calculation.
    // This is not efficient, but ensures correctness of SSE value.
    std::vector<Point> final_centroids(k);
    if (!data.empty()) {
        final_centroids.assign(k, Point()); // Initialize with empty points
        for(int i = 0; i < k; ++i) {
            final_centroids[i].features.resize(data[0].features.size(), 0.0);
        }
    }

    std::vector<std::vector<double>> final_centroids_sum(k, std::vector<double>(data[0].features.size(), 0.0));
    std::vector<int> final_cluster_counts(k, 0);

    for (const auto& p : data) {
        if (p.cluster_id != -1) {
            final_cluster_counts[p.cluster_id]++;
            for (size_t i = 0; i < p.features.size(); ++i) {
                final_centroids_sum[p.cluster_id][i] += p.features[i];
            }
        }
    }

    for (int i = 0; i < k; ++i) {
        if (final_cluster_counts[i] > 0) {
            for (size_t j = 0; j < final_centroids[i].features.size(); ++j) {
                final_centroids[i].features[j] = final_centroids_sum[i][j] / final_cluster_counts[i];
            }
        } else {
            // If a cluster is empty after final assignment, its centroid doesn't contribute to SSE calculation
            // or should be handled as an edge case, but for calculation purposes we need a valid centroid.
            // For now, we'll leave it as is, which means it won't be used in SSE calculation if no points are assigned.
        }
    }


    double final_sse = calculateSSE(data, final_centroids);

    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Serial Time: " << elapsed_seconds.count() << " seconds" << std::endl;
    std::cout << "Final SSE: " << final_sse << std::endl;

    return 0;
}
