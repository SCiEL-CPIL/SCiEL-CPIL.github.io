"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    
    Applies a multi-component penalty to the distance matrix:
    1. Usage-based penalty: edges used more often get penalized more.
    2. Structural bias: edges that are not in the K-nearest neighbors of either
       endpoint receive an additional penalty, targeting geometrically implausible edges.
    3. Length-ratio weight: edges significantly longer than the tour average get
       a sigmoidal penalty boost.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances (finite, same shape).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Extract tour edges
    tour = np.asarray(local_opt_tour)
    tour_edges_i = tour[:-1]
    tour_edges_j = tour[1:]
    # Closing edge
    tour_edges_i = np.concatenate([tour_edges_i, [tour[-1]]])
    tour_edges_j = np.concatenate([tour_edges_j, [tour[0]]])
    
    # Compute tour edge lengths
    tour_edge_lengths = edge_distance[tour_edges_i, tour_edges_j]
    avg_tour_edge_len = np.mean(tour_edge_lengths)
    if avg_tour_edge_len <= 0:
        avg_tour_edge_len = 1e-10
    
    # --- Component 1: Usage-based penalty ---
    # Normalize usage counts
    max_usage = edge_n_used.max()
    if max_usage > 0:
        usage_norm = edge_n_used / (1.0 + max_usage)
    else:
        usage_norm = np.zeros_like(edge_n_used)
    
    # Penalty strength for usage: scale with how much an edge is used
    # Only apply to tour edges (the current local optimum's edges)
    usage_penalty = np.zeros((n, n), dtype=float)
    usage_penalty[tour_edges_i, tour_edges_j] = 0.15 * usage_norm[tour_edges_i, tour_edges_j]
    usage_penalty[tour_edges_j, tour_edges_i] = usage_penalty[tour_edges_i, tour_edges_j]
    
    # --- Component 2: K-NN structural bias ---
    # For each node, find its K nearest neighbors
    K = 6
    # For efficiency, use a simple approach: for each node, get its K nearest
    knn_mask = np.zeros((n, n), dtype=bool)
    
    # To keep within time budget, use a vectorized approach
    # Sort distances for each row (excluding diagonal)
    dist_copy = edge_distance.copy()
    np.fill_diagonal(dist_copy, np.inf)
    
    # Get indices of K nearest neighbors for each node
    # This is O(n^2 log n) which for n=100 is fine
    knn_indices = np.argsort(dist_copy, axis=1)[:, :K]
    
    # Build K-NN mask: edge (i,j) is "good" if j is in K-NN of i OR i is in K-NN of j
    for i in range(n):
        for j in knn_indices[i]:
            knn_mask[i, j] = True
            knn_mask[j, i] = True
    
    # Structural penalty: edges NOT in K-NN of either endpoint get penalized
    # This targets geometrically implausible edges
    structural_penalty = np.zeros((n, n), dtype=float)
    # Only apply to tour edges that are structurally poor
    struct_poor = ~knn_mask[tour_edges_i, tour_edges_j]
    structural_penalty[tour_edges_i, tour_edges_j] = 0.10 * struct_poor.astype(float)
    structural_penalty[tour_edges_j, tour_edges_i] = structural_penalty[tour_edges_i, tour_edges_j]
    
    # --- Component 3: Length-ratio sigmoid weight ---
    # Edges significantly longer than average get a sigmoidal penalty boost
    # ratio = edge_length / avg_tour_edge_len
    # sigmoid(ratio - 1.0) gives a smooth transition around ratio=1
    length_ratio = edge_distance / avg_tour_edge_len
    # Sigmoid centered at ratio=1.2, steepness 5
    sigmoid_arg = 5.0 * (length_ratio - 1.2)
    sigmoid_weight = 1.0 / (1.0 + np.exp(-sigmoid_arg))
    
    # Apply length-ratio penalty only to tour edges
    length_penalty = np.zeros((n, n), dtype=float)
    length_penalty[tour_edges_i, tour_edges_j] = 0.08 * sigmoid_weight[tour_edges_i, tour_edges_j]
    length_penalty[tour_edges_j, tour_edges_i] = length_penalty[tour_edges_i, tour_edges_j]
    
    # --- Combine penalties ---
    total_penalty = usage_penalty + structural_penalty + length_penalty
    
    # Ensure penalty is non-negative
    total_penalty = np.maximum(total_penalty, 0.0)
    
    # Apply multiplicative perturbation
    perturbed = edge_distance * (1.0 + total_penalty)
    
    # Ensure symmetry
    perturbed = 0.5 * (perturbed + perturbed.T)
    
    # Ensure finiteness
    perturbed = np.nan_to_num(perturbed, nan=0.0, posinf=edge_distance.max(), neginf=0.0)
    
    # Ensure diagonal is zero (self-distances)
    np.fill_diagonal(perturbed, 0.0)
    
    # Check time budget
    if time.time() > deadline:
        # Return the true distances as a safe fallback
        return edge_distance.copy()
    
    return perturbed
# EVOLVE-BLOCK-END
