"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to cache computed values per instance
_cache_key = None
_cache_data = None

def _compute_cache(unvisited_nodes, distance_matrix):
    """Compute global features for the instance."""
    global _cache_key, _cache_data
    
    # Use the frozenset of unvisited nodes and a hash of the matrix as key
    # Since unvisited changes, we actually need per-call logic, but we can cache
    # the full distance matrix properties if the instance is static.
    # However, unvisited_nodes changes. The instance (distance_matrix) is static per evaluation run.
    # We can cache properties of distance_matrix.
    
    n = distance_matrix.shape[0]
    
    # Precompute 2-nearest neighbors for each node
    # This is O(n^2) log n, which is fast for n=50.
    # For each node i, find the 2 closest nodes j, k.
    
    # To avoid O(n^3) per call, we precompute these once per distance_matrix.
    
    # Key by the id of the distance_matrix object? Or a hash of its content?
    # Using id() is risky if the evaluator reuses memory, but for a single evaluation
    # run, the distance_matrix object is likely stable.
    # Let's use a hash of the first few elements + shape as a robust key.
    key = (distance_matrix.shape, tuple(distance_matrix[0, :5]), tuple(distance_matrix[-1, -5:]))
    
    if _cache_key == key:
        return _cache_data
    
    # Compute 2-nearest neighbors
    # dist_matrix is n x n
    # For each i, sort distances to all j != i
    # We want the 2 smallest distances.
    
    # np.argpartition is efficient.
    # For each row i, get indices of 2 smallest values (excluding self).
    # Set diagonal to inf to exclude self.
    
    dm = distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    
    # Get 2 smallest indices for each row
    # shape (n, 2)
    nn2 = np.argpartition(dm, 1, axis=1)[:, :2]
    
    # Store as a list of lists or a 2D array
    # nn2[i] contains the two nearest neighbors to i
    
    _cache_key = key
    _cache_data = nn2
    
    return nn2

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a hybrid heuristic:
    1. Compute distance from current to each unvisited node.
    2. Compute a 'global connectivity score' based on how well the candidate
       connects to the rest of the unvisited set (using precomputed 2-NNs).
    3. Combine these into a single score and pick the minimum.
    """
    if len(unvisited_nodes) == 0:
        return -1  # Should not happen if called correctly
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Get or compute cache
    nn2 = _compute_cache(unvisited_nodes, distance_matrix)
    
    # Vectorized distance from current_node to all unvisited nodes
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Compute global score for each candidate in unvisited_nodes
    # For a candidate v, its global score is the sum of distances from v
    # to its 2 nearest neighbors (in the full graph, not just unvisited).
    # Why full graph? Because those neighbors are likely to be visited soon,
    # or are structurally important.
    
    # Actually, a better metric: How many of the unvisited nodes have v
    # as one of their 2 nearest neighbors? This measures v's centrality
    # in the remaining unvisited set.
    
    # Let's compute for each candidate v in unvisited_nodes:
    # count_v = number of u in unvisited_nodes such that v is in nn2[u]
    
    # We can do this efficiently:
    # Create a boolean matrix or use bincount.
    
    # nn2 is an array of shape (n, 2).
    # We want to count, for each v in unvisited_nodes, how many times v appears
    # in nn2[u] for u in unvisited_nodes.
    
    # Flatten nn2 for unvisited nodes only
    # unvisited_nodes is an array of node indices.
    # nn2[unvisited_nodes] gives the 2-NNs for each unvisited node.
    
    # Let's count occurrences of each node in the 2-NNs of all unvisited nodes.
    # This can be done with np.bincount.
    
    # Flatten the 2-NNs of unvisited nodes
    nn_flat = nn2[unvisited_nodes].flatten()
    
    # Count occurrences of each node index in nn_flat
    # We only care about counts for nodes in unvisited_nodes, but bincount
    # gives counts for all nodes 0..n-1.
    
    n = distance_matrix.shape[0]
    counts = np.bincount(nn_flat, minlength=n)
    
    # Extract counts for the unvisited nodes
    global_scores = counts[unvisited_nodes]
    
    # Normalize global_scores to be in a similar range as dists
    # dists are Euclidean distances, likely in [0, ~1.4] for unit square.
    # global_scores are counts in [0, len(unvisited_nodes)].
    # We want to penalize nodes with low connectivity.
    # So we want to minimize (dists - alpha * global_scores).
    # Or minimize (dists + beta * (max_count - global_scores)).
    
    # Let's use a penalty for low connectivity.
    # penalty = max_count - global_scores
    # score = dists + penalty_weight * penalty
    
    # Choose penalty_weight dynamically.
    # If dists are small, we want to rely more on connectivity.
    # If dists are large, we want to rely more on distance.
    
    # Let's try a simple linear combination.
    # Normalize dists to [0, 1] and global_scores to [0, 1].
    
    min_dist = np.min(dists)
    max_dist = np.max(dists)
    dist_range = max_dist - min_dist
    if dist_range > 0:
        norm_dists = (dists - min_dist) / dist_range
    else:
        norm_dists = np.zeros_like(dists)
    
    min_count = np.min(global_scores)
    max_count = np.max(global_scores)
    count_range = max_count - min_count
    if count_range > 0:
        norm_counts = (global_scores - min_count) / count_range
    else:
        norm_counts = np.ones_like(global_scores)
    
    # We want to minimize distance, so high norm_dists is bad.
    # We want high connectivity, so low norm_counts is bad.
    # score = norm_dists + weight * (1 - norm_counts)
    
    # Let's try weight = 0.5
    weight = 0.5
    scores = norm_dists + weight * (1 - norm_counts)
    
    # Pick the index with minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
