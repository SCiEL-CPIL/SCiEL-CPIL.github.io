"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0

# Cache for the k-NN graph to avoid recomputation on every call
# Key: hash of distance_matrix (or a simple check if it's the same array)
# Since distance_matrix is likely static per instance, we can cache based on its shape and a checksum
# However, checksumming the whole matrix every call might be expensive if N is large, but N=50 is small.
# Let's just cache based on the object id or a simple hash of the first few elements + shape.
# Actually, since the evaluator might pass different instances, we need to be careful.
# The problem states "eight generated 50-customer instances".
# We can assume the distance matrix is fixed for a specific run of the evaluator on one instance,
# but changes between instances.
# To be safe and fast, we will compute the k-NN graph lazily and cache it keyed by a hash of the matrix.

_knn_cache = {}

def _get_knn_graph(distance_matrix, k=6):
    """
    Compute and cache the k-Nearest Neighbors graph for the given distance matrix.
    Returns a list of lists, where graph[i] is a list of k nearest neighbors of i.
    """
    # Create a simple hash key for the distance matrix
    # Using a tuple of shape and a few sample values to avoid full hash computation overhead if possible,
    # but for N=50, hashing the whole matrix is fast enough (<0.1ms).
    try:
        key = (distance_matrix.shape, distance_matrix[0,0], distance_matrix[0,1], distance_matrix[1,0], distance_matrix[1,1])
        if key in _knn_cache:
            return _knn_cache[key]
    except:
        pass
        
    n = distance_matrix.shape[0]
    # We want k nearest neighbors for each node.
    # Exclude self.
    # For depot (0), we also consider it.
    
    # Efficient way: for each node i, get indices of k smallest distances, excluding i.
    graph = [None] * n
    for i in range(n):
        # Create a copy of the row with self set to inf
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        # Get indices of k smallest values
        # argsort is O(N log N), for N=50 this is trivial.
        # We can use partition for O(N) average, but argsort is fine.
        neighbors_idx = np.argsort(dists)[:k]
        graph[i] = neighbors_idx.tolist()
        
    try:
        _knn_cache[key] = graph
    except:
        pass
        
    return graph

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP using a k-Nearest Neighbors spatial graph.
    
    Strategy:
    1. Build a sparse graph of k-nearest neighbors for all nodes (including depot).
    2. From the current node, look only at its k-nearest neighbors.
    3. Filter these neighbors by:
       a. They must be in unvisited_nodes.
       b. They must satisfy the capacity constraint (demands[node] <= rest_capacity).
    4. If no feasible neighbor is found in the k-NN set, fall back to the global nearest feasible node 
       (or return depot if we want to be strict, but fallback is safer for completeness).
    5. Select the feasible neighbor with the smallest distance to the current node.
    
    This approach enforces local planarity and prevents long jumps that cross other clusters,
    which often leads to better route structures in CVRP.
    """
    if _time_exceeded():
        return 0

    unvisited = set(unvisited_nodes)
    if not unvisited:
        return 0

    # Get or compute the k-NN graph
    k = 8  # Slightly higher k to ensure we have options
    graph = _get_knn_graph(distance_matrix, k)
    
    # Get neighbors of current node
    neighbors = graph[current_node]
    
    best_node = -1
    best_dist = float('inf')
    
    # Check neighbors for feasibility
    for n in neighbors:
        if n in unvisited:
            if demands[n] <= rest_capacity:
                d = distance_matrix[current_node, n]
                if d < best_dist:
                    best_dist = d
                    best_node = n
                    
    if best_node != -1:
        return int(best_node)
    
    # Fallback: If no feasible neighbor in k-NN graph, find the globally nearest feasible node
    # This ensures we don't get stuck if the k-NN graph is too restrictive
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0
        
    # Find min distance among all feasible
    min_dist = float('inf')
    min_node = -1
    for n in feasible:
        d = distance_matrix[current_node, n]
        if d < min_dist:
            min_dist = d
            min_node = n
            
    return int(min_node)
# EVOLVE-BLOCK-END
