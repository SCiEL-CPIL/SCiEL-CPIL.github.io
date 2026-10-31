"""Traveling salesman constructive heuristic: Dynamic KNN with Degree Penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for the dynamic heuristic
_start_time = None
_DEADLINE = 145.0
_visited_count = {}
_current_instance_id = None
_knn_graph = None
_n_nodes = 0
_k_neighbors = 8

def _fingerprint(distance_matrix):
    n = distance_matrix.shape[0]
    if n > 10:
        sample = distance_matrix[:10, :10].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)

def _init_instance(distance_matrix):
    global _current_instance_id, _knn_graph, _n_nodes, _visited_count, _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    fid = _fingerprint(distance_matrix)
    if fid != _current_instance_id:
        _current_instance_id = fid
        _n_nodes = distance_matrix.shape[0]
        _visited_count = {i: 0 for i in range(_n_nodes)}
        
        # Build KNN graph
        # For N=50, this is very fast.
        # We want a graph that captures local structure but allows some flexibility.
        # Using K=8 seems reasonable for 50 nodes.
        
        # Compute distances from each node to all others
        # distance_matrix is already provided.
        # For each node i, find k nearest neighbors.
        
        # To speed up, we can just use the provided distance_matrix directly in the selection step,
        # but having a pre-filtered list of candidates (KNN) can help if we want to restrict search.
        # However, the penalty approach works on the full set or a subset.
        # Let's stick to evaluating all unvisited nodes but using a penalty term.
        # The "KNN" aspect here is conceptual: we prioritize neighbors but penalize
        # nodes that have many unvisited neighbors (to avoid creating holes) 
        # OR penalize nodes that are too far.
        
        # Actually, a better dynamic heuristic:
        # Score(node) = distance(current, node) + lambda * (1 / (1 + unvisited_neighbors_count(node)))
        # Wait, if a node has many unvisited neighbors, it's a "hub". 
        # Visiting it early might be good to connect regions, or bad if it isolates others?
        # Usually, in TSP, you want to avoid long jumps.
        # A "degree" penalty often helps avoid getting stuck in a region.
        
        # Let's try: Score = dist + alpha * (count of unvisited nodes closer to this node than to current)
        # This is hard to compute dynamically.
        
        # Simpler dynamic strategy:
        # 1. Calculate distance to all unvisited.
        # 2. Calculate a "potential" for each unvisited node based on its connectivity to other unvisited nodes.
        #    If a node is isolated (far from other unvisited), we might want to visit it sooner?
        #    Or if it's central, visit it later to bridge?
        
        # Let's try the "Repulsion" idea from the observed changes, but implemented dynamically.
        # The observed change "Dynamic Lookahead with Repulsion" scored 0.86 (worse).
        # So we need a different dynamic approach.
        
        # Strategy: **Farthest Insertion** logic adapted for step-by-step?
        # No, that's global.
        
        # Strategy: **Nearest Neighbor with a "Turn" penalty**.
        # Minimize: dist(current, next) + beta * dist(next, destination_or_farthest_unvisited)
        # This encourages moving towards the "mass" of unvisited nodes.
        
        # Let's use a Centroid approach.
        # Calculate centroid of unvisited nodes.
        # Score = dist(current, next) + gamma * dist(next, centroid_unvisited)
        # This pulls the tour towards the center of the remaining points, preventing
        # the tour from drifting away and leaving a big chunk of unvisited nodes far away.
        
        pass

def _get_centroid(unvisited_nodes, distance_matrix):
    """
    Approximate centroid using distances.
    Since we don't have coordinates, we can't compute geometric centroid directly.
    However, we can use the node that minimizes the sum of distances to all other unvisited nodes
    as a proxy for a "central" node.
    This is O(K^2) where K is number of unvisited. K starts at 50, goes down.
    Total cost over the whole tour is sum(K^2) ~ 50^3/3 ~ 40k ops. Very fast.
    """
    if len(unvisited_nodes) == 0:
        return None
    
    unvisited_list = list(unvisited_nodes)
    n_unvisited = len(unvisited_list)
    
    if n_unvisited == 1:
        return unvisited_list[0]
    
    # Find the node in unvisited that has the minimum sum of distances to other unvisited nodes
    # This is the "1-median" of the unvisited set.
    best_node = None
    min_sum = np.inf
    
    # We can vectorize this for speed
    # Sub-matrix of distances between unvisited nodes
    sub_dm = distance_matrix[np.ix_(unvisited_list, unvisited_list)]
    sums = sub_dm.sum(axis=1)
    min_idx = np.argmin(sums)
    return unvisited_list[min_idx]

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time, _current_instance_id, _n_nodes
    
    if _start_time is None:
        _start_time = time.time()
        
    n_unvisited = len(unvisited_nodes)
    
    # Handle edge cases
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback to simple nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Initialize instance state if new
    fid = _fingerprint(distance_matrix)
    if fid != _current_instance_id:
        _current_instance_id = fid
        _n_nodes = distance_matrix.shape[0]
        # No heavy precomputation needed for this dynamic strategy, 
        # but we can precompute nothing really, just use the matrix.
        
    # Dynamic Heuristic:
    # 1. Get distances from current to all unvisited
    dists_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Find the "center" of the unvisited region (1-median)
    # This helps guide the path towards the remaining mass of nodes.
    center_node = _get_centroid(unvisited_nodes, distance_matrix)
    
    # 3. Get distances from the center to all unvisited
    # This is a heuristic: nodes close to the center are "core" nodes.
    # We might want to visit "peripheral" nodes first? Or "core" nodes first?
    # If we visit core nodes first, we might get stuck in the middle.
    # If we visit peripheral nodes first, we might create a long loop around the outside.
    # Actually, for TSP, a "spiral" or "sweep" is often good.
    # Moving towards the center of the remaining points is a robust heuristic to avoid
    # leaving a large empty region that requires a long jump later.
    
    dists_from_center = distance_matrix[center_node, unvisited_nodes]
    
    # 4. Combine scores
    # Score = dist_current_next + alpha * dist_center_next
    # Alpha controls how strongly we are pulled towards the center.
    # If alpha is high, we tend to visit nodes near the center of the unvisited set first.
    # If alpha is low, it behaves like Nearest Neighbor.
    
    # Let's try alpha = 0.5. 
    # Normalizing distances might be better.
    # Let's use a weighted sum.
    
    alpha = 0.3
    
    # To make alpha scale-invariant, we could normalize, but for uniform random points,
    # distances are comparable.
    
    scores = dists_to_unvisited + alpha * dists_from_center
    
    # 5. Pick the node with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
