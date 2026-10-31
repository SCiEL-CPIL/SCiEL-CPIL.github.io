"""Rewritten program: TSP constructive heuristic with constrained 2-opt on partial tour."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_global_tour = []

def _get_deadline():
    """Return a deadline that is 3 seconds before the 150s limit (safety margin)."""
    return _start_time + 145.0

def _two_opt_partial(tour, distance_matrix, max_iter=10):
    """
    Perform 2-opt improvement on the partial tour.
    Constraint: The last node in the tour must remain the last node.
    This means we only reverse segments A[i:j] where j < len(tour) - 1.
    """
    n = len(tour)
    if n < 3:
        return tour
    
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(n - 1):
            if time.time() > _get_deadline():
                return tour
            for j in range(i + 1, n - 1):
                # Calculate delta
                # Old edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[j+1])
                # Note: if i==0, there is no tour[i-1]. 
                # For open path starting at tour[0], the start is fixed.
                # If i==0, we are reversing the prefix.
                # Old: d(tour[0], tour[1]) + d(tour[j], tour[j+1])
                # New: d(tour[0], tour[j]) + d(tour[1], tour[j+1])
                
                idx_i_minus_1 = i - 1 if i > 0 else None
                idx_j_plus_1 = j + 1
                
                if i == 0:
                    d_old = distance_matrix[tour[0], tour[1]] + distance_matrix[tour[j], tour[j+1]]
                    d_new = distance_matrix[tour[0], tour[j]] + distance_matrix[tour[1], tour[j+1]]
                else:
                    d_old = distance_matrix[tour[i-1], tour[i]] + distance_matrix[tour[j], tour[j+1]]
                    d_new = distance_matrix[tour[i-1], tour[j]] + distance_matrix[tour[i], tour[j+1]]
                
                if d_new < d_old - 1e-10:
                    # Reverse segment tour[i:j+1]
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy:
    1. Maintain a global partial tour.
    2. Select the next node using a greedy heuristic (Nearest Neighbor + 2-step lookahead).
    3. Append the node to the partial tour.
    4. Apply constrained 2-opt to the partial tour (keeping the last node fixed).
    5. Return the selected node.
    """
    global _global_tour
    
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        # Last node, no choice
        # Add to global tour for consistency, though it won't be used for selection
        if _global_tour:
            _global_tour.append(int(unvisited_nodes[0]))
        return int(unvisited_nodes[0])
    
    # Check deadline early
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Initialize global tour if this is the first call
    if not _global_tour:
        _global_tour = [int(current_node)]
    
    # Compute distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from unvisited nodes to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 2-step lookahead: min distance to other unvisited nodes
    # This helps avoid picking a node that is isolated
    dists_between = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(dists_between, np.inf)
    min_next_dist = np.min(dists_between, axis=1)
    
    # Weighted combination
    # Tuned weights:
    # alpha: weight for destination (small, to encourage closing the loop eventually)
    # beta: weight for 2-step lookahead (moderate, to avoid bad local choices)
    alpha = 0.05
    beta = 0.2
    
    scores = dists_from_current + beta * min_next_dist + alpha * dists_to_dest
    
    best_idx = np.argmin(scores)
    next_node = int(unvisited_nodes[best_idx])
    
    # Add to global tour
    _global_tour.append(next_node)
    
    # Apply constrained 2-opt
    # We limit iterations to keep it fast
    _two_opt_partial(_global_tour, distance_matrix, max_iter=5)
    
    return next_node

# EVOLVE-BLOCK-END
