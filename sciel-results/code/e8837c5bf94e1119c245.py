"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Filter to feasible nodes (demand fits in remaining capacity)
    # Use numpy for speed
    unvisited_arr = np.array(unvisited_nodes)
    demands_arr = np.array(demands)
    
    # Create mask for feasible nodes
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Calculate composite scores: distance from current + distance to depot
    # This penalizes nodes that are far from the depot, encouraging compact routes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    scores = dist_from_current + dist_to_depot
    
    # Implement Regret-1 heuristic
    # Regret(j) = min(score(k) for k != j) - score(j)
    # We want to pick the node with the highest regret
    # This avoids picking a node that is "cheap" now but leaves us with expensive options later
    
    # To do this efficiently with numpy:
    # Sort scores
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    
    # For each node, we need the minimum score of the *other* nodes.
    # If a node is the global minimum (index 0 in sorted), its min_other is sorted_scores[1].
    # If a node is the second minimum (index 1), its min_other is sorted_scores[0].
    # If a node is index i > 1, its min_other is sorted_scores[0].
    
    min_other_scores = np.full(len(feasible_nodes), sorted_scores[0])
    
    # Handle the case where the best node's min_other is the second best
    if len(feasible_nodes) > 1:
        # The node at sorted_indices[0] has min_other = sorted_scores[1]
        min_other_scores[sorted_indices[0]] = sorted_scores[1]
        # The node at sorted_indices[1] has min_other = sorted_scores[0]
        min_other_scores[sorted_indices[1]] = sorted_scores[0]
        # All other nodes have min_other = sorted_scores[0] (which is already set)
        
    regrets = min_other_scores - scores
    
    # Select node with max regret
    # If tie in regret, select lower score (better immediate cost)
    # We can do this by creating a composite key: (regret, -score) and taking max
    # Or simply: find max regret, then among those, find min score
    
    max_regret = np.max(regrets)
    candidates_max_regret = np.where(regrets == max_regret)[0]
    
    if len(candidates_max_regret) == 1:
        best_idx_in_feasible = candidates_max_regret[0]
    else:
        # Among those with max regret, pick the one with min score
        sub_scores = scores[candidates_max_regret]
        best_sub_idx = np.argmin(sub_scores)
        best_idx_in_feasible = candidates_max_regret[best_sub_idx]
        
    return int(feasible_nodes[best_idx_in_feasible])

# EVOLVE-BLOCK-END
