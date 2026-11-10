"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = None
_TIMEOUT = 115.0
_DEADLINE = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic to pick the next customer or return to depot.
    
    Uses a savings-based scoring function that prioritizes nodes far from depot
    to create longer routes and reduce vehicle count, while still respecting
    capacity constraints.
    
    Score(j) = dist(current, j) + alpha * (dist(current, depot) + dist(depot, j) - dist(current, j))
    Simplified: Score(j) = (1-alpha) * dist(current, j) + alpha * (dist(current, depot) + dist(depot, j))
    
    With alpha > 0.5, this favors nodes that are far from the depot (large savings),
    which helps pack more customers into each route.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array if it's a list
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes)
    else:
        unvisited_arr = unvisited_nodes
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Compute distances
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    dist_current_to_depot = distance_matrix[current_node, depot]
    
    # 1-step lookahead: for each candidate j, estimate the best next node k after j
    # and compute total route extension cost: dist(current,j) + dist(j,k)
    # This encourages forming longer chains of high-savings connections.
    
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # For each candidate j, find the best next node k (excluding j itself)
    # that maximizes savings from j to k
    # savings(j,k) = dist(j, depot) + dist(depot, k) - dist(j, k)
    
    # dist matrix from each feasible node to all feasible nodes
    dist_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    dists_from_feasible_to_depot = distance_matrix[feasible_nodes, depot]
    
    # For each j, compute savings to all k, then find max savings (min cost)
    # savings[j,k] = dist(j,depot) + dist(depot,k) - dist(j,k)
    # = dists_from_feasible_to_depot[:,None] + dists_from_feasible_to_depot[None,:] - dist_feasible
    
    savings_matrix = (dists_from_feasible_to_depot[:, None] + 
                      dists_from_feasible_to_depot[None, :] - 
                      dist_feasible)
    
    # Set diagonal to -inf so we don't pick j as its own next
    np.fill_diagonal(savings_matrix, -np.inf)
    
    # For each j, the best next node k maximizes savings
    # If no valid k exists (n_feasible==1 handled above), this is safe
    best_savings_from_j = np.max(savings_matrix, axis=1)
    
    # Now compute the total cost for choosing j as next:
    # cost(j) = dist(current, j) - alpha * (savings(current->j) + gamma * best_savings_from_j[j])
    # where savings(current->j) = dist(current,depot) + dist(depot,j) - dist(current,j)
    
    alpha = 0.9
    gamma = 0.5  # weight for lookahead term
    
    savings_current_to_j = dist_current_to_depot + dists_to_depot - dists_from_current
    
    # Combined score: lower is better
    # Base savings score plus lookahead bonus
    scores = dists_from_current - alpha * (savings_current_to_j + gamma * best_savings_from_j)
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
