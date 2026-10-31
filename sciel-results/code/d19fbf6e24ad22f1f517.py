"""TSP constructive heuristic: 2-step lookahead with regret-based selection and destination awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Regret-based 2-step lookahead with destination awareness.
    For each candidate node u in unvisited_nodes, we estimate the cost of
    visiting u next as:
    cost(u) = d(current, u) + min_{v in unvisited, v != u} (d(u, v) + d(v, destination))
    
    We then apply a regret-based correction: the candidate whose cost is closest to the
    minimum cost among all candidates is preferred, as it has the least opportunity cost.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    # d_sub[i, j] is distance between unvisited_nodes[i] and unvisited_nodes[j]
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to the destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Calculate the "best next step cost" for each candidate u
    # For each u (row i in d_sub), we want min_{v != u} (d(u, v) + d(v, destination))
    # Let S be a matrix where S[i, j] = d_sub[i, j] + d_dest[j]
    # We need the min of each row i, excluding the diagonal (j=i).
    
    # Broadcast d_dest to add to each row of d_sub
    # d_dest is shape (n,), d_sub is shape (n, n)
    # We want to add d_dest[j] to d_sub[i, j]
    # So we add d_dest (reshaped to (1, n)) to d_sub
    S = d_sub + d_dest[np.newaxis, :]
    
    # Set diagonal to infinity so u doesn't pick itself as v
    np.fill_diagonal(S, np.inf)
    
    # min_next_cost[i] = min_{j != i} (d_sub[i, j] + d_dest[j])
    min_next_cost = np.min(S, axis=1)
    
    # 5. Total heuristic score for each candidate u
    # score(u) = d(current, u) + min_next_cost(u)
    scores = d_curr + min_next_cost
    
    # 6. Regret-based selection
    # Regret for u is scores[u] - min(scores)
    # We want to minimize regret, so we pick the u with the smallest regret (which is the min score).
    # However, to add a slight diversity or robustness, we can consider the second-best option
    # if the regret is very small, but for simplicity and speed, we stick to min score.
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
