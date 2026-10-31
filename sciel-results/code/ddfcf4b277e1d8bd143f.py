"""Rewritten program: TSP constructive heuristic using edge-density-aware multi-step lookahead."""
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
    
    Strategy: Edge-density-aware 3-step lookahead with adaptive weighting.
    
    For each candidate node u in unvisited_nodes, we estimate the cost as:
    cost(u) = d(current, u) 
             + alpha * min_{v != u} [d(u, v) + beta * min_{w != u,v} d(v, w)]
             + gamma * d(u, destination) * (1 / (1 + remaining_fraction))
    
    The key structural change: we use a 3-step lookahead that considers
    the "connectivity" of the path ahead. A node that is close to current
    but isolated from the rest of the unvisited set (high minimum distance
    to any other unvisited node) is penalized. We also adaptively weight
    the destination term based on how many nodes remain.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    if n == 2:
        # For 2 nodes, pick the one that minimizes the total path
        # current -> u -> v -> destination
        u0 = int(unvisited_nodes[0])
        u1 = int(unvisited_nodes[1])
        cost0 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        return u0 if cost0 <= cost1 else u1
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. For 3-step lookahead:
    # For each candidate i, find min over v != i of [d(i, v) + beta * min over w != i,v of d(v, w)]
    
    # First, compute for each node v, the minimum distance to any other node w (w != v)
    # This is the "connectivity" of node v within the unvisited set
    np.fill_diagonal(d_sub, np.inf)
    min_conn = np.min(d_sub, axis=1)  # min distance from each node to any other
    
    # Now for each pair (i, v) where v != i:
    # We want d(i, v) + beta * min_{w != i, w != v} d(v, w)
    # The inner min is min over w in unvisited, w != i, w != v of d(v, w)
    
    # To compute this efficiently:
    # For each v, we need the min distance to w where w != i and w != v
    # This is the second-smallest distance from v, unless the smallest is to i
    
    # Let's compute the two smallest distances from each node v to other nodes
    # d_sub has inf on diagonal
    sorted_d = np.sort(d_sub, axis=1)  # shape (n, n)
    min1 = sorted_d[:, 0]  # smallest distance from each v
    min2 = sorted_d[:, 1]  # second smallest distance from each v
    
    # For pair (i, v): the inner min is:
    # If the node that v is closest to (argmin of d_sub[v, :]) is i, then use min2[v]
    # Otherwise, use min1[v]
    
    # We need argmin for each row of d_sub
    argmin_idx = np.argmin(d_sub, axis=1)  # for each v, which node is closest
    
    # Build the matrix: for each (i, v), inner_min[i, v] = min_{w != i, w != v} d(v, w)
    # This is: if argmin_idx[v] == i, then min2[v], else min1[v]
    
    # Create a matrix where entry [i, v] is min2[v] if argmin_idx[v] == i, else min1[v]
    # We can do this with a mask
    is_closest_to_i = (argmin_idx[np.newaxis, :] == np.arange(n)[:, np.newaxis])  # shape (n, n), [i, v]
    
    inner_min = np.where(is_closest_to_i, min2[np.newaxis, :], min1[np.newaxis, :])
    
    # Now the 2-step lookahead cost for candidate i via next node v:
    # step_cost[i, v] = d_sub[i, v] + beta * inner_min[i, v]
    # But we only consider v != i, which is already handled by inf diagonal
    
    beta = 0.4
    step_cost = d_sub + beta * inner_min
    
    # For each candidate i, the best 2-step cost is min over v != i of step_cost[i, v]
    min_step_cost = np.min(step_cost, axis=1)
    
    # 5. Adaptive destination weighting
    # As fewer nodes remain, the destination becomes more important
    remaining_fraction = n / 50.0  # assuming ~50 cities
    gamma = 0.3 * (1.0 / (1.0 + remaining_fraction))
    
    # 6. Combine all terms
    alpha = 0.6
    scores = d_curr + alpha * min_step_cost + gamma * d_dest
    
    # 7. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
