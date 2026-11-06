import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0
_CAPACITY = 50.0

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def _quick_lookahead(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix, max_depth=3):
    """
    Fast lookahead using the same dynamic heuristic.
    Returns the total distance of the simulated path (excluding return to depot for the last node? 
    No, let's return total path length including return to depot at the end of the simulation).
    """
    if len(unvisited_nodes) == 0:
        return 0.0
    
    total_dist = 0.0
    current = current_node
    remaining_cap = rest_capacity
    remaining_set = set(unvisited_nodes)
    
    for _ in range(max_depth):
        if not remaining_set:
            break
        if _time_exceeded():
            break
            
        # Find feasible nodes
        feasible = [n for n in remaining_set if demands[n] <= remaining_cap]
        if not feasible:
            break
            
        # Score feasible nodes
        best_node = None
        best_score = float('inf')
        
        # Urgency factor
        urgency = 1.0 - (remaining_cap / _CAPACITY)
        
        for n in feasible:
            d_c_n = distance_matrix[current, n]
            d_n_d = distance_matrix[n, depot]
            score = d_c_n + urgency * d_n_d
            
            if score < best_score:
                best_score = score
                best_node = n
        
        if best_node is None:
            break
            
        total_dist += distance_matrix[current, best_node]
        current = best_node
        remaining_cap -= demands[best_node]
        remaining_set.remove(best_node)
        
    # Return to depot
    total_dist += distance_matrix[current, depot]
    
    return total_dist

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes
    unvisited_demands = demands[unvisited]
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    
    if _time_exceeded():
        # Fallback: Nearest neighbor
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Calculate scores
    dists_current = distance_matrix[current_node, feasible_nodes]
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # Urgency factor: higher when capacity is low
    urgency = 1.0 - (rest_capacity / _CAPACITY)
    
    # Base score: Distance to current + Urgency * Distance to Depot
    base_scores = dists_current + urgency * dists_depot
    
    # Identify top candidates for lookahead
    # If urgency is very high, we might want to consider returning to depot vs visiting a node.
    # Cost to return to depot: distance_matrix[current_node, depot]
    # Cost to visit best node: base_scores[min_idx] + ...
    
    top_k = min(3, n_feasible)
    top_indices = np.argsort(base_scores)[:top_k]
    
    # If we are very close to deadline, just pick the best base score
    if _time_exceeded() or n_feasible < 3:
         return int(feasible_nodes[top_indices[0]])

    lookahead_scores = np.zeros(top_k)
    for k, idx in enumerate(top_indices):
        candidate_node = feasible_nodes[idx]
        rem_cap = rest_capacity - demands[candidate_node]
        
        # If no capacity left, the route ends after this node
        if rem_cap <= 0:
            lookahead_scores[k] = distance_matrix[current_node, candidate_node] + distance_matrix[candidate_node, depot]
        else:
            # Simulate the rest
            rem_nodes_list = [n for n in unvisited_nodes if n != candidate_node]
            lookahead_scores[k] = _quick_lookahead(candidate_node, depot, rem_nodes_list, rem_cap, demands, distance_matrix, max_depth=3)
        
        if _time_exceeded():
            break

    best_local_idx = np.argmin(lookahead_scores)
    best_global_idx = top_indices[best_local_idx]
    
    return int(feasible_nodes[best_global_idx])
