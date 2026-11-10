"""CVRP heuristic: Route-Level Look-Ahead with Nearest-Neighbor completion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None


def _complete_route_nn(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Complete the current route using a simple nearest-neighbor heuristic.
    Returns the total additional distance to complete the route (including return to depot).
    """
    if len(unvisited_nodes) == 0:
        return distance_matrix[current_node, depot]
    
    # Work with copies
    remaining = list(unvisited_nodes)
    cap = rest_capacity
    cur = current_node
    total_dist = 0.0
    
    while remaining:
        # Find feasible nodes
        feasible = [n for n in remaining if demands[n] <= cap]
        if not feasible:
            break
        
        # Pick nearest feasible node
        best_node = min(feasible, key=lambda n: distance_matrix[cur, n])
        total_dist += distance_matrix[cur, best_node]
        cap -= demands[best_node]
        cur = best_node
        remaining.remove(best_node)
    
    # Return to depot
    total_dist += distance_matrix[cur, depot]
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a route-level look-ahead heuristic.
    For each feasible candidate, simulates completing the rest of the route
    with nearest-neighbor and picks the candidate that minimizes total route length.
    
    Returns 0 (depot) if no feasible node can be added or if closing is better.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to list if needed
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_list = list(unvisited_nodes)
    else:
        unvisited_list = list(unvisited_nodes)
    
    # If no unvisited nodes, return depot
    if len(unvisited_list) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible = [n for n in unvisited_list if demands[n] <= rest_capacity]
    
    if not feasible:
        return int(depot)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        best_node = min(feasible, key=lambda n: distance_matrix[current_node, n])
        return int(best_node)
    
    # For each feasible candidate, compute the total route length if we pick it
    # and then complete with NN
    best_candidate = None
    best_total = float('inf')
    
    # Cost of closing now (go straight to depot)
    close_cost = distance_matrix[current_node, depot]
    
    for candidate in feasible:
        # Distance from current to candidate
        d_cur_cand = distance_matrix[current_node, candidate]
        
        # Remaining capacity after visiting candidate
        new_cap = rest_capacity - demands[candidate]
        
        # Remaining unvisited nodes (excluding candidate)
        remaining = [n for n in unvisited_list if n != candidate]
        
        # Complete the route from candidate with NN
        if remaining:
            completion_cost = _complete_route_nn(candidate, depot, remaining, new_cap, demands, distance_matrix)
        else:
            completion_cost = distance_matrix[candidate, depot]
        
        total = d_cur_cand + completion_cost
        
        if total < best_total:
            best_total = total
            best_candidate = candidate
    
    # Decision: close if best candidate's total is worse than closing now
    # We need to compare: total route length with candidate vs without
    # Without candidate: close_cost (go to depot)
    # With candidate: best_total
    # But we also need to account for the fact that if we close, we're done with this route
    # The comparison should be: is it cheaper to add candidate and complete, or just close?
    # Actually, the total route length includes all future routes too, but we're only
    # looking at the current route. So we compare:
    # - If we close: cost = close_cost
    # - If we pick candidate: cost = best_total
    # We should pick the candidate if best_total < close_cost + some margin
    # But this is tricky because picking a candidate might allow more nodes to be
    # visited in this route, reducing the number of routes needed.
    
    # A simpler heuristic: pick the candidate if it's not too expensive
    # Use a threshold: pick if best_total < close_cost * 1.2 (20% tolerance)
    # Or just pick if best_total < close_cost
    
    # Actually, let's use a more principled approach:
    # The savings from picking candidate j is:
    # savings = d_cur_depot + d_depot_j - d_cur_j
    # If savings > 0, it's beneficial to pick j
    # But we also need to consider the impact on future nodes
    
    # For now, use a simple rule: pick the best candidate if its total cost
    # is not significantly worse than closing
    if best_candidate is not None and best_total <= close_cost * 1.1:
        return int(best_candidate)
    else:
        return int(depot)
# EVOLVE-BLOCK-END
