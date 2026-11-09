"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    # Simple cache using id, assuming matrix doesn't change
    if not hasattr(_get_dm, 'cache'):
        _get_dm.cache = {}
    key = id(distance_matrix)
    if key not in _get_dm.cache:
        _get_dm.cache[key] = np.asarray(distance_matrix)
    return _get_dm.cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 5 seconds
    return (time.time() - _start_time) > 145

def _simulate_route_length(start_node, current_node, unvisited_set, current_cap, demands, dm, depot):
    """
    Simulate the rest of the route using Nearest Neighbor heuristic.
    Returns the total length of the route segment starting from current_node,
    visiting all feasible nodes in unvisited_set (respecting capacity),
    and ending at depot.
    """
    sim_current = current_node
    sim_cap = current_cap
    sim_len = 0.0
    remaining = list(unvisited_set)
    
    while remaining:
        # Find feasible nodes
        feasible = [n for n in remaining if demands[n] <= sim_cap]
        if not feasible:
            break
        
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            # Pick nearest feasible node
            # Using numpy for speed if possible, but list is small
            dists = [dm[sim_current, n] for n in feasible]
            min_idx = np.argmin(dists)
            next_node = feasible[min_idx]
            
        sim_len += dm[sim_current, next_node]
        sim_current = next_node
        sim_cap -= demands[next_node]
        remaining.remove(next_node)
        
    # Return to depot
    sim_len += dm[sim_current, depot]
    return sim_len

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    Uses a 1-step lookahead heuristic with Nearest Neighbor simulation for the remainder.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to nearest feasible
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Lookahead: For each feasible candidate, simulate the rest of the route
    # and pick the candidate that minimizes the total route length.
    
    best_score = float('inf')
    best_node = depot # Default to closing if something goes wrong
    
    # We simulate starting from current_node -> candidate -> ... -> depot
    # The cost is d(current, candidate) + simulated_rest(candidate)
    
    feasible_set = set(feasible)
    
    for v in feasible:
        # Immediate cost
        cost_to_v = dm[current_node, v]
        
        # Remaining nodes for simulation
        # We must exclude v from the unvisited set for the simulation
        # The simulation will start from v, with capacity rest_capacity - demands[v]
        # It will visit nodes from feasible - {v}
        
        # Optimization: If capacity is very tight, simulation might be short
        new_cap = rest_capacity - demands[v]
        
        # If no capacity left, we must go to depot
        if new_cap < min(demands[n] for n in feasible if n != v) if len(feasible) > 1 else 0:
             # Actually, just simulate. If no feasible nodes, it goes to depot.
             pass
             
        # Simulate the rest
        # To speed up, we can pass the list of other feasible nodes
        other_feasible = [n for n in feasible if n != v]
        
        # Use the simulation function
        # Note: _simulate_route_length assumes it starts from 'start_node' (which is v)
        # and visits 'unvisited_set' (other_feasible)
        
        # We need to be careful: the simulation function calculates the length
        # of the path starting from 'current_node' (which we pass as v)
        # and ending at depot.
        
        sim_len = _simulate_route_length(v, v, set(other_feasible), new_cap, demands, dm, depot)
        
        total_len = cost_to_v + sim_len
        
        if total_len < best_score:
            best_score = total_len
            best_node = v
            
    return int(best_node)
# EVOLVE-BLOCK-END
