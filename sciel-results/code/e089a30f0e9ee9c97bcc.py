"""Global 2-opt improvement heuristic for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def _build_route_nn(current_node, depot, unvisited, rest_capacity, demands, distance_matrix):
    """
    Build a route starting from current_node using Nearest Neighbor.
    Returns the list of nodes in the route (excluding depot return).
    """
    route = [current_node]
    current = current_node
    remaining_cap = rest_capacity
    remaining = set(unvisited)
    
    while remaining:
        best_next = -1
        best_dist = float('inf')
        
        for node in remaining:
            if demands[node] > remaining_cap:
                continue
            d = distance_matrix[current, node]
            if d < best_dist:
                best_dist = d
                best_next = node
        
        if best_next == -1:
            break
            
        route.append(best_next)
        current = best_next
        remaining_cap -= demands[current]
        remaining.remove(current)
        
    return route

def _two_opt_improve(route, depot, demands, distance_matrix, rest_capacity):
    """
    Improve the route using 2-opt while respecting capacity constraints.
    Returns the improved route.
    """
    if len(route) < 2:
        return route
        
    improved = True
    # Limit iterations to prevent excessive time usage
    max_iters = 100
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < _DEADLINE:
        improved = False
        iter_count += 1
        n = len(route)
        
        for i in range(n - 1):
            if time.time() > _DEADLINE:
                return route
            for j in range(i + 1, n):
                # Calculate gain from reversing segment i+1 to j
                # New edges: (i, j) and (j+1, i+1) if j < n-1, else (j+1, depot)
                # Old edges: (i, i+1) and (j, j+1) if j < n-1, else (j, depot)
                
                node_i = route[i]
                node_i1 = route[i+1]
                node_j = route[j]
                
                if j + 1 < n:
                    node_j1 = route[j+1]
                else:
                    node_j1 = depot
                    
                # Check capacity feasibility for the new route
                # Reversing the segment [i+1, j] changes the order of demands.
                # Since capacity is a cumulative constraint, we must check if the new order violates capacity.
                # For 2-opt, the total demand in the route is constant, but the prefix sums change.
                
                # Efficient capacity check:
                # The prefix sum up to i is unchanged.
                # The segment i+1..j is reversed.
                # The suffix j+1..end is unchanged.
                
                # Let's just reconstruct the prefix sums for the modified route if a swap looks promising?
                # Or precompute prefix sums?
                # Given N is small (50), we can check capacity by simulating the prefix sums of the new order.
                # But doing this for every pair (i,j) is O(N^3) per 2-opt pass, which is 125k ops.
                # With 100 iterations, that's 12.5M ops. Might be slow in Python.
                
                # Optimization: Only check capacity if the distance gain is positive.
                d_old = distance_matrix[node_i, node_i1] + distance_matrix[node_j, node_j1]
                d_new = distance_matrix[node_i, node_j] + distance_matrix[node_i1, node_j1]
                
                if d_new < d_old:
                    # Potential improvement. Check capacity.
                    # Construct new route
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    
                    # Check capacity
                    cap_ok = True
                    curr_cap = rest_capacity
                    for node in new_route:
                        curr_cap -= demands[node]
                        if curr_cap < 0:
                            cap_ok = False
                            break
                    
                    if cap_ok:
                        route = new_route
                        improved = True
                        # Restart outer loop? Or continue? 
                        # Standard 2-opt restarts to find local optimum faster.
                        # But restarting breaks the 'improved' flag logic slightly.
                        # Let's just break inner and continue outer? No, standard is to restart.
                        # However, for speed, we can just continue scanning.
                        # Actually, restarting is safer for convergence.
                        # But to save time, let's just mark improved and continue.
                        # If we break, we miss other improvements in this pass.
                        # Let's break inner and outer to restart the pass.
                        break
            if improved:
                break
                
    return route

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node by constructing a full route with NN, improving it with 2-opt,
    and returning the first node in the improved route (or depot if route is empty).
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # 1. Build initial route using Nearest Neighbor
    route = _build_route_nn(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)
    
    # The route includes current_node at index 0.
    # If the route only has current_node, it means no feasible neighbors were found.
    if len(route) <= 1:
        return 0

    # 2. Improve the route using 2-opt
    # Note: The 'rest_capacity' passed to _two_opt_improve is the capacity available *after* visiting current_node?
    # No, the interface says 'rest_capacity' is the capacity remaining *at* current_node.
    # So the route built by _build_route_nn starts at current_node and uses rest_capacity.
    # The capacity check in _two_opt_improve must use the same rest_capacity.
    
    improved_route = _two_opt_improve(route, depot, demands, distance_matrix, rest_capacity)
    
    # 3. The next node is the second node in the improved route (index 1)
    # Because index 0 is current_node.
    if len(improved_route) > 1:
        return int(improved_route[1])
    else:
        return 0

# EVOLVE-BLOCK-END
