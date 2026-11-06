"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before the 120s kill

def _time_exceeded():
    return time.time() > _DEADLINE

def _two_opt_path(path, dm):
    """
    Applies 2-Opt to a path (not a cycle).
    path: list of node indices.
    dm: distance matrix (numpy array).
    Returns the optimized path and its length.
    """
    if len(path) < 3:
        return path, sum(dm[path[i], path[i+1]] for i in range(len(path)-1))
    
    improved = True
    max_iter = 10  # Limit iterations for speed
    iter_count = 0
    
    # Precompute edge lengths for faster update? 
    # For small paths, direct calculation is fine.
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        n = len(path)
        for i in range(n - 2):
            # i is the index before the segment to reverse
            # We reverse path[i+1 : j+1]
            for j in range(i + 2, n):
                # Nodes involved:
                # path[i], path[i+1], path[j], path[j+1] (if exists)
                
                u = path[i]
                v = path[i+1]
                w = path[j]
                
                if j < n - 1:
                    x = path[j+1]
                    old_len = dm[u, v] + dm[w, x]
                    new_len = dm[u, w] + dm[v, x]
                else:
                    # j is the last node
                    old_len = dm[u, v]
                    new_len = dm[u, w]
                    
                if new_len < old_len - 1e-10:
                    # Reverse path[i+1 : j+1]
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    improved = True
    
    # Calculate final length
    total_len = sum(dm[path[i], path[i+1]] for i in range(len(path)-1))
    return path, total_len

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Lookahead Nearest Neighbor with 2-Opt Refinement.
    
    For each feasible next node, simulate the completion of the current route
    using Nearest Neighbor, then optimize that segment with 2-Opt.
    Select the node that results in the shortest optimized route segment.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard: if we are running out of time, fall back to simple nearest
    if _time_exceeded():
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    dm = distance_matrix
    n_feasible = len(feasible)
    
    best_node = -1
    best_score = np.inf

    # We will evaluate each feasible candidate
    for c in feasible:
        # Tentative route starts with current -> c
        tentative_route = [current_node, c]
        remaining_cap = rest_capacity - demands[c]
        
        # Nodes still unvisited in this simulation
        # We copy the feasible list and remove c
        # Note: We should only consider nodes that were feasible initially? 
        # Or should we re-evaluate feasibility? 
        # In CVRP, if a node didn't fit initially, it won't fit later if we add more nodes.
        # So we can just use the `feasible` list minus `c`.
        
        unvisited_in_sim = [i for i in feasible if i != c]
        
        # Greedy Nearest Neighbor fill
        current = c
        while remaining_cap > 0 and unvisited_in_sim:
            # Find nearest neighbor in unvisited_in_sim that fits
            # Vectorized search
            u_arr = np.array(unvisited_in_sim)
            # Filter by demand
            valid_mask = demands[u_arr] <= remaining_cap
            if not np.any(valid_mask):
                break
            valid_u = u_arr[valid_mask]
            valid_dists = dm[current, valid_u]
            
            min_idx = np.argmin(valid_dists)
            next_node = valid_u[min_idx]
            next_dist = valid_dists[min_idx]
            
            tentative_route.append(next_node)
            remaining_cap -= demands[next_node]
            
            # Remove next_node from unvisited_in_sim
            unvisited_in_sim.remove(next_node)
            
            current = next_node
            
            # Time check inside inner loop
            if _time_exceeded():
                # Fallback to nearest neighbor for the rest of this candidate evaluation
                # Just return the current best or break out
                break

        # Apply 2-Opt to the tentative route
        optimized_route, route_len = _two_opt_path(tentative_route, dm)
        
        # The score is the length of this route segment.
        # Note: This doesn't include the return to depot, but that is constant for all candidates
        # IF the route ends at the same point? No, the route ends at different points.
        # However, the cost to return to depot from the last node varies.
        # Should we include dm[last_node, depot] in the score?
        # Yes, the total route cost is sum(edges) + dm[last, depot].
        # Since we are comparing candidates, we should include the return leg to depot
        # to ensure we don't pick a route that ends far from the depot if it saves internal distance.
        
        total_cost = route_len + dm[optimized_route[-1], depot]
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = c
            
        # Time check after each candidate
        if _time_exceeded():
            break

    if best_node == -1:
        # Fallback to nearest if something went wrong or timeout occurred before selection
        idx = np.argmin(distance_matrix[current_node, feasible])
        best_node = int(feasible[idx])
    else:
        best_node = int(best_node)
        
    return best_node

# EVOLVE-BLOCK-END
