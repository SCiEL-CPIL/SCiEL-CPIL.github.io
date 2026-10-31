"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_BUDGET = 145.0 # Tighter margin

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_BUDGET
    return _DEADLINE

def _two_opt_improve_path(path, dist_matrix, max_passes=5):
    """
    Fast 2-Opt on a list of nodes.
    """
    n = len(path)
    if n < 4:
        # Calculate cost
        cost = 0.0
        for i in range(n-1):
            cost += dist_matrix[path[i], path[i+1]]
        return path, cost
    
    path = path[:]
    cost = 0.0
    for i in range(n-1):
        cost += dist_matrix[path[i], path[i+1]]
        
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Vectorize inner loop? Hard for dynamic reversal.
        # N is small (<=50). Python loop is okay.
        for i in range(n - 2):
            ui = path[i]
            vi = path[i+1]
            for j in range(i + 1, n - 1):
                uj = path[j]
                vj = path[j+1]
                
                d_ui = dist_matrix[ui, vi]
                d_uj = dist_matrix[uj, vj]
                d_uuj = dist_matrix[ui, uj]
                d_vvi = dist_matrix[vi, vj] # Wait, edges are (i, i+1) and (j, j+1). New: (i, j) and (i+1, j+1).
                # Correct edges:
                # Remove: (ui, vi) and (uj, vj)
                # Add: (ui, uj) and (vi, vj) -- NO.
                # Path: ... ui - vi ... uj - vj ...
                # Reverse vi ... uj.
                # New Path: ... ui - uj ... vi - vj ...
                # Edges removed: (ui, vi), (uj, vj)
                # Edges added: (ui, uj), (vi, vj)
                
                delta = dist_matrix[ui, uj] + dist_matrix[vi, vj] - d_ui - d_uj
                
                if delta < -1e-10:
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    cost += delta
                    improved = True
                    # Restart inner loop? Standard is to continue, but restart is safer for convergence.
                    # For speed, continue.
                    
    return path, cost

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    deadline = _get_deadline()
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Fallback to NN if time is low
    if time.time() > deadline - 15.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    best_score = float('inf')
    best_node = -1
    
    unvisited_list = list(unvisited_nodes)
    
    for v in unvisited_list:
        cost_to_v = distance_matrix[current_node, v]
        
        # If cost_to_v is already > best_score, we can't beat it?
        # No, because comp_cost can be 0? No, comp_cost is positive.
        # So if cost_to_v >= best_score, we can skip?
        # Only if we have a lower bound on comp_cost. MST is a lower bound.
        # But calculating MST for each v is expensive.
        # Just skip if cost_to_v is large?
        # Let's just compute.
        
        remaining = [u for u in unvisited_list if u != v]
        
        # Build NN path from v to destination
        # Use NumPy for speed
        rem_set = set(remaining)
        path = [v]
        curr = v
        
        while rem_set:
            # Get distances from curr to all in rem_set
            # Convert rem_set to list for numpy
            rem_list = list(rem_set)
            dists = distance_matrix[curr, rem_list]
            min_idx = np.argmin(dists)
            next_node = rem_list[min_idx]
            path.append(next_node)
            curr = next_node
            rem_set.remove(next_node)
            
        path.append(destination_node)
        
        # Optimize
        # Skip 2-Opt if path is small or time is tight
        if len(path) > 4 and time.time() < deadline - 5.0:
            _, comp_cost = _two_opt_improve_path(path, distance_matrix, max_passes=5)
        else:
            comp_cost = 0.0
            for i in range(len(path)-1):
                comp_cost += distance_matrix[path[i], path[i+1]]
                
        total_score = cost_to_v + comp_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = v
            
    return int(best_node)

# EVOLVE-BLOCK-END
