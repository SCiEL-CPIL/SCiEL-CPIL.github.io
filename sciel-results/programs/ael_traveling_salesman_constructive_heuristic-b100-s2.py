# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # Pick the one that gives shorter total: current->A->dest vs current->B->dest
        a, b = unvisited_nodes
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, destination_node]
        return int(a if cost_a <= cost_b else b)
    
    # For each candidate, estimate the remaining tour cost using a greedy nearest-neighbor completion
    # with a 2-opt local improvement pass for better quality.
    
    best_cost = np.inf
    best_node = unvisited_nodes[0]
    
    for cand in unvisited_nodes:
        # Simulate greedy completion from cand
        curr = cand
        remaining = np.delete(unvisited_nodes, np.where(unvisited_nodes == cand)[0])
        
        total_cost = distance_matrix[current_node, cand]
        
        # Build the greedy tour path
        path = [cand]
        
        while len(remaining) > 0:
            # Find nearest neighbor in remaining
            dists = distance_matrix[curr, remaining]
            min_idx = np.argmin(dists)
            next_node = remaining[min_idx]
            total_cost += dists[min_idx]
            
            # Remove next_node from remaining
            remaining = np.delete(remaining, min_idx)
            curr = next_node
            path.append(curr)
            
            # Early termination if cost already exceeds best
            if total_cost >= best_cost:
                break
        
        # Add cost to close the tour back to destination
        if len(remaining) == 0:
            total_cost += distance_matrix[curr, destination_node]
            
            # Apply 2-opt improvement on the path
            improved_cost = _two_opt_improve(path, distance_matrix, current_node, destination_node)
            
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_node = cand
                
    return int(best_node)

def _two_opt_improve(path, distance_matrix, start_node, end_node):
    """Apply 2-opt local search to improve the path. Returns the improved total cost."""
    n = len(path)
    if n < 3:
        # Calculate direct cost
        cost = distance_matrix[start_node, path[0]]
        for i in range(n - 1):
            cost += distance_matrix[path[i], path[i + 1]]
        cost += distance_matrix[path[-1], end_node]
        return cost
    
    # Calculate initial cost
    def calc_cost(p):
        c = distance_matrix[start_node, p[0]]
        for i in range(len(p) - 1):
            c += distance_matrix[p[i], p[i + 1]]
        c += distance_matrix[p[-1], end_node]
        return c
    
    current_cost = calc_cost(path)
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Try reversing path[i+1:j+1]
                new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                new_cost = calc_cost(new_path)
                if new_cost < current_cost - 1e-10:
                    path = new_path
                    current_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    
    return current_cost
# EVOLVE-BLOCK-END
