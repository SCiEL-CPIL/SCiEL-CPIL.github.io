"""Traveling salesman constructive heuristic with Hybrid Lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 110.0  # Safety margin for 120s/150s limit

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _nn_completion_cost(current_node, unvisited_nodes, distance_matrix, start_node):
    """
    Fast nearest neighbor heuristic to estimate remaining cost.
    Returns the estimated cost of the path from current_node through 
    all unvisited_nodes and back to start_node.
    """
    if len(unvisited_nodes) == 0:
        return 0.0
    
    remaining = list(unvisited_nodes)
    curr = current_node
    total = 0.0
    
    while remaining:
        rem_arr = np.array(remaining)
        dists = distance_matrix[curr, rem_arr]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        total += dists[min_idx]
        curr = next_node
        remaining.pop(min_idx)
    
    # Add cost to return to start
    total += distance_matrix[curr, start_node]
    
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    # If only one node left, must pick it.
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # If two nodes left, pick the one that gives shorter closing
    if len(unvisited_nodes) == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        start_node = _current_tour[0]
        # Option 1: a then b, then back to start
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, start_node]
        # Option 2: b then a, then back to start
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, start_node]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _current_tour.append(next_node)
        return next_node
    
    n_cands = len(unvisited_nodes)
    start_node = _current_tour[0]
    
    # Immediate costs from current_node to all candidates
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # Hybrid Strategy:
    # If few nodes remain, use exact NN completion (high quality, low cost).
    # If many nodes remain, use vectorized 2-step lookahead (low quality, very low cost).
    
    if n_cands <= 10:
        # Use NN completion for high accuracy
        est_costs = np.zeros(n_cands)
        for idx in range(n_cands):
            candidate = unvisited_nodes[idx]
            mask = np.ones(n_cands, dtype=bool)
            mask[idx] = False
            remaining = unvisited_nodes[mask]
            
            if len(remaining) == 0:
                est_costs[idx] = distance_matrix[candidate, start_node]
            else:
                # Fast NN completion
                rem_list = list(remaining)
                curr = candidate
                rem_cost = 0.0
                while rem_list:
                    rem_arr = np.array(rem_list)
                    dists = distance_matrix[curr, rem_arr]
                    min_idx = np.argmin(dists)
                    rem_cost += dists[min_idx]
                    curr = rem_list[min_idx]
                    rem_list.pop(min_idx)
                rem_cost += distance_matrix[curr, start_node]
                est_costs[idx] = rem_cost
        
        total_costs = imm_costs + est_costs
        best_idx = np.argmin(total_costs)
        best_candidate = int(unvisited_nodes[best_idx])
        _current_tour.append(best_candidate)
        return best_candidate
        
    else:
        # Use Vectorized 2-step Lookahead
        # Compute sub-matrix of distances between unvisited nodes
        sub_matrix = distance_matrix[unvisited_nodes[:, np.newaxis], unvisited_nodes[np.newaxis, :]]
        np.fill_diagonal(sub_matrix, np.inf)
        
        # For each candidate u, find the minimum distance to any other unvisited node v
        # This represents the cost of the "next" step after u
        min_next_costs = np.min(sub_matrix, axis=1)
        
        # We also want to penalize nodes that are far from the start if they are likely to be late in the tour?
        # A simple 2-step score: d(current, u) + lambda * d(u, best_next)
        # Let's use lambda = 0.5
        lambda_weight = 0.5
        scores = imm_costs + lambda_weight * min_next_costs
        
        best_idx = np.argmin(scores)
        best_candidate = int(unvisited_nodes[best_idx])
        _current_tour.append(best_candidate)
        return best_candidate

# EVOLVE-BLOCK-END
