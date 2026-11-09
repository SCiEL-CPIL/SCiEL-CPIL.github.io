"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # Structural Change: 1-step Lookahead Heuristic
    # Instead of just picking the nearest neighbor to 'current_node',
    # we evaluate the cost of moving to candidate 'c' and then to the 
    # best subsequent node.
    #
    # For each candidate c:
    # 1. Cost to get to c from current: d(current, c)
    # 2. Cost to leave c:
    #    - If c is the last unvisited node, we must go to destination_node.
    #      Cost = d(c, destination_node)
    #    - Otherwise, we go to the nearest unvisited node v (v != c).
    #      Cost = min(d(c, v) for v in unvisited_nodes if v != c)
    #
    # Total estimated cost for choosing c now: d(current, c) + cost_to_leave(c)
    
    best_score = float('inf')
    best_node = candidates[0]
    
    # Pre-compute distances from current to all candidates
    dist_from_current = distance_matrix[current_node, candidates]
    
    # Pre-compute distance matrix subset for candidates to speed up min calculation
    # dist_sub[i, j] = distance_matrix[candidates[i], candidates[j]]
    # We can compute this on the fly or use advanced indexing.
    # Given N <= 50, N^2 = 2500 operations per step. 50 steps = 125,000 ops. Very fast.
    
    for i, c in enumerate(candidates):
        # Cost to reach c
        cost_in = dist_from_current[i]
        
        if n == 2:
            # If only 2 nodes left, the other one is the next
            # The one after that is destination
            # Actually, if n==2, we pick c, then the other node, then destination.
            # But we are just picking c now.
            # The "next" after c will be the other candidate.
            # So cost_to_leave is d(c, other_candidate).
            other_idx = 1 - i
            other_node = candidates[other_idx]
            cost_out = distance_matrix[c, other_node]
        else:
            # Find the nearest neighbor to c among the remaining unvisited nodes
            # Exclude c itself
            # We can look at all candidates except i
            # distances from c to all candidates
            dists_to_others = distance_matrix[c, candidates]
            # Set distance to self to inf to exclude it
            dists_to_others[i] = float('inf')
            # The minimum distance to any other unvisited node
            cost_out = np.min(dists_to_others)
            
            # Special case: If c is the last node we will visit (which we don't know yet,
            # but if we pick it, the tour will close to destination).
            # The lookahead assumes we go to the nearest neighbor.
            # If the nearest neighbor is far, but destination is closer, 
            # this heuristic might be slightly off if c turns out to be the last node.
            # However, usually the last node is determined by the flow.
            # A more robust check: 
            # If we pick c, and there are k other unvisited nodes, we will visit them.
            # The cost to leave c is towards the next node in the tour.
            # The "nearest neighbor" approximation is standard for this.
            
            # Refinement: 
            # If we pick c, and it happens to be the last one (which we don't know), 
            # the cost should be d(c, destination).
            # But we don't know if it's the last one.
            # However, if d(c, destination) is significantly smaller than min(d(c, others)),
            # and n is small, we might want to consider it.
            # But typically, we visit nodes in a cluster.
            
            # Let's stick to the nearest unvisited neighbor as the primary predictor.
            pass
        
        total_score = cost_in + cost_out
        
        if total_score < best_score:
            best_score = total_score
            best_node = c
            
    return int(best_node)
# EVOLVE-BLOCK-END
