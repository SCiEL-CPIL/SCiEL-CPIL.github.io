"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import hashlib

_DEADLINE = None
_DEADLINE_BUFFER = 10.0  # seconds safety margin before 150s limit
_tour_history = []
_prev_instance_hash = None

def _get_instance_hash(distance_matrix):
    # Use a small subset of the matrix to create a fast hash
    # Reshape to 1D and take every 100th element to reduce size
    flat = distance_matrix.flatten()
    sample = flat[::100]
    return hashlib.md5(sample.tobytes()).hexdigest()

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE, _tour_history, _prev_instance_hash
    
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0 - _DEADLINE_BUFFER
    
    # Check for new instance
    current_hash = _get_instance_hash(distance_matrix)
    if current_hash != _prev_instance_hash:
        _tour_history = []
        _prev_instance_hash = current_hash
        _tour_history.append(current_node) # Start of tour

    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        _tour_history.append(int(unvisited_nodes[0]))
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        d0 = distance_matrix[current_node, unvisited_nodes[0]]
        d1 = distance_matrix[current_node, unvisited_nodes[1]]
        d_dest_0 = distance_matrix[unvisited_nodes[0], destination_node]
        d_dest_1 = distance_matrix[unvisited_nodes[1], destination_node]
        d_cross = distance_matrix[unvisited_nodes[0], unvisited_nodes[1]]
        
        cost_pick_0 = d0 + d_cross + d_dest_1
        cost_pick_1 = d1 + d_cross + d_dest_0
        
        if cost_pick_0 <= cost_pick_1:
            next_node = int(unvisited_nodes[0])
        else:
            next_node = int(unvisited_nodes[1])
            
        _tour_history.append(next_node)
        return next_node

    # --- Core Strategy: Geometric Continuity (Angle-based) + Local Distance ---
    
    # 1. Compute distances from current node to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Compute distances from all unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Compute Angle Score
    # We need the previous node to calculate the angle.
    # _tour_history has at least [start, current_node] if n_unvisited < total.
    # If _tour_history length is 1, we are at the start, no angle.
    if len(_tour_history) >= 2:
        prev_node = _tour_history[-2]
        d_prev_current = distance_matrix[prev_node, current_node]
        
        # Distances from prev_node to all unvisited
        d_prev_unvisited = distance_matrix[prev_node, unvisited_nodes]
        
        # Law of Cosines: a^2 = b^2 + c^2 - 2bc cos(A)
        # a = d_prev_unvisited, b = d_prev_current, c = d_current
        # cos(A) = (b^2 + c^2 - a^2) / (2bc)
        
        # Avoid division by zero
        denom = 2 * d_prev_current * d_current
        # Handle case where d_current or d_prev_current is 0 (shouldn't happen for distinct nodes)
        mask = denom > 0
        cos_angles = np.zeros(n_unvisited)
        cos_angles[mask] = (d_prev_current**2 + d_current[mask]**2 - d_prev_unvisited[mask]**2) / denom[mask]
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        
        # Angle in radians
        angles = np.arccos(cos_angles)
        
        # We want to MINIMIZE the turn angle (prefer straight paths)
        # Normalize angle to 0..1
        max_angle = np.max(angles)
        if max_angle > 0:
            angle_score = angles / max_angle
        else:
            angle_score = np.zeros(n_unvisited)
    else:
        # First step, no angle preference
        angle_score = np.zeros(n_unvisited)
        d_prev_unvisited = None

    # 4. Combine Scores
    # Normalize d_current
    min_d_curr = np.min(d_current)
    max_d_curr = np.max(d_current)
    if max_d_curr > min_d_curr:
        norm_d_curr = (d_current - min_d_curr) / (max_d_curr - min_d_curr)
    else:
        norm_d_curr = np.zeros(n_unvisited)
        
    # Normalize d_dest
    min_d_dest = np.min(d_dest)
    max_d_dest = np.max(d_dest)
    if max_d_dest > min_d_dest:
        norm_d_dest = (d_dest - min_d_dest) / (max_d_dest - min_d_dest)
    else:
        norm_d_dest = np.zeros(n_unvisited)

    # Weights:
    # Early on: prioritize angle (smoothness) and local distance.
    # Later: prioritize closing the loop (dest).
    
    if n_unvisited > 20:
        w_local = 0.4
        w_angle = 0.4
        w_dest = 0.2
    elif n_unvisited > 10:
        w_local = 0.5
        w_angle = 0.3
        w_dest = 0.2
    else:
        w_local = 0.6
        w_angle = 0.1
        w_dest = 0.3

    # Score = w_local * norm_d_curr + w_angle * angle_score + w_dest * norm_d_dest
    scores = w_local * norm_d_curr + w_angle * angle_score + w_dest * norm_d_dest
    
    # To avoid getting stuck in a "straight line" trap that leads to a dead end,
    # we restrict candidates to the top K nearest neighbors.
    top_k = min(10, n_unvisited)
    
    if top_k < n_unvisited:
        k_smallest_indices = np.argpartition(d_current, top_k)[:top_k]
    else:
        k_smallest_indices = np.arange(n_unvisited)
        
    cand_scores = scores[k_smallest_indices]
    
    best_idx_in_candidates = np.argmin(cand_scores)
    best_orig_idx = k_smallest_indices[best_idx_in_candidates]
    
    next_node = int(unvisited_nodes[best_orig_idx])
    
    _tour_history.append(next_node)
    
    return next_node
# EVOLVE-BLOCK-END
