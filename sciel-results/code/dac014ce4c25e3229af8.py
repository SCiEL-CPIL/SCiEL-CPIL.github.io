"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary
    dist_x = 0.5 - np.abs(pos[:, 0])
    dist_y = 0.5 - np.abs(pos[:, 1])
    dist_boundary = np.minimum(dist_x, dist_y)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = float(np.min(dist_sq))
    min_pair = math.sqrt(min_pair_sq)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_hex_lattice(spacing, margin=0.0):
    """Generate a fine hexagonal lattice of candidate points in [-0.5, 0.5]^2."""
    points = []
    # spacing is the distance between adjacent lattice points
    dy = spacing * math.sqrt(3) / 2.0
    
    # Determine range of rows and cols
    max_coord = 0.5 - margin
    n_rows = int(np.ceil(2 * max_coord / dy)) + 2
    n_cols = int(np.ceil(2 * max_coord / spacing)) + 2
    
    for row in range(n_rows):
        y = -max_coord + row * dy
        if y > max_coord:
            break
        x_offset = 0.0 if row % 2 == 0 else spacing / 2.0
        for col in range(n_cols):
            x = -max_coord + col * spacing + x_offset
            if x > max_coord:
                break
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append([x, y])
    
    return np.array(points) if points else np.array([]).reshape(0, 2)

def _greedy_select(candidates, n_select, time_limit=None, start_time=None):
    """
    Greedy best-fit: select n_select points from candidates to maximize
    the minimum distance to already-selected points and boundaries.
    """
    if len(candidates) < n_select:
        return candidates.copy()
    
    if start_time is None:
        start_time = time.time()
    
    selected = []
    selected_set = set()
    
    # Precompute boundary distances for all candidates
    dist_x = 0.5 - np.abs(candidates[:, 0])
    dist_y = 0.5 - np.abs(candidates[:, 1])
    boundary_dist = np.minimum(dist_x, dist_y)
    
    # Track the current "score" for each candidate: min(boundary_dist, min_dist_to_selected)
    scores = boundary_dist.copy()
    
    for _ in range(n_select):
        if time_limit is not None and time.time() > start_time + time_limit:
            break
        
        # Find the candidate with the highest score
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        selected.append(candidates[best_idx])
        selected_set.add(best_idx)
        
        # Update scores: for each remaining candidate, the new min distance
        # to any selected point might be smaller than before
        if len(selected) > 1:
            # Compute distances from all candidates to the newly selected point
            new_point = candidates[best_idx]
            diff = candidates - new_point
            dist_sq = np.sum(diff**2, axis=1)
            dist = np.sqrt(dist_sq)
            dist[best_idx] = np.inf  # Exclude self
            
            # Update scores: score_i = min(old_score_i, dist_i_to_new)
            scores = np.minimum(scores, dist)
    
    return np.array(selected) if selected else np.array([]).reshape(0, 2)

def _local_refinement(pos, start_time, max_time=10.0):
    """
    Simple local refinement: try small perturbations of individual points
    to improve the minimum radius.
    """
    if len(pos) == 0:
        return pos
    
    current_r = _compute_radius(pos)
    pos = pos.copy()
    rng = np.random.RandomState(123)
    
    step_size = 0.002
    n_improvements = 0
    
    while time.time() < start_time + max_time:
        improved = False
        
        # Try perturbing each point
        for i in range(len(pos)):
            if time.time() > start_time + max_time:
                break
            
            # Try several random directions
            for _ in range(5):
                delta = rng.randn(2) * step_size
                new_pos = pos.copy()
                new_pos[i] += delta
                new_pos = np.clip(new_pos, -0.4999, 0.4999)
                
                new_r = _compute_radius(new_pos)
                
                if new_r > current_r + 1e-12:
                    pos = new_pos
                    current_r = new_r
                    improved = True
                    n_improvements += 1
                    break
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-7:
                break
    
    return pos, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_pos = None
    best_r = 0.0
    
    # Try multiple lattice spacings and greedy selections
    # The optimal radius is ~0.0366, so spacing should be around 2*r ~ 0.073
    # But we want a fine grid, so try smaller spacings for better resolution
    
    spacings = [0.05, 0.04, 0.03, 0.025, 0.02]
    
    for spacing in spacings:
        if time.time() > deadline - 20:
            break
        
        try:
            candidates = _generate_hex_lattice(spacing)
            if len(candidates) == 0:
                continue
            
            # Greedy selection
            selected = _greedy_select(candidates, N, time_limit=30, start_time=start_time)
            
            if len(selected) < N:
                continue
            
            r = _compute_radius(selected)
            
            if r > best_r:
                best_r = r
                best_pos = selected.copy()
                
        except Exception:
            continue
    
    # If we have a valid solution, try local refinement
    if best_pos is not None and time.time() < deadline - 5:
        refined_pos, refined_r = _local_refinement(best_pos, start_time, max_time=min(20, deadline - time.time() - 2))
        if refined_r > best_r:
            best_r = refined_r
            best_pos = refined_pos
    
    # Fallback: if greedy didn't work well, try a simple grid
    if best_pos is None or best_r < 0.01:
        # Simple uniform grid
        side = int(np.ceil(math.sqrt(N)))
        pos = np.zeros((N, 2))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                pos[count, 0] = -0.45 + 0.9 * i / (side - 1) if side > 1 else 0
                pos[count, 1] = -0.45 + 0.9 * j / (side - 1) if side > 1 else 0
                count += 1
            if count >= N:
                break
        
        if count == N:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos
    
    # Final safety check
    if best_pos is None:
        # Absolute fallback: grid
        side = int(np.ceil(math.sqrt(N)))
        best_pos = np.zeros((N, 2))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                best_pos[count, 0] = -0.49 + 0.98 * i / (side - 1) if side > 1 else 0
                best_pos[count, 1] = -0.49 + 0.98 * j / (side - 1) if side > 1 else 0
                count += 1
            if count >= N:
                break
    
    return best_pos
# EVOLVE-BLOCK-END
