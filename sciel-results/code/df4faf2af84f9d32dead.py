import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    def elapsed():
        return time.time() - t0

    def compute_areas_vec(pts):
        """
        Compute absolute doubled areas for all triples using vectorization.
        Returns a dictionary of area -> count.
        """
        n = len(pts)
        if n < 3:
            return {}
        
        # Convert to numpy array for fast operations
        P = np.array(pts, dtype=np.int64)
        
        area_counts = defaultdict(int)
        
        for i in range(n - 2):
            pi = P[i]
            sub = P[i+1:]
            m = len(sub)
            if m < 2:
                continue
            
            dx = sub[:, 0] - pi[0]
            dy = sub[:, 1] - pi[1]
            
            for u in range(m - 1):
                du = dx[u]
                dv = dy[u]
                
                dv_vec = dy[u+1:]
                du_vec = dx[u+1:]
                
                dets = np.abs(du * dv_vec - dv * du_vec)
                
                non_zero = dets[dets > 0]
                if len(non_zero) > 0:
                    vals, counts = np.unique(non_zero, return_counts=True)
                    for val, cnt in zip(vals, counts):
                        area_counts[int(val)] += int(cnt)
                        
        return area_counts

    def score_of(pts):
        areas = compute_areas_vec(pts)
        if not areas:
            return 0
        return max(areas.values())

    def is_valid_point(p, existing_points, ignore_idx=None):
        x, y = p
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
        for j, ep in enumerate(existing_points):
            if j == ignore_idx:
                continue
            if ep == p:
                return False
        return True

    # 1. Generate diverse initial configurations
    initial_candidates = []
    
    # Standard Rectangular Grids
    initial_candidates.append([(x, y) for x in range(6) for y in range(5)])
    initial_candidates.append([(x, y) for x in range(10) for y in range(3)])
    initial_candidates.append([(x, y) for x in range(15) for y in range(2)])
    initial_candidates.append([(x, y) for x in range(5) for y in range(6)])
    
    # Sheared Grids (Affine transformations)
    # Grid: x in [0, 5], y in [0, 4]
    # Shear: x' = x + y, y' = y
    initial_candidates.append([(x + y, y) for x in range(6) for y in range(5)])
    # Shear: x' = x + 2*y, y' = y
    initial_candidates.append([(x + 2*y, y) for x in range(6) for y in range(5)])
    
    # Centered grids to allow for symmetric moves
    # 6x5 grid centered at (0,0)
    # x in [-2.5, 2.5] -> not integer. 
    # Let's use x in [-3, 2], y in [-2, 3]
    initial_candidates.append([(x, y) for x in range(-3, 3) for y in range(-2, 4)])
    
    # Filter valid candidates
    valid_initials = []
    for cand in initial_candidates:
        if len(cand) != 30:
            continue
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in cand):
            # Check distinctness
            if len(set(cand)) == 30:
                valid_initials.append(cand)
    
    if not valid_initials:
        # Fallback
        valid_initials = [[(x, y) for x in range(6) for y in range(5)]]

    # Evaluate all initial candidates
    best_points = valid_initials[0]
    best_score = score_of(best_points)
    
    for cand in valid_initials[1:]:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = cand

    # 2. Local Search
    current_points = list(best_points)
    current_score = best_score
    
    random.seed(42)
    
    # We will run multiple "phases" or just a continuous loop with stagnation detection
    stagnation_counter = 0
    max_stagnation = 500  # If no improvement for 500 moves, try a bigger jump or restart
    
    while elapsed() < DEADLINE:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Determine step size
        # Base step size 2, occasionally larger
        if random.random() < 0.05:
            step_size = random.randint(5, 15)
        else:
            step_size = random.randint(1, 3)
            
        new_x = old_x + random.randint(-step_size, step_size)
        new_y = old_y + random.randint(-step_size, step_size)
        
        new_point = (new_x, new_y)
        
        if not is_valid_point(new_point, current_points, idx):
            continue
            
        # Try the move
        current_points[idx] = new_point
        new_score = score_of(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
            stagnation_counter = 0
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            stagnation_counter += 1
            
        # If stuck, perform a "kick" - move multiple points randomly or reset to best
        if stagnation_counter > max_stagnation:
            # Option 1: Randomly perturb 3-5 points significantly
            num_kicks = random.randint(3, 5)
            for _ in range(num_kicks):
                k_idx = random.randint(0, 29)
                k_x = current_points[k_idx][0]
                k_y = current_points[k_idx][1]
                # Large random jump
                new_kx = k_x + random.randint(-10, 10)
                new_ky = k_y + random.randint(-10, 10)
                if is_valid_point((new_kx, new_ky), current_points, k_idx):
                    current_points[k_idx] = (new_kx, new_ky)
            
            # Re-evaluate current state after kicks
            kick_score = score_of(current_points)
            if kick_score < current_score:
                # Revert to the state before kicks? 
                # We didn't save it. Let's just accept the kick if it's not terrible, 
                # or better, revert to best_points if we got stuck.
                # Actually, simpler: just reset current to best and restart search
                current_points = list(best_points)
                current_score = best_score
                stagnation_counter = 0
            else:
                current_score = kick_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = list(current_points)
                stagnation_counter = 0

    return best_points
