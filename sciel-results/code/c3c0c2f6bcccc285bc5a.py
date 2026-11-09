import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit

    def elapsed():
        return time.time() - t0

    def compute_max_area_count_fast(pts):
        """
        Compute the maximum multiplicity of any non-zero doubled area.
        Optimized using pairwise difference vectors and numpy.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]  # shape (M, 2), M = n*(n-1)/2
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)

        # For each base vector a (index u), compute cross products with all later vectors v
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)

        if not area_counts:
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count_fast(pts)

    # 1. Initialize with strong structured candidates
    
    # Candidate 1: Two lines, 15 points each
    cand1 = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    
    # Candidate 2: Grid 6x5
    cand2 = [(x, y) for x in range(6) for y in range(5)]
    
    # Candidate 3: Three lines, 10 points each
    cand3 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    
    # Candidate 4: Random
    random.seed(42)
    cand4 = []
    used = set()
    while len(cand4) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in used:
            used.add((x, y))
            cand4.append((x, y))
            
    # Candidate 5: Four lines with varying spacing
    cand5 = [(x, 0) for x in range(8)] + [(x, 2) for x in range(8)] + \
            [(x, 4) for x in range(7)] + [(x, 6) for x in range(7)]
    
    # Candidate 6: Concentric-like structure
    cand6 = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + \
            [(x, -1) for x in range(5)] + [(x, -2) for x in range(5)]
    
    # Candidate 7: 5 lines, 6 points each
    cand7 = [(x, 0) for x in range(6)] + [(x, 1) for x in range(6)] + \
            [(x, 2) for x in range(6)] + [(x, 3) for x in range(6)] + \
            [(x, 4) for x in range(6)]
    
    # Candidate 8: 2 lines with offset
    cand8 = [(x, 0) for x in range(15)] + [(x+1, 1) for x in range(14)] + [(14, 1)]
    # Ensure distinctness and bounds for cand8
    cand8 = []
    used8 = set()
    for x in range(15):
        pt = (x, 0)
        if pt not in used8:
            cand8.append(pt)
            used8.add(pt)
    for x in range(15):
        pt = (x, 1)
        if pt not in used8:
            cand8.append(pt)
            used8.add(pt)
    # If less than 30, fill randomly
    if len(cand8) < 30:
        while len(cand8) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used8:
                cand8.append((x, y))
                used8.add((x, y))

    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8]
    
    best_points = None
    best_score = 0
    
    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    # 2. Targeted Simulated Annealing
    # Strategy: Instead of maximizing the global max area count directly (which is noisy and hard to guide),
    # we identify the "dominant" area from the initial best and try to grow the clique of that specific area.
    # If we fail to grow it, we might switch to a new dominant area.
    
    current_points = list(best_points)
    current_score = best_score
    
    # Get the dominant area from the initial best
    def get_dominant_area(pts):
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)
        if not area_counts:
            return 1, 0
        max_count = max(area_counts.values())
        # Find an area with max_count
        for a, c in area_counts.items():
            if c == max_count:
                return a, c
        return 1, 0

    target_area, target_count = get_dominant_area(current_points)
    
    # Helper to count how many triangles have the target area
    def count_target_area(pts, target_A):
        n = len(pts)
        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        count = 0
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            # We want |cross| == target_A
            count += np.sum(np.abs(cross) == target_A)
        return count

    current_target_count = count_target_area(current_points, target_area)
    
    random.seed(42)
    temperature = 10.0
    cooling_rate = 0.999
    
    # We will run SA for a fixed number of iterations or until time runs out
    # We focus on moves that increase the count of the target area.
    
    while elapsed() < DEADLINE:
        # Occasionally re-evaluate the global score to see if we've beaten the best
        # and potentially update the target area
        if random.random() < 0.01:
            global_score = score_of(current_points)
            if global_score > best_score:
                best_score = global_score
                best_points = list(current_points)
                # Update target area to the new dominant one
                target_area, target_count = get_dominant_area(current_points)
                current_target_count = count_target_area(current_points, target_area)
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        # We use a smaller step size here to be more precise in "clique growth"
        step_size = random.randint(1, 5)
        new_x = old_x + random.randint(-step_size, step_size)
        new_y = old_y + random.randint(-step_size, step_size)
        
        # Ensure bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check distinctness
        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue
        is_valid = True
        for j in range(30):
            if j == idx:
                continue
            if current_points[j] == new_pt:
                is_valid = False
                break
        
        if not is_valid:
            continue
            
        # Try the move
        current_points[idx] = new_pt
        
        # Evaluate the change in target area count
        new_target_count = count_target_area(current_points, target_area)
        
        delta_target = new_target_count - current_target_count
        
        # Acceptance criterion:
        # If delta_target > 0, always accept (it helps our specific goal)
        # If delta_target == 0, accept with some probability to escape local minima
        # If delta_target < 0, accept with low probability (Boltzmann)
        
        if delta_target > 0:
            accept = True
        elif delta_target == 0:
            accept = random.random() < 0.5
        else:
            # Penalize moves that reduce the target count
            # The penalty scales with the magnitude of the loss
            accept_prob = np.exp(delta_target / temperature)
            accept = random.random() < accept_prob
            
        if accept:
            current_target_count = new_target_count
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 1.0 # Reheat occasionally to escape deep local minima

    # Final check
    final_score = score_of(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = list(current_points)

    return best_points
