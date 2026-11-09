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
        Uses vectorized cross-product computation for speed.
        """
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)

        # Build all pairwise difference vectors (i, j) with i < j
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0].astype(np.int64)
        dy = d[:, 1].astype(np.int64)
        M = len(dx)

        # Compute all pairwise cross products at once
        cross = np.outer(dx, dy) - np.outer(dy, dx)
        
        # Take upper triangle (u < v) to avoid double counting
        tri = np.triu(cross, k=1)
        nz = tri[tri != 0]
        if len(nz) == 0:
            return 0
        nz_abs = np.abs(nz)
        vals, cnts = np.unique(nz_abs, return_counts=True)
        return int(cnts.max())

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
    if len(cand8) < 30:
        while len(cand8) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used8:
                cand8.append((x, y))
                used8.add((x, y))

    # Candidate 9: 2 lines with different lengths
    cand9 = [(x, 0) for x in range(12)] + [(x, 1) for x in range(12)] + [(x, 2) for x in range(6)]
    
    # Candidate 10: 3 lines with offset patterns
    cand10 = [(x, 0) for x in range(10)] + [(x+1, 1) for x in range(10)] + [(x+2, 2) for x in range(10)]
    
    # Candidate 11: Dense cluster + spread
    cand11 = [(x, y) for x in range(4) for y in range(4)]
    for i in range(14):
        cand11.append((8 + i, 0))
    
    # Candidate 12: Two parallel lines with spacing 2
    cand12 = [(x, 0) for x in range(15)] + [(x, 2) for x in range(15)]
    
    # Candidate 13: 4 lines, 7-8 points each
    cand13 = [(x, 0) for x in range(8)] + [(x, 1) for x in range(8)] + \
             [(x, 2) for x in range(7)] + [(x, 3) for x in range(7)]
    
    # Candidate 14: 6 lines, 5 points each
    cand14 = [(x, 0) for x in range(5)] + [(x, 1) for x in range(5)] + \
             [(x, 2) for x in range(5)] + [(x, 3) for x in range(5)] + \
             [(x, 4) for x in range(5)] + [(x, 5) for x in range(5)]
    
    candidates = [cand1, cand2, cand3, cand4, cand5, cand6, cand7, cand8,
                  cand9, cand10, cand11, cand12, cand13, cand14]
    
    best_points = None
    best_score = 0
    
    for cand in candidates:
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            
    # 2. Local Search with improved strategy
    current_points = list(best_points)
    current_score = best_score
    
    random.seed(42)
    step_size = 3
    
    last_improvement_iter = 0
    iter_count = 0
    no_improve_count = 0
    
    while elapsed() < DEADLINE:
        iter_count += 1
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        r = random.random()
        if r < 0.02:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.12:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)
            
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
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
            
        current_points[idx] = new_pt
        
        new_score = score_of(current_points)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
                last_improvement_iter = iter_count
                if step_size > 1 and random.random() < 0.3:
                    step_size = max(1, step_size - 1)
        else:
            current_points[idx] = (old_x, old_y)
            no_improve_count += 1
            
        if no_improve_count > 2000:
            current_points = list(best_points)
            current_score = best_score
            no_improve_count = 0
            last_improvement_iter = iter_count
            step_size = random.randint(4, 10)
            
        if random.random() < 0.003:
            step_size = random.randint(1, 5)

    return best_points
