import time
import random
import itertools
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    total_deadline = start_time + 85.0

    # 1. Fast fully-vectorized scoring
    def score(pts):
        n = len(pts)
        if n < 3:
            return 0
        arr = np.array(pts, dtype=np.int64)
        # Generate all index triples (i, j, k) with i < j < k
        i_idx, j_idx, k_idx = np.meshgrid(
            np.arange(n-2), np.arange(1, n-1), np.arange(2, n), indexing='ij'
        )
        i_idx = i_idx[i_idx < j_idx]
        j_idx = j_idx[i_idx < j_idx]
        k_idx = k_idx[i_idx < j_idx]
        
        # Compute doubled areas for all triples
        xi = arr[i_idx, 0]
        yi = arr[i_idx, 1]
        xj = arr[j_idx, 0]
        yj = arr[j_idx, 1]
        xk = arr[k_idx, 0]
        yk = arr[k_idx, 1]
        
        doubled_areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Filter out zero areas
        nonzero = doubled_areas[doubled_areas > 0]
        if len(nonzero) == 0:
            return 0
        
        # Count occurrences of each doubled area value
        vals, cnts = np.unique(nonzero, return_counts=True)
        return int(cnts.max())

    # 2. Candidate Generation
    candidates = []
    
    # Grid patterns with various spacings
    for w in [2, 3, 5, 6, 10, 15]:
        h = 30 // w
        if h < 2 or w * h != 30:
            continue
        for dx in [1, 2, 3, 4, 5]:
            for dy in [1, 2, 3, 4, 5]:
                if dx * (w - 1) > 60 or dy * (h - 1) > 60:
                    continue
                for ox in range(-10, 11, 5):
                    for oy in range(-10, 11, 5):
                        pts = []
                        valid = True
                        for i in range(w):
                            for j in range(h):
                                x = ox + i * dx
                                y = oy + j * dy
                                if -30 <= x <= 30 and -30 <= y <= 30:
                                    pts.append((x, y))
                                else:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(pts) == 30:
                            candidates.append(pts)
    
    # Diagonal patterns
    for slope in [1, 2, 3, 4]:
        for intercept in range(-30, 31, 3):
            pts = []
            for x in range(-30, 31):
                y = slope * x + intercept
                if -30 <= y <= 30:
                    pts.append((x, y))
                if len(pts) >= 30:
                    break
            if len(pts) == 30:
                candidates.append(pts)
    
    # Lattice-like patterns with different orientations
    for a in [1, 2, 3]:
        for b in [1, 2, 3]:
            for c in [1, 2, 3]:
                for ox in range(-15, 16, 5):
                    for oy in range(-15, 16, 5):
                        pts = []
                        seen = set()
                        for i in range(30):
                            for j in range(30):
                                x = ox + a * i + b * j
                                y = oy + c * i + (a + c) * j
                                if -30 <= x <= 30 and -30 <= y <= 30:
                                    p = (x, y)
                                    if p not in seen:
                                        seen.add(p)
                                        pts.append(p)
                                        if len(pts) == 30:
                                            break
                            if len(pts) == 30:
                                break
                        if len(pts) == 30:
                            candidates.append(pts)
    
    # Random structured patterns
    random.seed(42)
    for trial in range(30):
        base_w = random.choice([3, 5, 6, 10])
        base_h = 30 // base_w
        dx = random.choice([1, 2, 3])
        dy = random.choice([1, 2, 3])
        ox = random.randint(-10, 10)
        oy = random.randint(-10, 10)
        
        pts = []
        seen = set()
        for i in range(base_w):
            for j in range(base_h):
                x = ox + i * dx + random.randint(-1, 1)
                y = oy + j * dy + random.randint(-1, 1)
                if -30 <= x <= 30 and -30 <= y <= 30:
                    p = (x, y)
                    if p not in seen:
                        seen.add(p)
                        pts.append(p)
        
        # Deduplicate and fill remaining with random points if needed
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            p = (x, y)
            if p not in seen:
                seen.add(p)
                pts.append(p)
        
        if len(pts) == 30:
            candidates.append(pts)
    
    # Pure random candidates
    for _ in range(15):
        random_pts = set()
        while len(random_pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            random_pts.add((x, y))
        candidates.append(list(random_pts))
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # 3. Initial Evaluation
    scored_candidates = []
    for c in unique_candidates:
        if time.time() > total_deadline - 20:
            break
        s = score(c)
        scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_overall_score = scored_candidates[0][0] if scored_candidates else 0
    best_overall_pts = scored_candidates[0][1] if scored_candidates else []
    
    # 4. Local Search with simulated annealing
    def local_search(init_pts, duration):
        current_pts = [p for p in init_pts]
        current_score = score(current_pts)
        current_set = set(current_pts)
        end_time = time.time() + duration
        
        # Temperature for simulated annealing
        T = 3.0
        T_min = 0.1
        cooling_rate = 0.999
        
        while time.time() < end_time:
            i = random.randint(0, 29)
            old_p = current_pts[i]
            
            # Move strategy: 50% local, 30% medium, 20% global
            r = random.random()
            if r < 0.5:
                # Small local move
                nx = old_p[0] + random.randint(-3, 3)
                ny = old_p[1] + random.randint(-3, 3)
            elif r < 0.8:
                # Medium move
                nx = old_p[0] + random.randint(-8, 8)
                ny = old_p[1] + random.randint(-8, 8)
            else:
                # Global move
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
            
            new_p = (nx, ny)
            if -30 <= nx <= 30 and -30 <= ny <= 30:
                if new_p != old_p and new_p not in current_set:
                    # Apply move
                    current_pts[i] = new_p
                    current_set.remove(old_p)
                    current_set.add(new_p)
                    
                    new_score = score(current_pts)
                    
                    delta = new_score - current_score
                    
                    # Simulated annealing acceptance
                    if delta > 0:
                        current_score = new_score
                    else:
                        # Accept with probability exp(delta/T)
                        if random.random() < np.exp(delta / T):
                            current_score = new_score
                        else:
                            # Revert
                            current_pts[i] = old_p
                            current_set.remove(new_p)
                            current_set.add(old_p)
                    
                    # Cool down
                    T = max(T_min, T * cooling_rate)
                    
        return current_score, current_pts
    
    # Run multiple searches with different time allocations
    search_configs = [
        (0, 25),  # Best candidate, 25s
        (1, 15),  # 2nd best, 15s
        (2, 10),  # 3rd best, 10s
        (3, 5),   # 4th best, 5s
    ]
    
    for idx, t in search_configs:
        if time.time() > total_deadline - 3:
            break
        if idx >= len(scored_candidates):
            break
        
        s, p = scored_candidates[idx]
        rem_time = total_deadline - time.time() - 2
        if rem_time <= 0:
            break
        
        t = min(t, rem_time)
        new_s, new_p = local_search(p, t)
        
        if new_s > best_overall_score:
            best_overall_score = new_s
            best_overall_pts = new_p
    
    return best_overall_pts
