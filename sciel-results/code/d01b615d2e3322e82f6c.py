import time
import numpy as np
from collections import defaultdict, Counter
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit
    
    def elapsed():
        return time.time() - t0
    
    def compute_max_area_count(pts):
        """Compute the maximum multiplicity of any non-zero doubled area."""
        n = len(pts)
        if n < 3:
            return 0

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
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count(pts)
    
    def count_target_areas(pts, target_areas):
        """Count how many triangles have area in the target set."""
        n = len(pts)
        if n < 3:
            return 0
        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)
        
        target_set = set(target_areas)
        count = 0
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
            for val in nz_abs:
                if int(val) in target_set:
                    count += 1
        return count
    
    # Generate diverse structured candidates
    candidates = []
    
    # Lines with different spacings and offsets
    for y_off in [0, 1, -1]:
        for x_start in range(-25, 5):
            if x_start + 15 <= 30:
                cand = [(x, y_off) for x in range(x_start, x_start + 15)]
                if len(cand) == 15:
                    candidates.append(cand)
    
    # Two parallel lines
    for d in [1, 2, 3, 4]:
        for x1 in range(-25, 2):
            for x2 in range(x1, x1 + 10):
                if x1 + 10 <= 30 and x2 + 10 <= 30:
                    line1 = [(x, 0) for x in range(x1, x1 + 10)]
                    line2 = [(x, d) for x in range(x2, x2 + 10)]
                    if len(line1) == 10 and len(line2) == 10:
                        candidates.append(line1 + line2)
    
    # Grids
    for w in [4, 5, 6, 7]:
        for h in [4, 5, 6, 7]:
            if w * h >= 25 and w * h <= 35:
                for x0 in range(-25, 2):
                    for y0 in range(-25, 2):
                        if x0 + w - 1 <= 30 and y0 + h - 1 <= 30:
                            cand = [(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)]
                            if 25 <= len(cand) <= 30:
                                if len(cand) < 30:
                                    remaining = 30 - len(cand)
                                    existing = set(cand)
                                    for _ in range(remaining):
                                        found = False
                                        for _ in range(50):
                                            x = random.randint(-30, 30)
                                            y = random.randint(-30, 30)
                                            if (x, y) not in existing:
                                                cand.append((x, y))
                                                existing.add((x, y))
                                                found = True
                                                break
                                    if len(cand) == 30:
                                        candidates.append(cand)
                                else:
                                    candidates.append(cand)
    
    # Three parallel lines
    for d1 in [1, 2, 3]:
        for d2 in [1, 2, 3]:
            if d2 > d1:
                for x in range(-25, 5):
                    if x + 8 <= 30:
                        l1 = [(x + i, 0) for i in range(10)]
                        l2 = [(x + i, d1) for i in range(10)]
                        l3 = [(x + i, d2) for i in range(10)]
                        if all(len(l) == 10 for l in [l1, l2, l3]):
                            candidates.append(l1 + l2 + l3)
    
    # Random samples
    for seed in range(20):
        random.seed(seed)
        cand = []
        used = set()
        while len(cand) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y))
                cand.append((x, y))
        candidates.append(cand)
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for cand in candidates:
        if len(cand) != 30:
            continue
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
    
    # Evaluate all candidates
    best_points = None
    best_score = 0
    best_target_areas = set()
    
    for cand in unique_candidates:
        if elapsed() > DEADLINE:
            break
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            # Find which area values contribute most
            P = np.array(cand, dtype=np.int64)
            i_idx, j_idx = np.triu_indices(30, k=1)
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
            # Keep top 5 area values as targets
            top_areas = sorted(area_counts.items(), key=lambda x: -x[1])[:5]
            best_target_areas = set(a for a, c in top_areas if c > 0)
    
    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score = score_of(best_points)
        # Compute target areas for fallback
        P = np.array(best_points, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(30, k=1)
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
        top_areas = sorted(area_counts.items(), key=lambda x: -x[1])[:5]
        best_target_areas = set(a for a, c in top_areas if c > 0)
    
    # Local search refinement with target-area guidance
    current_points = list(best_points)
    current_score = best_score
    
    random.seed(123)
    step_size = 3
    no_improve_count = 0
    last_improve_iter = 0
    iter_count = 0
    
    while elapsed() < DEADLINE:
        iter_count += 1
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        r = random.random()
        if r < 0.03:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif r < 0.15:
            new_x = old_x + random.randint(-6, 6)
            new_y = old_y + random.randint(-6, 6)
        else:
            new_x = old_x + random.randint(-step_size, step_size)
            new_y = old_y + random.randint(-step_size, step_size)
        
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        new_pt = (new_x, new_y)
        if new_pt == (old_x, old_y):
            continue
            
        # Check collision
        is_valid = True
        for j in range(30):
            if j != idx and current_points[j] == new_pt:
                is_valid = False
                break
        
        if not is_valid:
            continue
        
        current_points[idx] = new_pt
        new_score = score_of(current_points)
        
        # Also check target area count as a secondary signal
        target_count = count_target_areas(current_points, best_target_areas)
        
        if new_score > current_score or (new_score == current_score and target_count > 0):
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                last_improve_iter = iter_count
                # Update target areas
                P = np.array(current_points, dtype=np.int64)
                i_idx, j_idx = np.triu_indices(30, k=1)
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
                top_areas = sorted(area_counts.items(), key=lambda x: -x[1])[:5]
                best_target_areas = set(a for a, c in top_areas if c > 0)
                if step_size > 1 and random.random() < 0.2:
                    step_size = max(1, step_size - 1)
        else:
            current_points[idx] = (old_x, old_y)
            no_improve_count += 1
        
        if no_improve_count > 1500:
            current_points = list(best_points)
            current_score = best_score
            no_improve_count = 0
            step_size = random.randint(4, 8)
        
        if random.random() < 0.005:
            step_size = random.randint(1, 6)
    
    return best_points
