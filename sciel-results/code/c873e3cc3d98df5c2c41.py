import time
import random
from collections import Counter
import numpy as np

def _max_multiplicity(pts):
    n = len(pts)
    cnt = Counter()
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx1 = xj - xi
            dy1 = yj - yi
            for k in range(j + 1, n):
                xk, yk = pts[k]
                d = abs(dx1 * (yk - yi) - dy1 * (xk - xi))
                if d:
                    cnt[d] += 1
    return max(cnt.values()) if cnt else 0

def _valid(pts):
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    return all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts)

def _two_lines(n1, n2, y1, y2, x0, spacing=1):
    return [(x0 + spacing * t, y1) for t in range(n1)] + \
           [(x0 + spacing * t, y2) for t in range(n2)]

def _compute_all_dets(pts):
    """Vectorized computation of all C(30,3) doubled areas."""
    n = len(pts)
    arr = np.array(pts, dtype=np.int32)
    # For each pair (i,j), compute cross product with all k > j
    # More efficient: compute all determinants
    # det(i,j,k) = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
    # = xi*yj - xi*yk - xj*yi + xj*yk + xk*yi - xk*yj
    # Let's use a different approach
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Only keep i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    ii = i_idx[mask]
    jj = j_idx[mask]
    kk = k_idx[mask]
    xi = arr[ii, 0].astype(np.int64)
    yi = arr[ii, 1].astype(np.int64)
    xj = arr[jj, 0].astype(np.int64)
    yj = arr[jj, 1].astype(np.int64)
    xk = arr[kk, 0].astype(np.int64)
    yk = arr[kk, 1].astype(np.int64)
    dets = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    return dets

def _fast_score(pts):
    """Fast multiplicity computation using numpy."""
    dets = _compute_all_dets(pts)
    dets = dets[dets > 0]
    if len(dets) == 0:
        return 0
    # Count occurrences of each value
    vals, counts = np.unique(dets, return_counts=True)
    return int(counts.max())

def _delta_score(pts, idx, new_pt):
    """Compute change in multiplicity when moving point idx to new_pt.
    Returns (new_max_mult, old_max_mult) for comparison."""
    # This is expensive to do exactly, so we'll use full recomputation
    # but with numpy it's fast enough
    old_pts = list(pts)
    old_pts[idx] = new_pt
    return _fast_score(old_pts)

def _local_search(pts, deadline, rng):
    """Local search with fast numpy evaluation."""
    cur = list(pts)
    cur_val = _fast_score(cur)
    occupied = set(cur)
    stall = 0
    max_stall = 200
    best_pts = list(cur)
    best_val = cur_val
    
    while time.time() < deadline - 3.0 and stall < max_stall:
        idx = rng.randrange(30)
        ox, oy = cur[idx]
        # Try several candidate positions
        found_improvement = False
        for _ in range(5):
            nx = rng.randint(-30, 30)
            ny = rng.randint(-30, 30)
            if (nx, ny) in occupied:
                continue
            cur[idx] = (nx, ny)
            occupied.remove((ox, oy))
            occupied.add((nx, ny))
            v = _fast_score(cur)
            if v > cur_val:
                cur_val = v
                best_val = v
                best_pts = list(cur)
                stall = 0
                found_improvement = True
                break
            else:
                # revert
                cur[idx] = (ox, oy)
                occupied.remove((nx, ny))
                occupied.add((ox, oy))
        if not found_improvement:
            stall += 1
    
    return best_pts, best_val

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(12345)

    candidates = []
    candidates.append(_two_lines(15, 15, 0, 1, -7, 1))
    candidates.append([(0, y) for y in range(-7, 8)] + [(1, y) for y in range(-7, 8)])
    for n1 in range(10, 21):
        candidates.append(_two_lines(n1, 30 - n1, 0, 1, -7, 1))
    candidates.append(_two_lines(15, 15, 0, 1, -28, 2))
    candidates.append([(x, y) for x in range(6) for y in range(5)])
    # Additional structured candidates
    candidates.append(_two_lines(15, 15, 0, 1, -7, 2))
    candidates.append([(x, y) for x in range(5) for y in range(6)])
    candidates.append([(x, y) for x in range(10) for y in range(3)])
    candidates.append([(x, y) for x in range(3) for y in range(10)])

    best_pts = None
    best_val = -1
    for cand in candidates:
        if not _valid(cand):
            continue
        v = _fast_score(cand)
        if v > best_val:
            best_val = v
            best_pts = list(cand)
        if time.time() > deadline - 10.0:
            break

    if best_pts is None:
        best_pts = [(x, y) for x in range(6) for y in range(5)]
        best_val = _fast_score(best_pts)

    # Local search phase 1
    if time.time() < deadline - 10.0:
        best_pts, best_val = _local_search(best_pts, deadline, rng)

    # Simulated annealing phase
    if time.time() < deadline - 5.0:
        cur = list(best_pts)
        cur_val = best_val
        occupied = set(cur)
        temperature = 5.0
        cool_rate = 0.995
        min_temp = 0.5
        sa_deadline = deadline - 5.0
        
        while time.time() < sa_deadline and temperature > min_temp:
            idx = rng.randrange(30)
            ox, oy = cur[idx]
            nx = rng.randint(-30, 30)
            ny = rng.randint(-30, 30)
            if (nx, ny) in occupied:
                temperature *= cool_rate
                continue
            
            cur[idx] = (nx, ny)
            occupied.remove((ox, oy))
            occupied.add((nx, ny))
            v = _fast_score(cur)
            
            if v >= cur_val or rng.random() < np.exp((v - cur_val) / temperature):
                if v > best_val:
                    best_val = v
                    best_pts = list(cur)
                cur_val = v
            else:
                # revert
                cur[idx] = (ox, oy)
                occupied.remove((nx, ny))
                occupied.add((ox, oy))
            
            temperature *= cool_rate
        
        # Final local search polish
        if time.time() < deadline - 3.0:
            best_pts, best_val = _local_search(best_pts, deadline, rng)

    return best_pts
