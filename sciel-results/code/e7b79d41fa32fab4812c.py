import time
import random
import numpy as np
from collections import Counter

def _evaluate_fast(points):
    """Evaluate 16 points. Returns (is_valid, score)."""
    n = len(points)
    if n != 16:
        return False, 0
    xs = np.array([p[0] for p in points], dtype=np.int64)
    ys = np.array([p[1] for p in points], dtype=np.int64)
    if np.any(np.abs(xs) > 1000) or np.any(np.abs(ys) > 1000):
        return False, 0
    # distinctness
    pts = list(zip(xs.tolist(), ys.tolist()))
    if len(set(pts)) != 16:
        return False, 0
    # count pairs per line
    iu, ju = np.triu_indices(16, k=1)
    dx = xs[ju] - xs[iu]
    dy = ys[ju] - ys[iu]
    ax = np.abs(dx)
    ay = np.abs(dy)
    g = np.gcd(ax, ay)
    g_safe = np.where(g == 0, 1, g)
    dxn = dx // g_safe
    dyn = dy // g_safe
    neg = (dxn < 0) | ((dxn == 0) & (dyn < 0))
    dxn = np.where(neg, -dxn, dxn)
    dyn = np.where(neg, -dyn, dyn)
    c = dyn * xs[iu] - dxn * ys[iu]
    keys = list(zip(dxn.tolist(), dyn.tolist(), c.tolist()))
    cnt = Counter(keys)
    max_count = 0
    score = 0
    for v in cnt.values():
        if v > max_count:
            max_count = v
        if v == 6:
            score += 1
    if max_count >= 10:
        return False, 0
    return True, score

def _affine_seed():
    """Generate a random affine transform of a 4x4 grid."""
    base = np.array([[x, y] for y in range(4) for x in range(4)], dtype=np.int64)
    for _ in range(500):
        a = random.randint(-4, 4)
        b = random.randint(-4, 4)
        c = random.randint(-4, 4)
        d = random.randint(-4, 4)
        if a * d - b * c == 0:
            continue
        e = random.randint(-50, 50)
        f = random.randint(-50, 50)
        t = np.array([[a, b], [c, d]]) @ base.T + np.array([e, f])
        if np.any(np.abs(t) > 950):
            continue
        pts = t.T.tolist()
        is_valid, score = _evaluate_fast(pts)
        if is_valid:
            return pts, score
    return None, 0

def _local_search(points, score, deadline, max_iter=200000):
    """Run local search from given starting point."""
    best_points = [p[:] for p in points]
    best_score = score
    current_points = [p[:] for p in points]
    current_score = score
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        idx = random.randint(0, 15)
        old = current_points[idx]
        
        r = random.random()
        if r < 0.4:
            nx = old[0] + random.randint(-3, 3)
            ny = old[1] + random.randint(-3, 3)
        elif r < 0.7:
            nx = old[0] + random.randint(-10, 10)
            ny = old[1] + random.randint(-10, 10)
        else:
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)
        
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        
        current_points[idx] = [nx, ny]
        is_valid, new_score = _evaluate_fast(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
            # Accept if better, or occasionally accept worse to escape local optima
            if new_score >= current_score or (new_score >= current_score - 1 and random.random() < 0.1):
                current_score = new_score
            else:
                current_points[idx] = old
        else:
            current_points[idx] = old
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -1
    
    # Phase 1: Try multiple affine seeds to find a good starting point
    for attempt in range(50):
        if time.time() > deadline:
            break
        pts, score = _affine_seed()
        if pts is not None:
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in pts]
            if best_score >= 12:
                break
    
    # Fallback: use 4x4 grid
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, best_score = _evaluate_fast(best_points)
    
    # Phase 2: Local search from best seed
    best_points, best_score = _local_search(best_points, best_score, deadline, max_iter=100000)
    
    # Phase 3: If we have time, try more affine seeds and continue search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            pts, score = _affine_seed()
            if pts is not None:
                # Quick local search from this seed
                local_deadline = min(deadline - 2, time.time() + remaining_time / 20)
                lp, ls = _local_search(pts, score, local_deadline, max_iter=20000)
                if ls > best_score:
                    best_score = ls
                    best_points = lp
                if best_score >= 13:
                    break
    
    return best_points
