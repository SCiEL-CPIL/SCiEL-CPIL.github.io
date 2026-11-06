import time
import random
import numpy as np
from collections import Counter

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _score_fast(points):
    """Fast scoring using NumPy vectorization."""
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    dx_ji = pts[:, 0][None, :] - pts[:, 0][:, None]
    dy_ji = pts[:, 1][None, :] - pts[:, 1][:, None]
    
    det = dx_ji[:, :, None] * dy_ji[:, None, :] - dx_ji[:, None, :] * dy_ji[:, :, None]
    abs_det = np.abs(det)
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (ii < jj) & (jj < kk)
    vals = abs_det[mask]
    vals = vals[vals > 0]
    
    if len(vals) == 0:
        return 0
    
    vals_list = vals.tolist()
    c = Counter(vals_list)
    return max(c.values())

def _random_points(n=30, lo=-30, hi=30):
    pts = set()
    while len(pts) < n:
        pts.add((random.randint(lo, hi), random.randint(lo, hi)))
    return list(pts)

def _local_search(points, max_iters=150, neighborhood=3):
    current = list(points)
    current_score = _score_fast(current)
    n = len(current)
    no_improve_count = 0
    
    for it in range(max_iters):
        if _elapsed() > 85:
            break
        idx = random.randint(0, n-1)
        old_x, old_y = current[idx]
        best_new = None
        best_new_score = current_score
        
        for dx in range(-neighborhood, neighborhood+1):
            for dy in range(-neighborhood, neighborhood+1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = old_x + dx, old_y + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                if any(current[i][0] == nx and current[i][1] == ny for i in range(n) if i != idx):
                    continue
                current[idx] = (nx, ny)
                s = _score_fast(current)
                if s > best_new_score:
                    best_new_score = s
                    best_new = (nx, ny)
                current[idx] = (old_x, old_y)
        
        if best_new is not None:
            current[idx] = best_new
            current_score = best_new_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 15:
                nx, ny = random.randint(-30, 30), random.randint(-30, 30)
                if not any(current[i][0] == nx and current[i][1] == ny for i in range(n) if i != idx):
                    current[idx] = (nx, ny)
                    s = _score_fast(current)
                    if s >= current_score:
                        current_score = s
                    else:
                        current[idx] = (old_x, old_y)
                no_improve_count = 0
    
    return current, current_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 0
    
    n_restarts = 25
    
    for r in range(n_restarts):
        if _elapsed() > 80:
            break
        
        if r % 3 == 0:
            pts = _random_points(30)
        elif r % 3 == 1:
            xs = random.sample(range(-30, 31), 6)
            ys = random.sample(range(-30, 31), 5)
            pts = [(x, y) for x in xs for y in ys]
            existing = set(pts)
            while len(pts) < 30:
                p = (random.randint(-30, 30), random.randint(-30, 30))
                if p not in existing:
                    pts.append(p)
                    existing.add(p)
        else:
            xs = random.sample(range(-30, 31), 30)
            a = random.randint(-1, 1)
            b = random.randint(-2, 2)
            c = random.randint(-5, 5)
            pts = []
            for x in xs:
                y = a*x*x + b*x + c
                y = max(-30, min(30, y))
                pts.append((x, y))
            seen = set()
            unique_pts = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique_pts.append(p)
            while len(unique_pts) < 30:
                p = (random.randint(-30, 30), random.randint(-30, 30))
                if p not in seen:
                    seen.add(p)
                    unique_pts.append(p)
            pts = unique_pts[:30]
        
        pts, score = _local_search(pts, max_iters=100, neighborhood=3)
        
        if score > best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
