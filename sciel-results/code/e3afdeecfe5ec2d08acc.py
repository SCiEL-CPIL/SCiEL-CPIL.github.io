import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

def _compute_score(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _smoothed_score(pts, temp=0.02):
    """Smoothed min triangle area / hull area using log-sum-exp."""
    n = len(pts)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                areas[idx] = 0.5 * abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                idx += 1
    
    if len(areas) == 0:
        return 0.0
    
    # Log-sum-exp for minimum
    scaled = -areas / temp
    max_s = np.max(scaled)
    if max_s > 700:  # prevent overflow
        log_sum = max_s
    else:
        log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    smooth_min = -temp * log_sum
    
    return smooth_min / hull_area

def _neg_smoothed(flat, temp=0.02):
    pts = flat.reshape(N, 2)
    return -_smoothed_score(pts, temp)

def _neg_raw(flat):
    pts = flat.reshape(N, 2)
    return -_compute_score(pts)

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # 15s safety margin
    
    rng = np.random.RandomState(123)
    best_score = 0.0
    best_pts = None
    
    def try_optimize(init_pts, max_iter=150, atol=1e-6):
        nonlocal best_score, best_pts
        if time.time() > deadline:
            return
        try:
            res = minimize(_neg_smoothed, init_pts.flatten(), method='Powell',
                          options={'maxiter': max_iter, 'xatol': atol, 'fatol': 1e-8})
            pts = res.x.reshape(N, 2)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Generate diverse initial configurations
    inits = []
    
    # Regular 13-gon
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(ang), np.sin(ang)]))
    
    # Two concentric rings
    for r_out, r_in, n_out, n_in in [(1.0, 0.5, 9, 4), (1.0, 0.4, 8, 5), (1.0, 0.6, 7, 6)]:
        ang_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
        ang_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + np.pi/n_out
        pts = np.vstack([
            r_out*np.column_stack([np.cos(ang_out), np.sin(ang_out)]),
            r_in*np.column_stack([np.cos(ang_in), np.sin(ang_in)])
        ])
        inits.append(pts)
    
    # Random in unit disk (multiple)
    for _ in range(15):
        pts = []
        for _ in range(N):
            r = np.sqrt(rng.random()) * 0.95
            th = rng.uniform(0, 2*np.pi)
            pts.append([r*np.cos(th), r*np.sin(th)])
        inits.append(np.array(pts))
    
    # Random on circle + jitter
    for _ in range(10):
        ang = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(-0.1, 0.1, N)
        r = 1.0 + rng.uniform(-0.05, 0.05, N)
        inits.append(np.column_stack([r*np.cos(ang), r*np.sin(ang)]))
    
    # Optimize each initial config
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        try_optimize(init, max_iter=120)
    
    # Refine best with raw score
    if best_pts is not None and time.time() < deadline - 15:
        try:
            res = minimize(_neg_raw, best_pts.flatten(), method='Powell',
                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
            pts = res.x.reshape(N, 2)
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass
    
    # Small perturbations around best
    if best_pts is not None and time.time() < deadline - 5:
        for _ in range(8):
            if time.time() > deadline:
                break
            pert = best_pts + rng.normal(0, 0.02, (N, 2))
            try:
                res = minimize(_neg_raw, pert.flatten(), method='Powell',
                              options={'maxiter': 80, 'xatol': 1e-7, 'fatol': 1e-9})
                pts = res.x.reshape(N, 2)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
            except Exception:
                pass
    
    if best_pts is None:
        ang = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(ang), np.sin(ang)])
    
    return best_pts
