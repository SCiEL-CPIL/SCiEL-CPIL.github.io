import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14
DEADLINE = 145.0  # Safety margin

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(pts):
    """
    Returns an array of areas for all unique triangles (i, j, k) with i < j < k.
    """
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Generate indices for all combinations of 3 points
    # i < j < k
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    # Vectorized cross product calculation
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (x_k - x_i)*(y_j - y_i) |
    dx1 = x[j_idx] - x[i_idx]
    dy1 = y[j_idx] - y[i_idx]
    dx2 = x[k_idx] - x[i_idx]
    dy2 = y[k_idx] - y[i_idx]
    
    cross = dx1 * dy2 - dx2 * dy1
    areas = 0.5 * np.abs(cross)
    
    return areas

def _score(pts):
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_a

def _soft_score(pts, beta):
    """
    Soft version of the score: SoftMin(Areas) / HullArea.
    We want to maximize this, so we minimize the negative.
    """
    hull_a = _hull_area(pts)
    if hull_a <= 0 or not np.isfinite(hull_a):
        return -1e10
    areas = _triangle_areas(pts)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return -1e10
    
    # Soft Min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow/underflow, use log-sum-exp trick
    # z = -beta * areas
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -700: # Underflow check
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    
    soft_min_val = -log_sum / beta
    
    if soft_min_val < 1e-12:
        return -1e10
        
    return soft_min_val / hull_a

def _neg_soft_score(pts_flat, beta):
    pts = pts_flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    s = _soft_score(pts, beta)
    return -s

def _initial_guesses():
    """Generate diverse high-quality initial guesses."""
    inits = []
    
    # 1. Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts1)
    
    # 2. Two concentric 7-gons (one rotated)
    angles_outer = 2 * np.pi * np.arange(7) / 7
    angles_inner = 2 * np.pi * np.arange(7) / 7 + np.pi / 7
    r_outer = 1.0
    r_inner = 0.5
    pts_outer = r_outer * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    pts_inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    pts2 = np.vstack([pts_outer, pts_inner])
    inits.append(pts2)
    
    # 3. Center + 2 rings (1 + 6 + 7)
    pts3 = np.array([[0.0, 0.0]])
    angles_ring1 = 2 * np.pi * np.arange(6) / 6
    pts_ring1 = 0.6 * np.column_stack([np.cos(angles_ring1), np.sin(angles_ring1)])
    angles_ring2 = 2 * np.pi * np.arange(7) / 7 + np.pi/7
    pts_ring2 = 1.0 * np.column_stack([np.cos(angles_ring2), np.sin(angles_ring2)])
    pts3 = np.vstack([pts3, pts_ring1, pts_ring2])
    inits.append(pts3)
    
    # 4. Random but constrained to convex hull shape (jittered regular polygon)
    rng = np.random.RandomState(42)
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    jitter = rng.normal(0, 0.1, size=base.shape)
    pts4 = base + jitter
    inits.append(pts4)
    
    return inits

def construct():
    start_time = time.time()
    best_pts = None
    best_score = -1e10
    
    # We will optimize -SoftScore. Lower is better.
    # Best Score (actual) is higher.
    
    inits = _initial_guesses()
    
    # Continuation parameters
    betas = [10, 50, 200, 1000]
    
    for init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        current_pts = init.copy()
        
        # Normalize initial guess to roughly unit hull area to help optimizer
        hull_a = _hull_area(current_pts)
        if hull_a > 0:
            current_pts = current_pts / np.sqrt(hull_a)
        
        # Continuation loop
        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                res = minimize(
                    _neg_soft_score,
                    current_pts.flatten(),
                    args=(beta,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    cand = res.x.reshape(N, 2)
                    # Evaluate true score
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand.copy()
                    # Update current point for next beta
                    current_pts = cand
            except Exception:
                pass
        
        # Final polish with high beta
        if time.time() - start_time < DEADLINE - 2:
            try:
                res = minimize(
                    _neg_soft_score,
                    current_pts.flatten(),
                    args=(2000,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    cand = res.x.reshape(N, 2)
                    s = _score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand.copy()
            except Exception:
                pass

    if best_pts is None:
        # Fallback
        best_pts = _initial_guesses()[0]
        
    return best_pts
