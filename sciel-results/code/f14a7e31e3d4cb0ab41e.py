"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 115.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                area = 0.5 * abs((x[j] - x[i]) * (y[k] - y[i]) - (x[k] - x[i]) * (y[j] - y[i]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return min_area
    return min_area

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_a

def _regular_polygon(n, r=1.0):
    angles = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _run_sa(initial_pts, rng, time_budget):
    """Run Simulated Annealing starting from initial_pts."""
    start_t = time.time()
    current_pts = initial_pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T_start = 0.2
    T_end = 1e-5
    alpha = 0.999
    base_step = 0.1
    
    T = T_start
    
    while time.time() - start_t < time_budget:
        # Adaptive step size
        step_size = base_step * (T / T_start) + 0.005
        noise = rng.normal(0, step_size, (N, 2))
        new_pts = current_pts + noise
        
        if not np.all(np.isfinite(new_pts)):
            continue
            
        new_score = _score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = new_pts.copy()
        else:
            if T > T_end:
                # Accept with probability exp(-delta/T)
                # delta is negative
                prob = np.exp((new_score - current_score) / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
        
        T *= alpha
        if T < T_end:
            # Reheat if we have time, or break
            # To avoid getting stuck in a local min for the rest of the time,
            # we can restart with a new T if we are far from best
            if time.time() - start_t < time_budget * 0.8:
                T = T_start * 0.1
            else:
                break
                
    return best_pts, best_score

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = 0.0
    
    # Multi-start strategy
    # 1. Regular polygon
    # 2. Random points
    # 3. Perturbed regular polygon
    
    inits = [
        _regular_polygon(N, 1.0),
        rng.uniform(-1, 1, (N, 2)),
        _regular_polygon(N, 1.0) + rng.normal(0, 0.1, (N, 2))
    ]
    
    # Allocate time to each start
    # We have ~110s. Let's give 30s to each of 3 starts, plus final polish.
    time_per_start = 30.0
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        remaining = DEADLINE - 10 - (time.time() - start_time)
        alloc = min(time_per_start, remaining)
        
        pts, score = _run_sa(init, rng, alloc)
        
        if score > best_score:
            best_score = score
            best_pts = pts
            
    # Final Polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        def neg_score(pts_flat):
            pts = pts_flat.reshape(-1, 2)
            return -_score(pts)
            
        try:
            res = minimize(neg_score, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _regular_polygon(N, 1.0)
        
    return best_pts
# EVOLVE-BLOCK-END
