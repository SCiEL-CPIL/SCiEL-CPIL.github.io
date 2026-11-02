import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 145.0  # Safety margin before the 150s kill

# Pre-compute triple indices for vectorization
_ti = []
_tj = []
_tk = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _ti.append(i)
            _tj.append(j)
            _tk.append(k)
_ti = np.array(_ti, dtype=np.int32)
_tj = np.array(_tj, dtype=np.int32)
_tk = np.array(_tk, dtype=np.int32)

def _hull_area(pts):
    """Calculate convex hull area. Returns 0.0 on failure."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(pts):
    """Calculate areas of all 286 triangles. Vectorized."""
    x = pts[:, 0]
    y = pts[:, 1]
    # Cross product magnitude / 2
    cross = np.abs(
        (x[_tj] - x[_ti]) * (y[_tk] - y[_ti])
        - (y[_tj] - y[_ti]) * (x[_tk] - x[_ti])
    )
    return 0.5 * cross

def _score_points(pts):
    """
    Calculate the score: min triangle area / convex hull area.
    pts: (N, 2) array of points.
    """
    h_area = _hull_area(pts)
    if h_area < 1e-12:
        return 0.0
    
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
        
    return float(min_area / h_area)

def _neg_score_from_angles(angles):
    """
    Objective function for minimization.
    angles: array of 13 angles.
    Returns negative of the score (min triangle area / hull area).
    """
    angles = np.asarray(angles, dtype=np.float64)
    # Ensure angles are in [0, 2*pi) for consistency, though cos/sin are periodic
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    score = _score_points(pts)
    return -score

def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = -1e18
    best_angles = None
    
    def _update(angles):
        nonlocal best_pts, best_score, best_angles
        if angles is None:
            return
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape != (N,):
            return
        
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_angles = angles.copy()

    # Initial guess: Regular polygon
    angles0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    _update(angles0)
    
    # Define bounds for differential evolution
    bounds = [(0, 2 * np.pi) for _ in range(N)]
    
    # Run Differential Evolution multiple times with different seeds
    # DE is stochastic, so multiple runs with different seeds help explore the space.
    # We will run DE, then polish the result locally.
    
    num_runs = 10
    maxiter_de = 300
    popsize_de = 20
    
    for run in range(num_runs):
        if time.time() > DEADLINE - 20.0:
            break
            
        # Use different seed for each run to diversify
        seed = 42 + run * 101
        
        try:
            # Differential Evolution
            # tol and maxiter control convergence
            res = differential_evolution(
                _neg_score_from_angles,
                bounds,
                seed=seed,
                maxiter=maxiter_de,
                popsize=popsize_de,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False # We will do our own polish
            )
            
            angles_de = res.x
            _update(angles_de)
            
            # Local Polish using Powell (good for non-smooth)
            if time.time() < DEADLINE - 5.0:
                res_polish = minimize(
                    _neg_score_from_angles,
                    angles_de,
                    method='Powell',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                _update(res_polish.x)
                
                # Additional Nelder-Mead polish if time permits
                if time.time() < DEADLINE - 2.0:
                    res_nm = minimize(
                        _neg_score_from_angles,
                        res_polish.x,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    _update(res_nm.x)
                    
        except Exception as e:
            # If DE fails, continue to next run
            continue

    # If we still have time, try to polish the global best one more time with tighter settings
    if best_angles is not None and time.time() < DEADLINE - 2.0:
        try:
            res_final = minimize(
                _neg_score_from_angles,
                best_angles,
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            _update(res_final.x)
        except Exception:
            pass

    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
