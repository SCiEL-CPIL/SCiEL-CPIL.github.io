import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
I_IDX = TRI_INDICES[:, 0]
J_IDX = TRI_INDICES[:, 1]
K_IDX = TRI_INDICES[:, 2]

def _score(pts):
    """Compute min triangle area / convex hull area."""
    if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return -1e10
    try:
        hull = ConvexHull(pts)
        h_area = hull.volume
    except Exception:
        return -1e10
    if h_area < 1e-8:
        return -1e10
    pi = pts[I_IDX]
    pj = pts[J_IDX]
    pk = pts[K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    min_a = np.min(np.abs(cross)) * 0.5
    return min_a / h_area

def _neg_score(pts_flat):
    pts = pts_flat.reshape(N, 2)
    s = _score(pts)
    if s < -1e9:
        return 1e10
    return -s

def _initial_regular():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_star(rng):
    """Two rings."""
    n_outer = N // 2
    n_inner = N - n_outer
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    r_out = 1.0
    r_in = 0.6
    x = np.concatenate([r_out * np.cos(angles_out), r_in * np.cos(angles_in)])
    y = np.concatenate([r_out * np.sin(angles_out), r_in * np.sin(angles_in)])
    pts = np.column_stack([x, y])
    idx = rng.permutation(N)
    return pts[idx]

def _initial_random(rng):
    # Points in unit disk
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2 * np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def _initial_grid(rng):
    """Grid-like initialization to avoid clustering."""
    # Create a 4x4 grid, pick 14 points
    x_coords = np.linspace(-1, 1, 4)
    y_coords = np.linspace(-1, 1, 4)
    pts = []
    for x in x_coords:
        for y in y_coords:
            pts.append([x, y])
    pts = np.array(pts)
    # Shuffle and take first 14
    idx = rng.permutation(len(pts))[:N]
    return pts[idx]

def _nm_polish(pts, maxiter=2000):
    """Polish using Nelder-Mead on hard objective."""
    try:
        res = minimize(
            _neg_score,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.fun < _neg_score(pts.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.RandomState(42)
    
    best_pts = _initial_regular()
    best_score = _score(best_pts)
    
    def time_ok():
        return time.time() < deadline
    
    # List of initializations
    inits = [
        _initial_regular(),
        _initial_star(rng),
        _initial_random(rng),
        _initial_grid(rng),
        _initial_random(rng),
        _initial_star(rng)
    ]
    
    # Shuffle inits to ensure diversity in order
    rng.shuffle(inits)
    
    # Phase 1: Multi-start Nelder-Mead on Hard Objective
    # NM is derivative-free and handles the non-smoothness of min-area well.
    # We allocate time based on number of inits.
    # Each NM run should be capped in time to ensure we try all starts.
    
    time_per_run = (deadline - time.time()) / (len(inits) + 1) # +1 for final polish
    
    for i, init_pts in enumerate(inits):
        if not time_ok():
            break
        
        run_deadline = time.time() + time_per_run - 1.0
        if run_deadline > deadline:
            run_deadline = deadline - 1.0
            
        # We cannot easily pass a time limit to minimize, so we estimate maxiter.
        # NM in 28D: ~1000-2000 iterations usually suffice for good convergence.
        # Function eval is fast (< 1ms). 2000 evals ~ 2s.
        # So 2000-5000 iterations is safe.
        
        max_iter_est = 3000
        if time_per_run < 2.0:
            max_iter_est = 1000
            
        try:
            polished = _nm_polish(init_pts, maxiter=max_iter_est)
            score = _score(polished)
            
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            continue
        
        # Check time
        if time.time() > deadline - 15.0:
            break

    # Phase 2: Final Intensive Polish
    if time_ok():
        # Run a very high-precision NM on the best candidate
        # Use smaller tolerances
        try:
            final_pts = _nm_polish(best_pts, maxiter=5000)
            fs = _score(final_pts)
            if fs > best_score:
                best_score = fs
                best_pts = final_pts
        except Exception:
            pass

    return best_pts
