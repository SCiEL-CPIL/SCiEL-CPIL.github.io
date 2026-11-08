import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.spatial import ConvexHull
import time
import itertools
import sys

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

def _soft_score(pts, beta):
    """Soft-min objective for smoother landscape."""
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
    cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) * 0.5
    # Soft min using log-sum-exp trick
    if beta < 1e-6:
        return -np.min(cross) / h_area
    # -soft_min = -(-1/beta * log(sum(exp(-beta*c)))) = (1/beta)*log(sum(exp(-beta*c)))
    # We want to maximize soft_min, so minimize -soft_min
    # soft_min ≈ -(1/beta) * log(sum(exp(-beta * c_i)))
    # -soft_min ≈ (1/beta) * log(sum(exp(-beta * c_i)))
    # For numerical stability:
    m = np.min(cross)
    shifted = cross - m
    soft_min_val = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))
    return soft_min_val / h_area

def _neg_soft(pts_flat, beta):
    pts = pts_flat.reshape(N, 2)
    return -_soft_score(pts, beta)

def _neg_hard(pts_flat):
    pts = pts_flat.reshape(N, 2)
    return -_score(pts)

def _initial_regular():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_perturbed(rng, base=None):
    if base is None:
        base = _initial_regular()
    noise = rng.normal(0, 0.1, base.shape)
    return base + noise

def _initial_star(rng):
    """Two rings: inner and outer."""
    n_outer = N // 2
    n_inner = N - n_outer
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    r_out = 1.0
    r_in = 0.5
    x = np.concatenate([r_out * np.cos(angles_out), r_in * np.cos(angles_in)])
    y = np.concatenate([r_out * np.sin(angles_out), r_in * np.sin(angles_in)])
    pts = np.column_stack([x, y])
    # Shuffle to mix
    idx = rng.permutation(N)
    return pts[idx]

def _initial_grid(rng):
    """Perturbed grid-like arrangement."""
    n_per_row = 4
    rows = (N + n_per_row - 1) // n_per_row
    pts = []
    for i in range(rows):
        for j in range(n_per_row):
            if i * n_per_row + j >= N:
                break
            x = (j - (n_per_row - 1) / 2.0) * 0.6
            y = (i - (rows - 1) / 2.0) * 0.6
            pts.append([x, y])
    pts = np.array(pts)
    pts += rng.normal(0, 0.05, pts.shape)
    return pts

def _lbfgs_polish(pts, beta, maxiter=2000):
    """Polish using L-BFGS-B on soft objective."""
    res = minimize(
        _neg_soft,
        pts.flatten(),
        args=(beta,),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
    )
    if res.success or res.fun < _neg_soft(pts.flatten(), beta):
        return res.x.reshape(N, 2)
    return pts

def _nm_polish(pts, maxiter=2000):
    """Final hard-objective polish with Nelder-Mead."""
    res = minimize(
        _neg_hard,
        pts.flatten(),
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    if res.fun < _neg_hard(pts.flatten()):
        return res.x.reshape(N, 2)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Aggressive safety margin to avoid timeout
    
    rng = np.random.RandomState(42)
    
    # Start with a good regular polygon configuration
    best_pts = _initial_regular()
    best_score = _score(best_pts)
    
    def time_ok():
        return time.time() < deadline
    
    # Generate a smaller set of diverse initial points
    # We reduce the count from 12 to 5 to save time
    initial_sets = [
        _initial_regular(),
        _initial_perturbed(rng, _initial_regular()),
        _initial_star(rng),
        _initial_grid(rng),
        _initial_perturbed(rng, _initial_star(rng)),
    ]
    
    # Shorter beta schedule, fewer iterations
    beta_schedule = [50.0, 10.0, 2.0, 0.5]
    
    for init_pts in initial_sets:
        if not time_ok():
            break
        
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        for beta in beta_schedule:
            if not time_ok():
                break
            
            # Reduced maxiter for L-BFGS-B
            polished = _lbfgs_polish(current_pts, beta, maxiter=200)
            polished_score = _score(polished)
            
            if polished_score > current_score:
                current_pts = polished
                current_score = polished_score
        
        # Skip Nelder-Mead for intermediate steps to save time, 
        # we will do a final polish on the best result later.
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
    
    # Phase 2: Intensive polish on the best candidate found
    if time_ok():
        # Use a finer beta schedule for the final polish
        final_beta_schedule = [20.0, 5.0, 1.0, 0.5, 0.2]
        for beta in final_beta_schedule:
            if not time_ok():
                break
            polished = _lbfgs_polish(best_pts, beta, maxiter=500)
            ps = _score(polished)
            if ps > best_score:
                best_score = ps
                best_pts = polished
        
        # Final Nelder-Mead polish with limited iterations
        if time_ok():
            final_pts = _nm_polish(best_pts, maxiter=500)
            fs = _score(final_pts)
            if fs > best_score:
                best_score = fs
                best_pts = final_pts

    return best_pts
