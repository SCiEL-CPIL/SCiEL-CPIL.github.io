"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _log_score_vec(pts):
    """
    Fast vectorized computation of log(Delta / n^n) after scaling to diameter 2.
    pts: numpy array of shape (8, 2)
    Returns the log of the score (higher is better).
    """
    n = pts.shape[0]
    if n < 2:
        return -1e10
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Mask diagonal (self-distances)
    np.fill_diagonal(d2, 1.0)  # temporary, will be excluded
    d = np.sqrt(d2)
    # Diameter = max pairwise distance
    D = np.max(d)
    if D < 1e-12:
        return -1e10
    s = 2.0 / D
    # Log of product of all ordered-pair distances after scaling
    d_scaled = d * s
    np.fill_diagonal(d_scaled, 1.0)
    
    # Numerical stability: clip very small distances to avoid log(0)
    d_scaled = np.clip(d_scaled, 1e-15, None)
    
    log_p = np.sum(np.log(d_scaled))
    log_score = log_p - n * math.log(n)
    return log_score

def _objective(pts_flat):
    """
    Objective for scipy minimization (we minimize negative log score).
    pts_flat: array of 16 floats (8 points x 2 coords)
    """
    pts = pts_flat.reshape(8, 2)
    ls = _log_score_vec(pts)
    return -ls

def _two_squares_points(params):
    """
    Generate 8 points from two squares with independent radii, rotation, and center offset.
    params: [r1, r2, theta1, theta2, cx, cy]
    Square 1: 4 points at radius r1, rotation theta1, centered at (cx, cy)
    Square 2: 4 points at radius r2, rotation theta2, centered at (cx, cy)
    """
    r1, r2, theta1, theta2, cx, cy = params
    angles1 = 2 * math.pi * np.arange(4) / 4 + theta1
    angles2 = 2 * math.pi * np.arange(4) / 4 + theta2
    pts1 = np.column_stack([r1 * np.cos(angles1) + cx, r1 * np.sin(angles1) + cy])
    pts2 = np.column_stack([r2 * np.cos(angles2) + cx, r2 * np.sin(angles2) + cy])
    return np.vstack([pts1, pts2])

def _obj_manifold(params):
    """Objective for the 6-parameter two-square manifold."""
    pts = _two_squares_points(params)
    ls = _log_score_vec(pts)
    return -ls

def _obj_full(pts_flat):
    """Objective for full 16-parameter space."""
    pts = pts_flat.reshape(8, 2)
    ls = _log_score_vec(pts)
    return -ls

def construct():
    """
    Find 8 distinct points maximizing the product of distances at fixed diameter.
    Strategy:
    1. Use differential_evolution on a 6-parameter two-square manifold to find a good symmetric configuration.
    2. Refine with full 16-parameter local optimization (Powell + Nelder-Mead) to potentially break symmetry.
    3. Use multiple random initializations for robustness.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_pts = None
    best_log_score = -1e10

    def regular_octagon(radius=1.0):
        angles = 2 * math.pi * np.arange(8) / 8
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    # Start with regular octagon
    best_pts = regular_octagon()
    best_log_score = _log_score_vec(best_pts)

    # Phase 1: Differential evolution on 6-parameter two-square manifold
    # Parameters: [r1, r2, theta1, theta2, cx, cy]
    # Bounds: radii in [0.1, 3.0], angles in [0, 2*pi], center in [-1, 1]
    bounds = [(0.1, 3.0), (0.1, 3.0), (0.0, 2 * math.pi), (0.0, 2 * math.pi), (-1.0, 1.0), (-1.0, 1.0)]
    
    if time.time() < deadline - 60.0:
        try:
            de_result = differential_evolution(
                _obj_manifold,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=20,
                mutation=(0.3, 0.8),
                recombination=0.8,
                polish=False
            )
            if de_result.success or de_result.fun < -best_log_score:
                best_params = de_result.x
                best_pts_de = _two_squares_points(best_params)
                ls_de = _log_score_vec(best_pts_de)
                if ls_de > best_log_score:
                    best_log_score = ls_de
                    best_pts = best_pts_de
        except Exception:
            pass

    # Phase 2: Full 16-parameter local optimization from multiple starting points
    # Collect candidates: the DE result, regular octagon, and some random perturbations
    candidates = []
    
    # Add current best
    if best_pts is not None:
        candidates.append(best_pts.copy())
    
    # Add regular octagon
    candidates.append(regular_octagon().copy())
    
    # Add some random perturbations of the regular octagon
    random.seed(42)
    np.random.seed(42)
    for pert in [0.05, 0.1, 0.2, 0.3, 0.5]:
        base = regular_octagon()
        noise = np.random.randn(8, 2) * pert
        init = base + noise
        candidates.append(init)
    
    # Add some random points in unit disk
    for trial in range(5):
        init = np.random.randn(8, 2)
        D = np.max(np.sqrt(np.sum((init[:, np.newaxis, :] - init[np.newaxis, :])**2, axis=2)))
        if D > 1e-12:
            init = init * (2.0 / D)
        candidates.append(init)

    # Shuffle candidates
    random.shuffle(candidates)

    for i, init in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
        
        # Check distinctness
        d2_mat = np.sum((init[:, np.newaxis, :] - init[np.newaxis, :])**2, axis=2)
        np.fill_diagonal(d2_mat, 1.0)
        if np.min(d2_mat) < 1e-8:
            continue
        
        init_flat = init.flatten()
        
        # Powell method for broader local search
        try:
            res_p = minimize(
                _obj_full, init_flat,
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            pts_p = res_p.x.reshape(8, 2)
            ls_p = _log_score_vec(pts_p)
            if ls_p > best_log_score:
                best_log_score = ls_p
                best_pts = pts_p
        except Exception:
            pass

        # Nelder-Mead for fine polishing
        if time.time() > deadline - 10.0:
            break
        
        try:
            res_nm = minimize(
                _obj_full, init_flat,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_nm = res_nm.x.reshape(8, 2)
            ls_nm = _log_score_vec(pts_nm)
            if ls_nm > best_log_score:
                best_log_score = ls_nm
                best_pts = pts_nm
        except Exception:
            pass

    # Phase 3: Final high-precision polish on the absolute best
    if time.time() < deadline - 2.0 and best_pts is not None:
        try:
            res_final = minimize(
                _obj_full, best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_final = res_final.x.reshape(8, 2)
            ls_final = _log_score_vec(pts_final)
            
            if ls_final > best_log_score:
                best_log_score = ls_final
                best_pts = pts_final
        except Exception:
            pass

    # Convert back to list of [x, y]
    best_list = [[float(p[0]), float(p[1])] for p in best_pts]
    
    # Ensure distinctness
    for a in range(8):
        for b in range(a + 1, 8):
            d = math.hypot(best_list[a][0] - best_list[b][0], best_list[a][1] - best_list[b][1])
            if d < 1e-6:
                best_list[b][0] += 1e-5 * (b - a + 1)
                best_list[b][1] += 1e-5 * (b - a + 2)
    
    # Ensure coordinate bounds
    for p in best_list:
        if abs(p[0]) > 1e6:
            p[0] = math.copysign(1e6 - 1, p[0])
        if abs(p[1]) > 1e6:
            p[1] = math.copysign(1e6 - 1, p[1])
    
    return best_list
# EVOLVE-BLOCK-END
