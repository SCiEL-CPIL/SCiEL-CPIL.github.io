# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

_N = 14


def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2


def _generate_tetrahedral_points(rv, rm, rc):
    """Generate 14 points based on tetrahedral symmetry orbits."""
    sqrt3 = np.sqrt(3)
    
    # Orbit 1: 4 Vertices of regular tetrahedron
    v_coords = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / sqrt3
    v_points = rv * v_coords
    
    # Orbit 2: 6 Edge Midpoints
    m_coords = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ])
    m_points = rm * m_coords
    
    # Orbit 3: 4 Face Centroids
    c_coords = np.array([
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, -1, -1]
    ]) / sqrt3
    c_points = rc * c_coords
    
    return np.vstack([v_points, m_points, c_points])


def _neg_score_radii(params):
    """Objective: negative score for minimization of radii."""
    rv, rm, rc = params
    rv = abs(rv)
    rm = abs(rm)
    rc = abs(rc)
    
    if rv == 0 and rm == 0 and rc == 0:
        return 1.0
        
    points = _generate_tetrahedral_points(rv, rm, rc)
    if not np.all(np.isfinite(points)):
        return 1.0
        
    return -_score(points)


def _soft_min_max_score(points, alpha=100.0):
    """
    Smooth surrogate for log((d_min/d_max)^2) using log-sum-exp.
    Returns 2*(softmin(log_dists) - softmax(log_dists))
    """
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return -np.inf
    
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max) or d_min <= 0:
        return -np.inf
    
    log_dists = np.log(dists)
    
    # softmin(log_dists, alpha) = -(1/alpha)*logsumexp(-alpha*log_dists)
    neg_alpha_ld = -alpha * log_dists
    neg_alpha_ld_max = np.max(neg_alpha_ld)
    softmin_val = -(1.0/alpha) * (neg_alpha_ld_max + np.log(np.sum(np.exp(neg_alpha_ld - neg_alpha_ld_max))))
    
    # softmax(log_dists, alpha) = (1/alpha)*logsumexp(alpha*log_dists)
    alpha_ld = alpha * log_dists
    alpha_ld_max = np.max(alpha_ld)
    softmax_val = (1.0/alpha) * (alpha_ld_max + np.log(np.sum(np.exp(alpha_ld - alpha_ld_max))))
    
    log_ratio = softmin_val - softmax_val
    
    if not np.isfinite(log_ratio):
        return -np.inf
    
    return 2.0 * log_ratio


def _neg_soft_score_radii(params, alpha=100.0):
    """Negative soft score for optimization."""
    rv, rm, rc = params
    rv = abs(rv)
    rm = abs(rm)
    rc = abs(rc)
    
    if rv == 0 and rm == 0 and rc == 0:
        return 1.0
        
    points = _generate_tetrahedral_points(rv, rm, rc)
    if not np.all(np.isfinite(points)):
        return 1.0
        
    soft_val = _soft_min_max_score(points, alpha)
    if not np.isfinite(soft_val):
        return 1.0
        
    return -soft_val


def _polish_with_soft(params_init, deadline, alpha_start=50.0, alpha_end=500.0, n_steps=8):
    """Polish using soft objective with increasing alpha for sharper convergence."""
    current = np.array(params_init, dtype=float)
    best_score = -np.inf
    best_params = current.copy()
    
    for i in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        t = i / max(n_steps - 1, 1)
        alpha = alpha_start * (alpha_end / alpha_start) ** t
        
        try:
            result = minimize(
                lambda p, a=alpha: _neg_soft_score_radii(p, a),
                current,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-14,
                }
            )
            
            if result.x is not None and np.all(np.isfinite(result.x)):
                current = result.x
                rv, rm, rc = abs(current[0]), abs(current[1]), abs(current[2])
                points = _generate_tetrahedral_points(rv, rm, rc)
                if np.all(np.isfinite(points)):
                    score = _score(points)
                    if score > best_score:
                        best_score = score
                        best_params = current.copy()
                        
        except Exception:
            continue
    
    return best_params, best_score


def _final_polish(params_init, deadline):
    """Final high-precision polish using exact objective with very tight tolerances."""
    current = np.array(params_init, dtype=float)
    best_score = -np.inf
    best_params = current.copy()
    
    tolerances = [1e-12, 1e-14, 1e-15]
    
    for tol in tolerances:
        if time.time() > deadline - 3:
            break
        
        try:
            result = minimize(
                _neg_score_radii,
                current,
                method='Nelder-Mead',
                options={
                    'maxiter': 15000,
                    'maxfev': 15000,
                    'xatol': tol,
                    'fatol': tol,
                }
            )
            
            if result.x is not None and np.all(np.isfinite(result.x)):
                current = result.x
                rv, rm, rc = abs(current[0]), abs(current[1]), abs(current[2])
                points = _generate_tetrahedral_points(rv, rm, rc)
                if np.all(np.isfinite(points)):
                    score = _score(points)
                    if score > best_score:
                        best_score = score
                        best_params = current.copy()
                        
        except Exception:
            continue
    
    return best_params, best_score


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses a high-symmetry combinatorial template (Tetrahedron Vertices, 
    Edge Midpoints, Face Centroids) and optimizes the 3 radial scaling factors.
    """
    n = _N
    deadline = time.time() + 580.0  # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Initial guesses for radii (rv, rm, rc)
    initial_guesses = [
        [1.0, 1.0, 1.0],
        [1.2, 1.0, 0.8],
        [1.0, 1.2, 0.8],
        [1.0, 1.0, 1.2],
        [1.5, 1.0, 0.5],
        [0.5, 1.5, 1.0],
        [1.0, 0.5, 1.5],
        [1.1, 0.9, 1.1],
        [0.9, 1.1, 0.9],
        [1.0, 1.1, 0.9]
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(10):
        initial_guesses.append(rng.uniform(0.5, 1.5, 3))
    
    best_params = None
    
    # Phase 1: Nelder-Mead on exact objective
    for idx, guess in enumerate(initial_guesses):
        if time.time() > deadline - 120:
            break
            
        try:
            result = minimize(
                _neg_score_radii,
                guess,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                }
            )
            
            rv, rm, rc = result.x
            rv, rm, rc = abs(rv), abs(rm), abs(rc)
            
            points = _generate_tetrahedral_points(rv, rm, rc)
            
            if np.all(np.isfinite(points)):
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_params = np.array([rv, rm, rc])
                    
        except Exception:
            continue

    # Phase 2: Soft objective polish with increasing alpha
    if best_params is not None and time.time() < deadline - 60:
        polished_params, polished_score = _polish_with_soft(
            best_params, deadline,
            alpha_start=50.0, alpha_end=500.0, n_steps=8
        )
        if polished_score > best_score:
            best_score = polished_score
            best_params = polished_params
            rv, rm, rc = abs(best_params[0]), abs(best_params[1]), abs(best_params[2])
            best_points = _generate_tetrahedral_points(rv, rm, rc)
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 10:
        final_params, final_score = _final_polish(best_params, deadline)
        if final_score > best_score:
            best_score = final_score
            best_params = final_params
            rv, rm, rc = abs(best_params[0]), abs(best_params[1]), abs(best_params[2])
            best_points = _generate_tetrahedral_points(rv, rm, rc)
    
    # Fallback if optimization failed
    if best_points is None:
        best_points = _generate_tetrahedral_points(1.0, 1.0, 1.0)
    
    # Ensure shape and validity
    if best_points.shape != (n, 3) or not np.all(np.isfinite(best_points)):
        golden_angle = np.pi * (3 - np.sqrt(5))
        thetas = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        phis = golden_angle * np.arange(n)
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        best_points = np.column_stack([x, y, z])

    return best_points


# EVOLVE-BLOCK-END
