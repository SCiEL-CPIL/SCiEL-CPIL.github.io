# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

# Deadline guard: keep a safety margin before the 630s wall-clock limit.
_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0


def _objective(points: np.ndarray) -> float:
    """Compute (d_min / d_max)^2, with safeguards for degenerate cases."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if not (np.isfinite(d_min) and np.isfinite(d_max)):
        return 0.0
    if d_max <= 1e-15:
        return 0.0
    if d_min <= 1e-15:
        return 0.0
    ratio = d_min / d_max
    return float(ratio * ratio)


def _smooth_objective(points: np.ndarray, beta: float = 10.0) -> float:
    """
    Smooth surrogate objective to minimize.
    We want to maximize (d_min / d_max)^2.
    This is equivalent to maximizing 2 * (log(d_min) - log(d_max)).
    
    Approximation:
    log(d_min) ~ (1/beta) * log(sum(exp(-beta * d_i^2)))  [Soft Min of d_i]
    log(d_max) ~ (1/beta) * log(sum(exp(beta * d_i^2)))   [Soft Max of d_i]
    
    So we minimize: -2 * (log(sum(exp(-beta*d^2))) - log(sum(exp(beta*d^2)))) / beta
    Which is: 2/beta * (log(sum(exp(beta*d^2))) - log(sum(exp(-beta*d^2))))
    
    Note: To avoid overflow, we use log-sum-exp trick.
    """
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Ensure no zeros to avoid log(0) issues, though pdist of distinct points is > 0
    # If points are very close, d^2 is small.
    d2 = dists * dists
    
    # Soft max of d is approx by log-sum-exp of beta*d^2
    # log(sum(exp(beta*d2))) = beta*max(d2) + log(sum(exp(beta*(d2-max(d2)))))
    max_d2 = np.max(d2)
    min_d2 = np.min(d2)
    
    # For log-sum-exp stability
    # term_max = log(sum(exp(beta*d2)))
    # term_min = log(sum(exp(-beta*d2)))
    
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    
    x_max = beta * d2
    m_max = np.max(x_max)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(x_max - m_max)))
    
    x_min = -beta * d2
    m_min = np.max(x_min)
    log_sum_exp_min = m_min + np.log(np.sum(np.exp(x_min - m_min)))
    
    # Objective to minimize: - (log_d_min - log_d_max)
    # log_d_min approx = (1/beta) * log_sum_exp_min
    # log_d_max approx = (1/beta) * log_sum_exp_max
    
    val = (log_sum_exp_max - log_sum_exp_min) / beta
    return val


def _project_to_sphere(points: np.ndarray) -> np.ndarray:
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return points / norms


def _initial_guess(seed: int = 0) -> np.ndarray:
    """Return a random starting configuration of 14 points in 3D, projected to sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((14, 3))
    return _project_to_sphere(pts)


def _make_icosahedron_plus_2() -> np.ndarray:
    """
    Generate an icosahedron (12 points) and add 2 points.
    Icosahedron vertices are well-distributed.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ]
    pts = np.array(verts, dtype=float)
    pts = _project_to_sphere(pts)
    
    # Add 2 points. We can start with the center of two opposite faces or just random.
    # For now, add two points slightly perturbed from existing ones or random.
    # Better: Add points at (1,0,0) and (-1,0,0) if they are far from existing?
    # Icosahedron has vertices at (0, +/-1, +/-phi) etc. (1,0,0) is not a vertex.
    # Distance from (1,0,0) to (1, phi, 0) is sqrt(0 + phi^2 + 0) = phi ~ 1.6.
    # Min dist in icosahedron is 2.
    # So (1,0,0) is a good candidate for the 13th point.
    # (-1,0,0) for the 14th.
    
    new_pts = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
    new_pts = _project_to_sphere(new_pts)
    
    return np.vstack([pts, new_pts])


def _optimize_from(points: np.ndarray, deadline: float, beta: float = 10.0) -> np.ndarray:
    """
    Optimize a starting configuration using scipy.optimize.minimize on the smooth surrogate.
    """
    start = time.time()
    if start > deadline:
        return points

    # Flatten points for scipy
    x0 = points.flatten()
    
    def fun(x):
        pts = x.reshape(14, 3)
        pts = _project_to_sphere(pts)
        return _smooth_objective(pts, beta)
    
    # Nelder-Mead is robust for this kind of problem
    res = minimize(fun, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
    
    best_pts = _project_to_sphere(res.x.reshape(14, 3))
    
    # Check time and potentially do a second pass with higher beta for sharper approximation
    if time.time() < deadline - 10:
        beta2 = 50.0
        x1 = best_pts.flatten()
        
        def fun2(x):
            pts = x.reshape(14, 3)
            pts = _project_to_sphere(pts)
            return _smooth_objective(pts, beta2)
            
        res2 = minimize(fun2, x1, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
        best_pts = _project_to_sphere(res2.x.reshape(14, 3))
        
    return best_pts


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    best_points = None
    best_score = 0.0

    # List of initial configurations to try
    # 1. Icosahedron + 2
    # 2. Random starts
    starts = []
    starts.append(_make_icosahedron_plus_2())
    
    # Add some random starts
    for i in range(10):
        starts.append(_initial_guess(seed=i))
        
    # Shuffle starts to ensure we don't always try the same one first if we run out of time
    # But keep the icosahedron one first as it's a strong prior
    # Actually, let's just iterate through them.
    
    for i, init_pts in enumerate(starts):
        if time.time() > deadline:
            break
            
        # If we have limited time, use fewer iterations or skip some
        # For now, just run the optimizer
        opt_pts = _optimize_from(init_pts, deadline)
        
        score = _objective(opt_pts)
        
        if best_points is None or score > best_score:
            best_score = score
            best_points = opt_pts.copy()
            
        # If we have enough time, we can try more random starts
        # The loop above handles the first 11. If we still have time, we can add more.
        # But the current structure is fixed to 11 starts. 
        # Let's add a loop for additional random starts if time permits.
        
    # Additional random starts if time remains
    seed = 100
    while time.time() < deadline - 5:
        init_pts = _initial_guess(seed=seed)
        opt_pts = _optimize_from(init_pts, deadline)
        score = _objective(opt_pts)
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
        seed += 1

    if best_points is None:
        best_points = _initial_guess(0)

    # Final check: ensure shape and finiteness
    best_points = np.asarray(best_points, dtype=np.float64)
    if best_points.shape != (14, 3):
        best_points = _initial_guess(0)

    if not np.all(np.isfinite(best_points)):
        best_points = _initial_guess(0)

    return best_points


# EVOLVE-BLOCK-END
