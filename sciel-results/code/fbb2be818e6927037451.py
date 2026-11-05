<<<<<<< SEARCH
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score(points):
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return -np.inf
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return np.exp(log_score)
=======
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score(points):
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    dists = dists[mask]
    diameter = np.max(dists)
    if diameter < 1e-12:
        return -np.inf
    log_prod = np.sum(np.log(dists))
    log_score = log_prod - n * (n - 1) * np.log(diameter)
    return np.exp(log_score)

def _points_from_coefficients(coeffs):
    """Convert polynomial coefficients to 8 point coordinates."""
    # coeffs: [c7, c6, ..., c1, c0] for z^7 + c7*z^6 + ... + c0
    # Use np.roots which expects [c7, c6, ..., c0]
    try:
        roots = np.roots(coeffs)
    except Exception:
        return None
    # Filter to get 8 roots (should always be 8 for degree 7)
    if len(roots) != 8:
        return None
    pts = np.column_stack([roots.real, roots.imag])
    return pts

def _coefficients_from_points(pts):
    """Convert 8 points to polynomial coefficients (monic)."""
    roots = pts[:, 0] + 1j * pts[:, 1]
    coeffs = np.poly(roots)  # Returns [1, c6, c5, ..., c0]
    return coeffs[1:]  # Drop leading 1

def _neg_log_score_from_coeffs(coeffs):
    """
    Optimize over polynomial coefficients.
    Maximize log-product of pairwise distances subject to max |root| <= 1.
    Uses soft-max penalty for the constraint.
    """
    n = 8
    pts = _points_from_coefficients(coeffs)
    if pts is None:
        return 1e10
    
    # Check for near-duplicate points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    pairwise_dists = dists[mask]
    
    if np.min(pairwise_dists) < 1e-10:
        return 1e10
    
    # Soft-max constraint: penalty for max |root| > 1
    moduli = np.abs(pts[:, 0] + 1j * pts[:, 1])
    beta = 50.0
    soft_max = np.logaddexp.reduce(moduli)  # This gives log-sum-exp, not soft-max
    
    # Better: use log-sum-exp / beta as soft-max approximation
    log_sum_exp = np.log(np.sum(np.exp(beta * moduli)))
    soft_max_val = log_sum_exp / beta
    
    constraint_violation = np.maximum(0, soft_max_val - 1.0)
    penalty = 100.0 * constraint_violation ** 2
    
    log_prod = np.sum(np.log(pairwise_dists))
    
    # We want to maximize log_prod, so minimize -log_prod + penalty
    return -log_prod + penalty

def _neg_log_score_cartesian(pts_flat):
    """Direct Cartesian optimization with diameter constraint via penalty."""
    n = 8
    pts = pts_flat.reshape(n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    pairwise_dists = dists[mask]
    
    if np.min(pairwise_dists) < 1e-10:
        return 1e10
    
    diameter = np.max(pairwise_dists)
    if diameter < 1e-12:
        return 1e10
    
    # Soft constraint: penalize if diameter > 2
    beta = 50.0
    soft_diam = np.logaddexp.reduce(diameter * beta) / beta  # Not quite right
    
    # Simpler: just use the diameter directly with a penalty
    constraint_violation = np.maximum(0, diameter - 2.0)
    penalty = 100.0 * constraint_violation ** 2
    
    log_prod = np.sum(np.log(pairwise_dists))
    log_score = log_prod - n * (n - 1) * np.log(max(diameter, 1e-10))
    
    return -log_score + penalty
>>>>>>> REPLACE
