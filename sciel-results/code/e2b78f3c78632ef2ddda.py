# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    deadline = time.time() + 610  # safety margin before 630s kill

    def score(points):
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(flat):
        pts = flat.reshape(n, d)
        return -score(pts)

    def soft_min_max_objective(flat, beta=50.0):
        """
        Smooth surrogate: minimize -soft_min(dists) + soft_max(dists)
        where soft_min = -logsumexp(-dists/beta)/beta and soft_max = logsumexp(dists/beta)/beta
        Actually we want to maximize min/max, so minimize (soft_max - soft_min)
        Or equivalently minimize -soft_min + soft_max
        """
        pts = flat.reshape(n, d)
        dists = pdist(pts)
        if np.any(dists == 0):
            return 1e10
        # soft min and max using log-sum-exp
        # soft_min = -logsumexp(-dists/beta)/beta
        # soft_max = logsumexp(dists/beta)/beta
        # We want to maximize soft_min/soft_max, or minimize soft_max - soft_min
        # Let's minimize soft_max - soft_min
        scaled_max = dists / beta
        scaled_min = -dists / beta
        # logsumexp for stability
        max_val = np.max(scaled_max)
        logsumexp_max = max_val + np.log(np.sum(np.exp(scaled_max - max_val)))
        min_val = np.max(scaled_min)
        logsumexp_min = min_val + np.log(np.sum(np.exp(scaled_min - min_val)))
        
        soft_max = logsumexp_max * beta
        soft_min = -logsumexp_min * beta
        
        # We want to maximize soft_min / soft_max
        # Equivalent to minimizing soft_max - soft_min (in a normalized sense)
        # Or minimizing -log(soft_min/soft_max) = log(soft_max) - log(soft_min)
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
        return -np.log(soft_min / soft_max)

    # Initial guess: 4x4 grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    grid = np.array([(xx, yy) for yy in y for xx in x])
    
    best_points = grid.copy()
    best_score = score(best_points)
    
    # Try several initializations
    inits = [grid]
    
    # Random initialization
    np.random.seed(42)
    for i in range(5):
        inits.append(np.random.randn(n, d))
    
    # Concentric rings
    ring1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    ring2 = np.array([[2*np.cos(np.pi/4), 2*np.sin(np.pi/4)],
                      [2*np.cos(3*np.pi/4), 2*np.sin(3*np.pi/4)],
                      [2*np.cos(5*np.pi/4), 2*np.sin(5*np.pi/4)],
                      [2*np.cos(7*np.pi/4), 2*np.sin(7*np.pi/4)]])
    inits.append(np.vstack([ring1, ring2]))
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Optimize using smooth surrogate first
        try:
            res = minimize(soft_min_max_objective, init.flatten(), 
                          method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if time.time() > deadline:
                break
            candidate = res.x.reshape(n, d)
            s = score(candidate)
            if s > best_score:
                best_score = s
                best_points = candidate.copy()
        except Exception:
            pass
        
        # Then polish with the actual objective
        if time.time() > deadline:
            break
        try:
            res2 = minimize(objective, best_points.flatten(),
                           method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            if time.time() > deadline:
                break
            candidate2 = res2.x.reshape(n, d)
            s2 = score(candidate2)
            if s2 > best_score:
                best_score = s2
                best_points = candidate2.copy()
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
