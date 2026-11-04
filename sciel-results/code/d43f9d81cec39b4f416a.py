"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_START_TIME = time.time()
_TIME_LIMIT = 130.0  # safety margin before 150s kill

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points."""
    n = len(points)
    areas = []
    for i, j, k in combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        areas.append(0.5 * abs(cross))
    return np.array(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, .volume is area
    except Exception:
        return 1e-10

def _score(points):
    """Compute min triangle area / hull area."""
    areas = _triangle_areas(points)
    h_area = _hull_area(points)
    if h_area < 1e-15:
        return 0.0
    return np.min(areas) / h_area

def _soft_min_objective(params, tau):
    """Negative soft-min triangle area / hull area."""
    points = params.reshape(-1, 2)
    areas = _triangle_areas(points)
    h_area = _hull_area(points)
    if h_area < 1e-15:
        return 0.0
    # Soft min: -tau * log(sum(exp(-a/tau))/n)
    # For stability, shift areas
    a_max = np.max(areas)
    shifted = -areas / tau
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)) / len(areas))
    soft_min = -tau * (log_sum + np.max(shifted))
    # Actually: soft_min = -tau * log(mean(exp(-areas/tau)))
    # = -tau * (log_sum_exp(-areas/tau) - log(n))
    # Let me redo:
    # log(mean(exp(-a/tau))) = log_sum_exp(-a/tau) - log(n)
    # soft_min = -tau * (log_sum_exp(-a/tau) - log(n))
    # = tau * log(n) - tau * log_sum_exp(-a/tau)
    
    # More stable computation:
    neg_a_over_tau = -areas / tau
    max_val = np.max(neg_a_over_tau)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_a_over_tau - max_val)))
    soft_min = tau * (np.log(len(areas)) - log_sum_exp)
    
    return -soft_min / h_area

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    best_points = _regular_polygon(N)
    best_score = _score(best_points)
    
    # Multiple restarts with different initial configurations
    rng = np.random.RandomState(42)
    
    # Initial guesses: regular polygon, random points in unit disk, etc.
    inits = [
        _regular_polygon(N, 1.0),
        _regular_polygon(N, 1.0) + rng.randn(N, 2) * 0.1,
        rng.uniform(-1, 1, (N, 2)),
        _regular_polygon(N, 1.0) * 0.8,
        _regular_polygon(N, 1.0) * 1.2,
    ]
    
    # Annealing schedule for tau
    tau_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for init in inits:
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        current_points = init.copy()
        current_score = _score(current_points)
        
        for tau in tau_schedule:
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
            params = current_points.flatten()
            
            try:
                result = minimize(
                    _soft_min_objective,
                    params,
                    args=(tau,),
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-12},
                )
                if result.success and result.fun < 0:
                    new_points = result.x.reshape(-1, 2)
                    new_score = _score(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points
    
    return best_points
# EVOLVE-BLOCK-END
