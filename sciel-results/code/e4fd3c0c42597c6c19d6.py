"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.spatial import Delaunay
import time

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(c):
    """Compute the radius for a given configuration c of shape (N, 2)."""
    # Wall constraints
    wall_min = 0.5 - np.max(np.abs(c), axis=1)
    r_wall = np.min(wall_min)
    
    # Pairwise distances
    # Use broadcasting for all pairs
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to infinity
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_wall, r_pair)


def _objective_neg_radius(c_flat):
    """Negative radius to be minimized."""
    c = c_flat.reshape(N, 2)
    return -_compute_radius(c)


def _soft_radius(c, beta=1000.0):
    """Smooth approximation of the radius using log-sum-exp for min."""
    c = c.reshape(N, 2)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c).max(axis=1)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all distances into one array
    all_dists = np.concatenate([wall_dists, pair_dists.ravel()])
    
    # Soft min using log-sum-exp trick
    m = np.min(all_dists)
    # To avoid overflow, subtract m first
    log_sum = np.log(np.sum(np.exp(beta * (all_dists - m))))
    soft_min = m - np.log(log_sum) / beta
    
    return soft_min


def _gradient_soft_radius(c_flat, beta=1000.0):
    """Numerical gradient of the soft radius."""
    eps = 1e-6
    grad = np.zeros_like(c_flat)
    f0 = -_soft_radius(c_flat, beta)
    for i in range(len(c_flat)):
        c_plus = c_flat.copy()
        c_plus[i] += eps
        f_plus = -_soft_radius(c_plus, beta)
        grad[i] = (f_plus - f0) / eps
    return grad


def _is_valid(c):
    """Check if configuration is valid (inside square, no overlaps)."""
    c = c.reshape(N, 2)
    if not np.all(np.isfinite(c)):
        return False
    if np.any(np.abs(c) > 0.5 + 1e-10):
        return False
    # Check for coincident centres
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        return False
    return True


def _delaunay_contact_graph(c):
    """Identify contact graph using Delaunay triangulation."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        return []


def _initial_grid():
    """Create an initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with grid
    c_best = _initial_grid().flatten().copy()
    r_best = _compute_radius(c_best.reshape(N, 2))
    
    # Try multiple random restarts
    np.random.seed(42)
    
    # First, try to improve the grid with gradient-based optimization
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try a few different approaches
    
    # Approach 1: Direct optimization of negative radius
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            c0 = c_best.copy()
        else:
            # Random perturbation
            c0 = c_best + np.random.randn(2 * N) * 0.02
            # Clip to bounds
            c0 = np.clip(c0, -0.49, 0.49)
        
        try:
            result = minimize(
                _objective_neg_radius,
                c0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_new = result.x
            r_new = _compute_radius(c_new.reshape(N, 2))
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()
        except Exception:
            pass
    
    # Approach 2: Use soft radius for smoother optimization
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            c0 = c_best.copy()
        else:
            c0 = c_best + np.random.randn(2 * N) * 0.01
            c0 = np.clip(c0, -0.49, 0.49)
        
        try:
            result = minimize(
                lambda x: -_soft_radius(x, beta=500.0),
                c0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_new = result.x
            r_new = _compute_radius(c_new.reshape(N, 2))
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()
        except Exception:
            pass
    
    # Approach 3: Local refinement using contact graph
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        c0 = c_best + np.random.randn(2 * N) * 0.005
        c0 = np.clip(c0, -0.49, 0.49)
        
        try:
            # Use Nelder-Mead for local refinement
            result = minimize(
                _objective_neg_radius,
                c0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            c_new = result.x
            r_new = _compute_radius(c_new.reshape(N, 2))
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()
        except Exception:
            pass
    
    # Final validation
    c_final = c_best.reshape(N, 2)
    if not _is_valid(c_final):
        # Fall back to grid
        c_final = _initial_grid()
    
    return c_final
# EVOLVE-BLOCK-END
