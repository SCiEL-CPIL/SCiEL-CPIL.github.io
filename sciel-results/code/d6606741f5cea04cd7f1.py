"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _pairwise_distances(pts):
    """Compute all pairwise distances for n points in d dimensions."""
    # pts: (n, d)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    return dists

def _score_from_points(pts):
    """Compute the squared ratio of max to min pairwise distance."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, D):
        return np.inf
    # Check distinctness
    dists = _pairwise_distances(pts)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-12:
        return np.inf
    ratio = max_d / min_d
    return ratio**2

def _objective(params):
    """Objective function for optimization: minimize squared distance ratio."""
    pts = params.reshape(N, D)
    return _score_from_points(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    best_score = np.inf
    best_params = None
    
    # Try multiple random initializations with Nelder-Mead
    np.random.seed(42)
    
    # Initial guess: random points in a sphere
    def random_init():
        # Generate points on a sphere or in a ball
        pts = np.random.randn(N, D)
        # Scale to unit sphere region
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.where(norms > 0, norms, 1) * np.random.uniform(0.5, 1.0, size=(N, 1))
        return pts.flatten()
    
    # Try a few different starting points
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        x0 = random_init()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
        except Exception:
            pass
    
    # If we still have time, try differential evolution for global search
    if time.time() < deadline - 10 and best_params is not None:
        try:
            # Bounds: points in [-5, 5]^3
            bounds = [(-5, 5)] * (N * D)
            
            result_de = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=True
            )
            
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_params = result_de.x.copy()
                
        except Exception:
            pass
    
    # Final polish with high-precision Nelder-Mead if we have time
    if time.time() < deadline - 3 and best_params is not None:
        try:
            result_polish = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result_polish.fun < best_score:
                best_score = result_polish.fun
                best_params = result_polish.x.copy()
                
        except Exception:
            pass
    
    if best_params is None:
        # Fallback: return random points
        best_params = random_init()
    
    return best_params.reshape(N, D)
# EVOLVE-BLOCK-END
