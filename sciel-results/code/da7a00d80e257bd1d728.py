"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    d = dist_sq[iu]
    
    if len(d) == 0:
        return 1e10
        
    min_d_sq = np.min(d)
    max_d_sq = np.max(d)
    
    if min_d_sq <= 1e-12:
        return 1e10
        
    return max_d_sq / min_d_sq

def _objective(points_flat):
    """
    Objective function for optimization.
    Minimizes (max_dist / min_dist)^2.
    """
    points = points_flat.reshape(N, D)
    return _compute_score(points)

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses Differential Evolution for global search, followed by Nelder-Mead polish.
    """
    # Define bounds for each coordinate. 
    # Since the ratio is scale-invariant, we can bound the points to a sphere of radius 1.
    # This helps the optimizer converge faster.
    bounds = [(-1.0, 1.0) for _ in range(N * D)]
    
    # Run Differential Evolution
    # maxiter and tol control the precision. 
    # seed ensures reproducibility.
    try:
        result_de = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-9, 
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True  # Uses Nelder-Mead at the end
        )
        
        best_points = result_de.x.reshape(N, D)
        best_score = result_de.fun
        
    except Exception:
        # Fallback if DE fails
        best_points = np.random.randn(N, D)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
        best_score = _compute_score(best_points)

    # Additional local polish from the DE result and a few random starts
    # to ensure we are in a tight local minimum.
    
    candidates = [best_points]
    
    # Add a few random spherical points as additional local starts
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = rng.standard_normal((N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
        
    # Also try the Fibonacci sphere initialization as a candidate
    phi = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((N, D))
    for i in range(N):
        z = 1 - (i + 0.5) * 2.0 / N
        r = np.sqrt(1 - z * z)
        theta = phi * i
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    candidates.append(fib_pts)

    for init_pts in candidates:
        x0 = init_pts.flatten()
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            score = _compute_score(res.x.reshape(N, D))
            if score < best_score:
                best_score = score
                best_points = res.x.reshape(N, D)
        except Exception:
            continue

    return best_points
# EVOLVE-BLOCK-END
