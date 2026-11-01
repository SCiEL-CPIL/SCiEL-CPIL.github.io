import time
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    d = cdist(pts, pts)
    mask = ~np.eye(N, dtype=bool)
    return d[mask]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    d = _pairwise_distances(pts)
    if len(d) == 0:
        return np.inf
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _objective(x):
    """
    Objective function to minimize.
    Minimizes the variance of the log-distances.
    Projects points onto the unit sphere to enforce spherical geometry.
    """
    pts = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    pts = pts - np.mean(pts, axis=0)
    
    # Project onto unit sphere to constrain geometry
    norms = np.linalg.norm(pts, axis=1)
    # Handle zero norms
    norms[norms < 1e-9] = 1e-9
    pts = pts / norms[:, np.newaxis]
    
    d = _pairwise_distances(pts)
    
    # Avoid log(0) and penalize close points
    if np.any(d < 1e-12):
        return 1e10
        
    log_d = np.log(d)
    
    # Variance of log distances
    var = np.var(log_d)
    
    return var

def _generate_icosahedron_plus_2():
    """
    Generate points based on an icosahedron (12 vertices) plus 2 additional points.
    """
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    
    pts = np.vstack([v, p1, p2])
    return pts

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Initial candidates
    candidates = []
    candidates.append(_generate_fib_sphere_points())
    candidates.append(_generate_icosahedron_plus_2())
    
    # Add some random Gaussian starts
    for _ in range(5):
        candidates.append(np.random.randn(N, D))
        
    # We will use Basin Hopping for the structured starts (Icosahedron, Fib)
    # and standard L-BFGS for random starts, or just Basin Hopping for all.
    # Basin Hopping is slower per run but more robust.
    # Given 115s budget, we can afford ~10-20 Basin Hopping runs.
    
    max_bh_runs = 15
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 10: # Leave time for final check
            break
            
        if i >= max_bh_runs:
            break

        try:
            # Center initial points
            init_pts = init_pts - np.mean(init_pts, axis=0)
            
            # Define the step size for basin hopping
            # The scale of the problem is roughly 1.0
            step_size = 0.5
            
            # Run Basin Hopping
            # minimizer_kwargs for L-BFGS-B
            res = basinhopping(
                _objective, 
                init_pts.flatten(), 
                minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 200}},
                niter=50, # Increased iterations for better search
                stepsize=step_size,
                seed=i # Deterministic per candidate
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            # Check for distinct points
            d = _pairwise_distances(opt_pts)
            if d.min() < 1e-6:
                continue
                
            score = _true_score(opt_pts)
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts
                
        except Exception:
            continue
            
    # If we have time, run more random restarts with L-BFGS only (faster)
    if time.time() - t_start < DEADLINE - 5:
        # Do a few more quick L-BFGS runs from random starts
        for j in range(10):
            if time.time() - t_start > DEADLINE - 2:
                break
            init_pts = np.random.randn(N, D)
            init_pts = init_pts - np.mean(init_pts, axis=0)
            try:
                res = minimize(
                    _objective, 
                    init_pts.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 100}
                )
                opt_pts = res.x.reshape(N, D)
                opt_pts = opt_pts - np.mean(opt_pts, axis=0)
                d = _pairwise_distances(opt_pts)
                if d.min() < 1e-6:
                    continue
                score = _true_score(opt_pts)
                if score < best_score:
                    best_score = score
                    best_pts = opt_pts
            except Exception:
                continue

    # Fallback
    if best_pts is None:
        best_pts = _generate_fib_sphere_points()
        
    return best_pts
