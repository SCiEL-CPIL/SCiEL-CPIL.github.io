"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _log_score(points_flat):
    """
    Compute the negative log-score for minimization.
    Score = (Product of all pairwise distances) / (Diameter)^(n*(n-1))
    We want to maximize Score, so we minimize -log(Score).
    -log(Score) = -log(Product) + n*(n-1)*log(Diameter)
                = -sum(log(d_ij)) + n*(n-1)*log(D)
    """
    n = 8
    pts = points_flat.reshape(n, 2)
    
    # Compute all pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    
    # Avoid log(0) for diagonal
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Mask diagonal for product
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    diameter = np.max(valid_dists)
    if diameter < 1e-10:
        return 1e10
        
    # Sum of logs of all ordered pairs (n*(n-1) terms)
    # Note: dists includes both (i,j) and (j,i), so we just sum all valid entries
    log_prod = np.sum(np.log(valid_dists))
    
    # Objective to minimize: -log_score = -log_prod + n*(n-1)*log(diameter)
    neg_log_score = -log_prod + n * (n - 1) * np.log(diameter)
    
    return neg_log_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 8
    # Bounds for coordinates. 
    # Since the score is scale invariant, the absolute position doesn't matter, 
    # but the relative geometry does. 
    # A diameter of 2 means points are roughly within [-1, 1] if centered at origin.
    # Using [-10, 10] gives plenty of room for the optimizer to move points.
    bounds = [(-10.0, 10.0)] * (n * 2)
    
    # Initial guess: Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    # Center at 0,0
    octagon -= np.mean(octagon, axis=0)
    
    best_score = _log_score(octagon.flatten())
    best_points = octagon.copy()
    
    # 1. Global Optimization with Differential Evolution
    # DE is good at finding global optima in multimodal spaces.
    try:
        result_de = differential_evolution(
            _log_score,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=False, # We will do our own polishing to control time
            workers=-1    # Use all available cores if supported, else 1
        )
        
        if result_de.fun < best_score:
            best_score = result_de.fun
            best_points = result_de.x.reshape(n, 2)
            
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        # If time is running out, skip refinement and return best global result
        pass
    else:
        # 2. Local Refinement with BFGS
        # Use the result from DE as the starting point for a local optimizer.
        # BFGS is efficient for smooth, differentiable functions.
        try:
            result_bfgs = minimize(
                _log_score,
                best_points.flatten(),
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            
            if result_bfgs.fun < best_score:
                best_score = result_bfgs.fun
                best_points = result_bfgs.x.reshape(n, 2)
                
        except Exception:
            pass
            
        # 3. Try a few random restarts with local optimization if time permits
        rng = np.random.default_rng(123)
        remaining_time = deadline - time.time()
        if remaining_time > 20:
            for i in range(5):
                if time.time() > deadline - 5:
                    break
                # Generate a random point set within bounds
                # We can bias towards the current best or just random
                # Let's try random points in [-2, 2]
                rand_pts = rng.uniform(-2, 2, size=(n, 2))
                x0 = rand_pts.flatten()
                
                try:
                    res = minimize(
                        _log_score,
                        x0,
                        method='BFGS',
                        options={'maxiter': 100, 'gtol': 1e-8}
                    )
                    if res.fun < best_score:
                        best_score = res.fun
                        best_points = res.x.reshape(n, 2)
                except Exception:
                    pass

    # Final scaling to ensure diameter is 2 (though the score is invariant, 
    # it's good practice to return a normalized set)
    pts = np.array(best_points)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-10:
        pts = pts * (2.0 / diameter)
    
    return [[float(x), float(y)] for x, y in pts]

# EVOLVE-BLOCK-END
