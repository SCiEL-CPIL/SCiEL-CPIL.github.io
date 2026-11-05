# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import basinhopping, minimize

def _score(points):
    # points: (8, 2) array
    # Calculate all pairwise distances
    # Use broadcasting
    # dist[i, j] = ||p_i - p_j||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Diameter
    D = np.max(dists)
    if D < 1e-10:
        return -1e100
        
    # Product of distances for i != j
    # Mask diagonal
    mask = ~np.eye(8, dtype=bool)
    d_non_diag = dists[mask]
    
    # Sum of logs
    # Avoid log(0)
    d_non_diag = np.maximum(d_non_diag, 1e-10)
    log_prod = np.sum(np.log(d_non_diag))
    
    # Score = exp(log_prod - 8 * log(D))
    # We want to maximize this, so minimize negative log score
    neg_log_score = -(log_prod - 8 * np.log(D))
    return neg_log_score

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    # Initial guess: Regular Octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    p0 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = _score(p0)
    best_points = p0
    
    # Optimization
    # We will run a few basinhopping instances with different initial guesses or just one robust one
    
    # To save time, we limit the number of basinhopping iterations
    # and the number of steps in the minimizer.
    
    # Define the local minimizer
    def local_minimizer(x):
        return minimize(_score, x, method='BFGS', options={'maxiter': 50})
        
    # We can try a few restarts
    # 1. Regular octagon
    # 2. Regular octagon + noise
    # 3. Two squares (one rotated) + noise
    
    restarts = [p0]
    
    # Add a perturbed version
    rng = np.random.default_rng(42)
    p_pert = p0 + rng.normal(0, 0.1, p0.shape)
    restarts.append(p_pert)
    
    # Try a configuration with lower symmetry?
    # e.g. 4 points on x-axis, 4 on y-axis? No, that's a cross.
    # Let's just rely on basinhopping to find the way out if it's a saddle.
    
    for p_init in restarts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = basinhopping(_score, p_init, niter=20, stepsize=0.1, minimizer_kwargs={'method': 'BFGS', 'options': {'maxiter': 30}}, seed=42)
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x
        except Exception:
            pass

    # Final check and return
    # The evaluator scales to diameter 2, but our points are already roughly diameter 2.
    # We should return the points as is. The evaluator handles scaling.
    # But wait, if my points have diameter D, the evaluator scales by 2/D.
    # The score calculation in _score already accounts for this scaling invariance.
    # So best_points is the optimal shape.
    
    # Ensure distinct points
    # If points are too close, _score would have been bad.
    
    return best_points.tolist()
# EVOLVE-BLOCK-END
