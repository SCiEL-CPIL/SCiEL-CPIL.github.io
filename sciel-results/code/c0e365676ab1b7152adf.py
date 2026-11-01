"""Rewritten program: minimize squared max/min pairwise distance ratio for N=14 in 3D."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 14
D = 3

def _pairwise_dists_sq(pts):
    """Return matrix of squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=-1)

def _score(pts):
    """Compute (max_dist / min_dist)^2. Returns inf if any points coincide."""
    d2 = _pairwise_dists_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    d2_off = d2[mask]
    if d2_off.size == 0:
        return np.inf
    min_d2 = np.min(d2_off)
    max_d2 = np.max(d2_off)
    if min_d2 <= 1e-12:
        return np.inf
    ratio = np.sqrt(max_d2 / min_d2)
    return ratio * ratio

def _objective(x):
    """Objective for optimizer: squared ratio. We want to minimize this."""
    pts = x.reshape(N, D)
    return _score(pts)

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_pts = None
    best_score = np.inf
    
    # Strategy: use differential evolution to find good basins,
    # then refine with L-BFGS-B.
    
    # Define bounds for DE: center at origin, radius ~5
    bounds = [(-5, 5)] * (N * D)
    
    # Use a smooth surrogate for DE to help it navigate
    # Actually, let's just use the true objective but with DE's robustness.
    # DE can handle non-smoothness (kinks at min/max switches) better than gradient methods.
    
    # To speed up DE, we can use a smaller population and fewer generations initially,
    # or use the 'best1bin' strategy.
    
    # Let's try DE first.
    # DE can be slow. We need to be careful with the time budget.
    # Let's set a time limit for DE by checking time in the callback or just running a fixed number of iterations.
    # scipy's differential_evolution doesn't have a direct time limit, but we can monitor it.
    
    # Alternative: Random restarts with L-BFGS-B might be faster and effective if we have good inits.
    # The knowledge says DE finds better basins. Let's try a hybrid.
    
    # 1. Generate a few random initial points
    # 2. Run DE for a short time
    # 3. Take the best result and refine with L-BFGS-B
    
    # Let's implement a time-aware DE loop.
    # Since we can't easily interrupt DE, we will run it with limited iterations.
    # Or we can use `minimize` with method='Nelder-Mead' for robustness, but DE is better for global.
    
    # Let's try running DE with maxiter=100, popsize=15.
    # This should take a reasonable amount of time.
    
    start_time = time.time()
    
    try:
        # Run Differential Evolution
        # Using 'best1bin' strategy
        res_de = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=False  # We will polish ourselves
        )
        
        if res_de.success or res_de.fun < best_score:
            best_pts = res_de.x.reshape(N, D)
            best_score = res_de.fun
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        if best_pts is None:
            # Fallback to simple grid or random
            pts = np.random.randn(N, D)
            best_pts = pts
            best_score = _score(pts)
        return best_pts
    
    # Refine with L-BFGS-B
    # L-BFGS-B requires smooth objective. The ratio function is smooth almost everywhere.
    # We will try to refine the best point found by DE.
    
    if best_pts is not None:
        try:
            res_lbfgs = minimize(
                _objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_lbfgs.fun < best_score:
                best_pts = res_lbfgs.x.reshape(N, D)
                best_score = res_lbfgs.fun
        except Exception:
            pass
    
    # If we still have time, try a few random restarts with L-BFGS-B
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for i in range(10):
            if time.time() > deadline - 10:
                break
            try:
                # Random init on a sphere
                init = np.random.randn(N, D)
                init /= np.linalg.norm(init, axis=1, keepdims=True) * 2
                res_r = minimize(
                    _objective,
                    init.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res_r.fun < best_score:
                    best_pts = res_r.x.reshape(N, D)
                    best_score = res_r.fun
            except Exception:
                pass
    
    if best_pts is None:
        # Final fallback
        best_pts = np.random.randn(N, D)
    
    return best_pts
