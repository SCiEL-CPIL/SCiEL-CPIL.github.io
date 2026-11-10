import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x, beta=10.0):
        """
        Smooth approximation of - (d_min / d_max)^2 using log-sum-exp.
        d_min ~ -1/beta * log(sum(exp(-beta * d)))
        d_max ~  1/beta * log(sum(exp( beta * d)))
        """
        pts = x.reshape((n, d))
        # Centering helps with translation invariance, though DE handles it
        pts = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts)
        if len(dists) == 0 or np.any(~np.isfinite(dists)):
            return 1e6 # Large penalty for invalid

        # Numerical stability for log-sum-exp
        # d_min approx
        min_term = -beta * dists
        min_max_val = np.max(min_term)
        if min_max_val > 700: # Avoid overflow
             min_log_sum_exp = min_max_val + np.log(np.sum(np.exp(min_term - min_max_val)))
        else:
             min_log_sum_exp = np.log(np.sum(np.exp(min_term)))
        d_min_approx = -min_log_sum_exp / beta
        
        # d_max approx
        max_term = beta * dists
        max_max_val = np.max(max_term)
        if max_max_val > 700:
             max_log_sum_exp = max_max_val + np.log(np.sum(np.exp(max_term - max_max_val)))
        else:
             max_log_sum_exp = np.log(np.sum(np.exp(max_term)))
        d_max_approx = max_log_sum_exp / beta
        
        if d_max_approx < 1e-12:
            return 1e6
            
        ratio = d_min_approx / d_max_approx
        # We want to maximize ratio^2, so minimize -ratio^2
        return -ratio ** 2

    def exact_objective(x):
        """Exact objective for local polishing."""
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return 1e6
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1e6
        return -(d_min / d_max) ** 2

    best_score = 0.0
    best_points = None

    # 1. Global Search with Differential Evolution
    # Bounds: Since the problem is scale invariant, we can constrain the search space.
    # A cube of side length 20 is usually sufficient for normalized configurations.
    bounds = [(-10, 10)] * (n * d)
    
    # DE parameters
    # popsize is multiplier of N. N=42. popsize=20 -> 840 individuals.
    # maxiter=500: 840 * 500 = 420k evals. Each eval ~10-20us. Total ~4-8s. Fast.
    
    try:
        de_result = differential_evolution(
            smooth_objective,
            bounds,
            args=(10.0,), # beta
            maxiter=500,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=True, # Uses L-BFGS-B at the end of DE
            updating='immediate',
            init='sobol' # Sobol initialization is often better for uniform coverage
        )
        
        if de_result.success or de_result.fun < 0:
            cand = de_result.x.reshape((n, d))
            cand = cand - np.mean(cand, axis=0)
            s = score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
    except Exception:
        pass

    # 2. Local Refinement from multiple random starts + DE result
    # We will try to polish the best DE result and a few random starts using L-BFGS-B on the smooth objective
    # and then on the exact objective.
    
    candidates_to_polish = []
    if best_points is not None:
        candidates_to_polish.append(best_points.copy())
    
    rng = np.random.RandomState(123)
    # More diverse initial points: random, spherical shell, and perturbed Fibonacci
    for _ in range(10):
        # Random points in a sphere of radius 5
        pts = rng.randn(n, d) * 5
        pts = pts - np.mean(pts, axis=0)
        candidates_to_polish.append(pts)
    
    # Add Fibonacci sphere initialization as another candidate
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    fib_pts = np.column_stack([x, y, z])
    fib_pts = fib_pts - np.mean(fib_pts, axis=0)
    candidates_to_polish.append(fib_pts)

    for init_pts in candidates_to_polish:
        if time.time() > deadline - 60:
            break
            
        x0 = init_pts.flatten()
        
        # Multi-stage beta annealing: start with lower beta, increase for sharper optimization
        for beta in [10.0, 20.0, 50.0, 100.0]:
            if time.time() > deadline - 60:
                break
            try:
                res_smooth = minimize(
                    smooth_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if res_smooth.x is not None:
                    cand = res_smooth.x.reshape((n, d))
                    cand = cand - np.mean(cand, axis=0)
                    s = score(cand)
                    if s > best_score:
                        best_score = s
                        best_points = cand.copy()
                    # Use improved point as starting point for next beta
                    x0 = cand.flatten()
            except Exception:
                pass

        # Polish with Exact Objective (Nelder-Mead or Powell, as gradients are tricky for min/max)
        # L-BFGS-B on non-smooth can fail. Let's use Powell or Nelder-Mead.
        # Powell is often robust.
        try:
            res_exact = minimize(
                exact_objective,
                x0,
                method='Powell',
                options={'maxiter': 1000, 'ftol': 1e-14, 'xtol': 1e-14}
            )
            if res_exact.x is not None:
                cand = res_exact.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass

    # 3. Final high-precision polish pass if we have time
    if best_points is not None and np.all(np.isfinite(best_points)) and time.time() < deadline - 30:
        # Do a final very precise polish on the best solution found
        try:
            # Use multiple methods for final polish
            x0 = best_points.flatten()
            
            # Nelder-Mead with very tight tolerance
            res_nm = minimize(
                exact_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'fatol': 1e-16, 'xatol': 1e-16, 'adaptive': True}
            )
            if res_nm.x is not None:
                cand = res_nm.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass
        
        try:
            # Another Powell pass with different starting perturbation
            if time.time() < deadline - 20:
                x0 = best_points.flatten()
                # Add tiny random perturbation to escape potential local minima
                perturbation = rng.randn(n * d) * 1e-6
                x0_perturbed = x0 + perturbation
                
                res_powell2 = minimize(
                    exact_objective,
                    x0_perturbed,
                    method='Powell',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'xtol': 1e-15}
                )
                if res_powell2.x is not None:
                    cand = res_powell2.x.reshape((n, d))
                    cand = cand - np.mean(cand, axis=0)
                    s = score(cand)
                    if s > best_score:
                        best_score = s
                        best_points = cand.copy()
        except Exception:
            pass
    
    # 4. Final Check and Normalization
    if best_points is not None and np.all(np.isfinite(best_points)):
        # The metric is scale invariant, so normalization is not strictly required for the score,
        # but it ensures the points are "finite" and well-behaved.
        # We can scale them to have max distance 1 or just leave them.
        # The task says "returning an array of shape (14, 3) of finite point coordinates".
        # It does not require a specific scale.
        pass
    else:
        # Fallback: Fibonacci sphere
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        best_points = np.column_stack([x, y, z])
        best_points = best_points - np.mean(best_points, axis=0)

    return best_points
