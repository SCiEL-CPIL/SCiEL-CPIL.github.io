import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n_points = 14
    dim = 3
    n_vars = n_points * dim
    
    start_time = time.time()
    timeout = 630.0
    safety_margin = 30.0
    
    def exact_score(points):
        """Compute (d_min/d_max)^2, the actual metric."""
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2
    
    def exact_objective(flat_points):
        """Minimize negative exact score."""
        points = flat_points.reshape(n_points, dim)
        if not np.all(np.isfinite(points)):
            return 1.0
        return -exact_score(points)
    
    def logsumexp_smoothed_objective(flat_points, beta):
        """Smoothed objective using log-sum-exp approximation.
        Minimizes -log(min_dist) + log(max_dist), approximated via log-sum-exp."""
        points = flat_points.reshape(n_points, dim)
        if not np.all(np.isfinite(points)):
            return 1.0
        dists = pdist(points)
        if np.any(dists < 1e-12):
            return 1.0
        # log-sum-exp for min: -logsumexp(-dists/beta)/beta
        neg_dists = -dists / beta
        max_neg = np.max(neg_dists)
        logsumexp_neg = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
        log_min = -logsumexp_neg * beta
        
        # log-sum-exp for max: logsumexp(dists/beta)/beta
        pos_dists = dists / beta
        max_pos = np.max(pos_dists)
        logsumexp_pos = max_pos + np.log(np.sum(np.exp(pos_dists - max_pos)))
        log_max = logsumexp_pos * beta
        
        # We want to maximize (d_min/d_max)^2, so minimize -(2*log_min - 2*log_max)
        return -(2.0 * (log_min - log_max))
    
    def fibonacci_sphere(n, dim=3):
        """Generate n points on a sphere using Fibonacci lattice."""
        points = np.zeros((n, dim))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            z = 1.0 - 2.0 * i / (n - 1)
            r = np.sqrt(max(0, 1.0 - z*z))
            theta = golden_angle * i
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
            points[i, 2] = z
        return points
    
    def get_initial_guesses():
        """Generate diverse initial guesses."""
        guesses = []
        # Fibonacci sphere with various scales and rotations
        base = fibonacci_sphere(n_points)
        for scale in [0.5, 1.0, 1.5]:
            for rot_seed in range(4):
                rng = np.random.RandomState(rot_seed * 100 + 42)
                # Random rotation matrix
                Q, _ = np.linalg.qr(rng.randn(dim, dim))
                rotated = base @ Q.T
                guesses.append((rotated * scale).flatten())
        
        # Random perturbations of Fibonacci sphere
        for pert_seed in range(8):
            rng = np.random.RandomState(pert_seed * 7 + 13)
            perturbed = base + rng.randn(n_points, dim) * 0.1
            guesses.append(perturbed.flatten())
        
        return guesses
    
    best_points = None
    best_score = -np.inf
    
    def update_best(flat_pts):
        nonlocal best_points, best_score
        pts = flat_pts.reshape(n_points, dim)
        if not np.all(np.isfinite(pts)):
            return
        s = exact_score(pts)
        if s > best_score:
            best_score = s
            best_points = flat_pts.copy()
    
    # Phase 1: Multi-start with smoothed objective using DE-like exploration
    elapsed = time.time() - start_time
    if elapsed > timeout - safety_margin:
        return np.random.randn(n_points, dim)
    
    # Use differential evolution with smoothed objective (adaptive beta)
    bounds = [(-2.0, 2.0)] * n_vars
    
    # Try multiple beta values for the smoothed objective
    betas = [0.5, 0.2, 0.1, 0.05]
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > timeout - safety_margin - 60:
            break
        
        def obj_smoothed(flat_pts):
            return logsumexp_smoothed_objective(flat_pts, beta)
        
        try:
            res = differential_evolution(
                obj_smoothed,
                bounds,
                seed=int(beta * 1000) % 10000,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube'
            )
            
            if res.x is not None:
                update_best(res.x)
                
        except Exception:
            pass
    
    # Phase 2: Local refinement from best point and diverse starts
    elapsed = time.time() - start_time
    if elapsed > timeout - safety_margin:
        if best_points is not None:
            return best_points.reshape(n_points, dim)
        return np.random.randn(n_points, dim)
    
    # Collect candidate starting points for polish
    candidates = []
    if best_points is not None:
        candidates.append(best_points.copy())
    
    for guess in get_initial_guesses()[:6]:
        candidates.append(guess)
    
    # Polish each candidate with exact objective using L-BFGS-B
    for i, cand in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > timeout - safety_margin - 15:
            break
        
        try:
            res_nm = minimize(
                exact_objective,
                cand,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            
            if res_nm.x is not None:
                update_best(res_nm.x)
                
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish on the best found
    elapsed = time.time() - start_time
    if elapsed > timeout - safety_margin - 20 and best_points is not None:
        if best_points is not None:
            try:
                res_nm2 = minimize(
                    exact_objective,
                    best_points,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_nm2.x is not None:
                    update_best(res_nm2.x)
            except Exception:
                pass
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > timeout - 5:
        if best_points is not None:
            return best_points.reshape(n_points, dim)
        return np.random.randn(n_points, dim)
    
    if best_points is None:
        return np.random.randn(n_points, dim)
    
    return best_points.reshape(n_points, dim)
