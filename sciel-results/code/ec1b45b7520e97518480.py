import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer for 150s limit
    
    n = 8
    num_pairs = n * (n - 1) // 2  # 28
    
    def log_product_minus_scaled_log_diameter(pts_flat):
        """
        Objective function to maximize.
        Returns the negative of the log-score (to be minimized) or just the log-score?
        differential_evolution minimizes by default.
        So we return - (sum log d - 28 log D).
        """
        if time.time() > deadline:
            return 0.0 # Arbitrary penalty
        
        pts = pts_flat.reshape(n, 2)
        
        # Calculate pairwise distances
        # Using broadcasting
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        
        # Avoid log(0)
        d = np.maximum(d, 1e-10)
        
        log_prod = np.sum(np.log(d))
        D = np.max(d)
        
        # If D is very small, the term -28 log D goes to infinity (good for maximization, bad for minimization of negative).
        # But D cannot be 0 for distinct points.
        # The function to maximize is log_prod - 28 log D.
        # We minimize the negative: 28 log D - log_prod.
        
        return 28 * np.log(D) - log_prod

    def compute_score(pts):
        """Calculate the actual score for the final output."""
        pts = np.array(pts)
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        D = np.max(d)
        if D < 1e-9:
            return -np.inf
        pts_scaled = pts / D * 2.0
        dists_scaled = np.linalg.norm(pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :], axis=2)
        d_scaled = dists_scaled[iu]
        log_delta = np.sum(np.log(d_scaled))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        return score

    # Bounds for differential evolution
    # Since the problem is scale invariant, we can constrain the points to a box.
    # If points are too far apart, D increases, and the penalty 28 log D increases.
    # If points are too close, log_prod decreases.
    # A box of [-2, 2] is safe.
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    # Regular octagon as a seed
    angles = 2 * np.pi * np.arange(n) / n
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    octagon_flat = octagon.flatten()
    
    best_score = compute_score(octagon)
    best_pts = octagon
    
    # Run Differential Evolution
    # maxiter=1000 might be too slow. Let's try 200-500.
    # tol=1e-8
    # popsize=15
    try:
        if time.time() < deadline - 20: # Ensure we have time for refinement
            res = differential_evolution(
                log_product_minus_scaled_log_diameter, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-8,
                popsize=15,
                polish=True, # Use L-BFGS-B to polish the best result
                polish_options={'maxiter': 1000}
            )
            
            if res.success:
                candidate_pts = res.x.reshape(n, 2)
                cand_score = compute_score(candidate_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate_pts
                    
            # Also check the initial octagon again just in case
            # DE might have moved away if the octagon is a local max and DE is noisy.
            # But DE minimizes the negative log score. The octagon is a strong candidate.
            
    except Exception as e:
        pass

    # Additional local refinement from multiple random starts if time permits
    if time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        # Try a few random perturbations of the octagon and random points
        inits = [octagon.flatten()]
        for _ in range(5):
            inits.append(octagon.flatten() + rng.normal(0, 0.1, size=2*n))
        for _ in range(5):
            inits.append(rng.uniform(-1, 1, size=2*n))
            
        for init_flat in inits:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    log_product_minus_scaled_log_diameter, 
                    init_flat, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                if res.success:
                    candidate_pts = res.x.reshape(n, 2)
                    cand_score = compute_score(candidate_pts)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = candidate_pts
            except Exception as e:
                pass

    return best_pts.tolist()
