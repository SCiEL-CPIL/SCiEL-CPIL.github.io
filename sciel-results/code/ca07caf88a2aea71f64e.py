import numpy as np
import time
import random
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    timeout = 550  # Safety margin for 600s limit

    # Precompute indices for projections
    # P is (10, 10) with P[y, x]
    # y ranges 0..9 (rows), x ranges 0..9 (cols)
    
    # We need to compute entropies for:
    # 1. x - y (r = -1). Range: -9 to 9. 19 bins.
    # 2. x (r = 0). Range: 0 to 9. 10 bins.
    # 3. x + y (r = 1). Range: 0 to 18. 19 bins.
    # 4. x + 2y (r = 2). Range: 0 to 27. 28 bins.
    
    # Let's create index matrices for each projection
    # y_idx: (10, 1) -> (10, 10)
    # x_idx: (1, 10) -> (10, 10)
    
    y_grid, x_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    
    # idx_neg1: x - y + 9 (to make non-negative)
    idx_neg1 = (x_grid - y_grid + 9).flatten()
    # idx_0: x
    idx_0 = x_grid.flatten()
    # idx_1: x + y
    idx_1 = (x_grid + y_grid).flatten()
    # idx_2: x + 2*y
    idx_2 = (x_grid + 2 * y_grid).flatten()
    
    # Max sizes
    size_neg1 = 19
    size_0 = 10
    size_1 = 19
    size_2 = 28

    def compute_ratio(P):
        """
        Compute the ratio H(x-y) / max(H(x), H(x+y), H(x+2y)).
        P is a flat array of length 100 or a 10x10 matrix.
        """
        if P.shape != (10, 10):
            P = P.reshape(10, 10)
        
        # Clip and normalize
        P = np.clip(P, 0, None)
        s = np.sum(P)
        if s < 1e-9:
            return 0.0
        P = P / s
        p_flat = P.flatten()
        
        # Helper to compute entropy from counts
        def entropy_from_counts(counts):
            # counts sum to 1
            nonzero = counts[counts > 1e-15]
            if len(nonzero) == 0:
                return 0.0
            return -np.sum(nonzero * np.log2(nonzero))
        
        # Compute counts for each projection using np.bincount
        # np.bincount is very fast
        
        c_neg1 = np.bincount(idx_neg1, weights=p_flat, minlength=size_neg1)
        c_0 = np.bincount(idx_0, weights=p_flat, minlength=size_0)
        c_1 = np.bincount(idx_1, weights=p_flat, minlength=size_1)
        c_2 = np.bincount(idx_2, weights=p_flat, minlength=size_2)
        
        H_neg1 = entropy_from_counts(c_neg1)
        H_0 = entropy_from_counts(c_0)
        H_1 = entropy_from_counts(c_1)
        H_2 = entropy_from_counts(c_2)
        
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return 0.0
        
        return H_neg1 / M

    # Objective function for minimizers (minimizing negative ratio)
    def objective(params):
        P = params.reshape(10, 10)
        return -compute_ratio(P)
    
    # Initial best
    P_best = np.ones((10, 10)) / 100.0
    best_score = compute_ratio(P_best)
    
    # 1. Try optimizing from uniform distribution using L-BFGS-B
    # We need to handle the simplex constraint (sum=1, P>=0).
    # L-BFGS-B supports bounds. We can fix one variable to enforce sum=1, or use penalty.
    # Better: use a transformation or just rely on bounds and normalization inside objective?
    # Normalization inside objective makes the landscape flat along the radial direction, which can slow down gradient methods.
    # However, L-BFGS-B is quite robust.
    # Alternative: use scipy.optimize.minimize with method 'L-BFGS-B' on 99 variables, fixing the 100th.
    # Or simply use bounds [0, inf] and normalize in objective. The gradient might be noisy due to normalization.
    # Let's try a different approach: use a parameterization that guarantees sum=1 and P>=0.
    # e.g., P_i = exp(theta_i) / sum(exp(theta)).
    # This maps R^100 to the simplex.
    # But we have 100 variables. We can fix one to 0 to reduce dimensionality to 99.
    
    def softmax_params_to_P(theta):
        # theta is length 99
        # theta_full is length 100, last element is 0
        theta_full = np.concatenate([theta, [0.0]])
        # Shift to avoid overflow
        theta_max = np.max(theta_full)
        exp_theta = np.exp(theta_full - theta_max)
        P_flat = exp_theta / np.sum(exp_theta)
        return P_flat
    
    def objective_softmax(theta):
        P_flat = softmax_params_to_P(theta)
        return -compute_ratio(P_flat)
    
    # Start with uniform theta (all zeros)
    theta0 = np.zeros(99)
    
    try:
        res = minimize(objective_softmax, theta0, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success:
            P_opt = softmax_params_to_P(res.x).reshape(10, 10)
            score = compute_ratio(P_opt)
            if score > best_score:
                best_score = score
                P_best = P_opt
    except Exception:
        pass
    
    if time.time() - start_time > timeout:
        return np.clip(P_best, 0, None) / np.sum(P_best)
    
    # 2. Multi-start with Differential Evolution (DE)
    # DE is robust for global search.
    # We can use the softmax parameterization again to handle constraints.
    # DE works on unconstrained domains.
    
    # Let's define a DE objective
    def de_objective(theta):
        P_flat = softmax_params_to_P(theta)
        return -compute_ratio(P_flat)
    
    # Bounds for theta: [-10, 10] is usually enough for softmax to span the simplex reasonably.
    # If theta is very large negative, P is near 0.
    # If theta is very large positive, P is near 1.
    bounds = [(-10, 10)] * 99
    
    # Run DE with a few iterations
    # Pop size 50, max iter 50
    # This might be slow. Let's check time.
    # 50 * 50 * 100 evaluations = 250k evaluations.
    # Each eval is fast (bincount). Should be okay.
    
    try:
        from scipy.optimize import differential_evolution
        
        # Use a smaller pop size and fewer iterations for speed
        # Or use the result from L-BFGS-B as a seed? DE doesn't take a single seed easily in standard interface, but we can initialize near it.
        # Standard DE initializes randomly.
        
        # Let's do a few restarts of DE with small populations
        for i in range(5):
            if time.time() - start_time > timeout - 100:
                break
            
            # Use a random initial population or seed near best
            # scipy's DE generates its own.
            
            res_de = differential_evolution(
                de_objective, bounds, 
                maxiter=20, 
                popsize=15, 
                seed=random.randint(0, 10000),
                tol=1e-8,
                disp=False
            )
            
            if res_de.success:
                P_opt = softmax_params_to_P(res_de.x).reshape(10, 10)
                score = compute_ratio(P_opt)
                if score > best_score:
                    best_score = score
                    P_best = P_opt
    except Exception:
        pass
    
    if time.time() - start_time > timeout:
        return np.clip(P_best, 0, None) / np.sum(P_best)
    
    # 3. Final polish with L-BFGS-B from the best DE result
    if best_score > compute_ratio(np.ones((10, 10)) / 100.0):
        # Convert best_P to theta
        # This is tricky. We can just run L-BFGS-B on the original 100-var space with bounds and normalization?
        # Or re-parameterize.
        # Let's try a simple local search with random perturbations in the original space, normalized.
        
        current_P = P_best.copy()
        current_score = best_score
        
        for i in range(2000):
            if time.time() - start_time > timeout - 10:
                break
            
            # Random perturbation
            delta = np.random.normal(0, 0.05, (10, 10))
            P_new = current_P + delta
            P_new = np.clip(P_new, 0, None)
            s = np.sum(P_new)
            if s < 1e-9:
                continue
            P_new = P_new / s
            
            new_score = compute_ratio(P_new)
            if new_score > current_score:
                current_P = P_new
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    P_best = current_P

    # Final cleanup
    P_best = np.clip(P_best, 0, None)
    total = np.sum(P_best)
    if total < 1e-9:
        return np.ones((10, 10)) / 100.0
    P_best = P_best / total
    
    return P_best

# EVOLVE-BLOCK-END
