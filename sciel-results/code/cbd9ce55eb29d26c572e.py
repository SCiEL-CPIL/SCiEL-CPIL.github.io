"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-Start CMA-ES Optimizer for BBOB f11 (Discus).
    
    Strategy:
    1. Initial Sampling: Evaluate a small set of random points to establish 
       an initial best and identify promising regions. This ensures early 
       progress for AOCC.
    2. CMA-ES: Run Covariance Matrix Adaptation Evolution Strategy from the 
       best initial point. CMA-ES is robust and effective for smooth, 
       non-convex, and anisotropic functions like f11. It adapts the search 
       distribution's covariance matrix to the function's landscape.
    3. Multi-Start: If budget remains, restart CMA-ES from the second-best 
       initial point or a point near the current best to escape local minima 
       or refine further.
    
    This approach is deterministic in structure (though random seeds are used 
    for robustness, we can fix them for reproducibility if needed, but 
    standard CMA-ES uses randomness) and highly efficient for dim=5.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Helper to check budget and time
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=float)
            # Clip to box [-5, 5]
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Initial Sampling ---
        # Sample a few points to get a decent starting point and early progress.
        # Use a mix of random and structured points.
        # For dim=5, 20 random points is cheap and likely to find the basin.
        initial_samples = 20
        initial_xs = []
        
        # Add origin and some structured points for safety
        initial_xs.append(np.zeros(dim))
        for i in range(dim):
            x = np.zeros(dim)
            x[i] = 4.0
            initial_xs.append(x)
            x[i] = -4.0
            initial_xs.append(x)
        
        # Add random points
        rng = np.random.RandomState(42)
        for _ in range(initial_samples):
            x = rng.uniform(-4, 4, size=dim)
            initial_xs.append(x)
            
        # Evaluate initial samples
        initial_results = []
        for x in initial_xs:
            if not can_call():
                break
            fx = evaluate(x)
            initial_results.append((fx, x.copy()))
            
        if not initial_results:
            # Fallback
            x0 = np.zeros(dim)
            evaluate(x0)
            if best_x is None:
                return np.zeros(dim)
                
        # Sort by function value
        initial_results.sort(key=lambda x: x[0])
        
        # --- Phase 2: CMA-ES ---
        # Implement a lightweight CMA-ES
        def run_cmaes(x_start, max_iter=100):
            nonlocal calls_made, best_x, best_f
            
            n = dim
            # Parameters
            # Population size
            lambda_ = 4 + int(3 * np.log(n))  # ~9 for n=5
            # Step size (sigma)
            # Heuristic for sigma: based on box size
            sigma = 2.0 
            # Initial mean
            mean = x_start.copy()
            # Covariance matrix
            C = np.eye(n)
            # Eigenvalues of C
            eigvals = np.ones(n)
            # Learning rates
            c_sigma = (mu_eff + 2) / ((n + 1.3) ** 2 + mu_eff)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            # mu_eff is typically lambda/2 for weighted recombination
            mu_eff = lambda_ / 2.0
            c_cov = (2 + 2 * min(mu_eff / (n + 2), 1)) / \
                    ((n + 1) ** 2 + mu_eff + 2 * min(mu_eff / (n + 2), 1))
            
            # Weights for recombination
            # Standard weights: log ranks
            weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ / 2 + 1))
            weights /= weights.sum()
            # For simplicity, use top half weights, rest 0
            w = np.zeros(lambda_)
            w[:lambda_//2] = weights
            w /= w.sum()
            
            # Initial state
            best_so_far_in_cma = best_f
            best_x_in_cma = best_x.copy()
            
            for iter_ in range(max_iter):
                if not can_call():
                    break
                    
                # Sample lambda_ points
                # Standard normal
                z = rng.standard_normal((lambda_, n))
                # Apply covariance matrix square root (Cholesky or eigendecomp)
                # For efficiency and simplicity, use eigendecomposition
                # Update C and eigendecomp every 1/n iterations or so
                # Here we do it every iteration for simplicity, n=5 is small
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive definite
                eigvals = np.maximum(eigvals, 1e-12)
                # Cholesky-like transform: z * eigvecs * sqrt(eigvals)
                # y = eigvecs @ (sqrt(eigvals) * z.T).T
                y = (eigvecs * np.sqrt(eigvals)) @ z.T
                y = y.T
                # Points
                points = mean + sigma * y
                
                # Evaluate points
                f_vals = []
                for p in points:
                    if not can_call():
                        return
                    fx = evaluate(p)
                    f_vals.append(fx)
                    
                f_vals = np.array(f_vals)
                # Sort indices
                sorted_idx = np.argsort(f_vals)
                f_sorted = f_vals[sorted_idx]
                x_sorted = points[sorted_idx]
                
                # Update best
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                    
                # Check convergence
                if np.std(f_sorted[:lambda_//2]) < 1e-12:
                    break
                if sigma < 1e-12:
                    break
                    
                # Update mean
                new_mean = np.sum(w[:, None] * x_sorted[:lambda_//2], axis=0)
                
                # Update sigma
                # Need to compute d_sigma
                # d_sigma is a bias correction
                # Approximation for small n
                d_sigma = 1.0 + 2 * max(0, (np.sqrt((lambda_ / 2 + 1) * (n + 1 / 2)) - 1) / (n + 2)) - 1
                # Actually, standard formula:
                # d_sigma = 1 + 2*max(0, (sqrt((lambda/2+1)*(n+1/2))-1)/(n+2)) - 1
                # Let's use a simpler heuristic or standard value
                # For n=5, lambda=9, d_sigma is around 1.1
                # Let's compute it properly
                lambda_half = lambda_ // 2
                d_sigma = 1.0 + 2.0 * max(0.0, (np.sqrt((lambda_half + 1) * (n + 0.5)) - 1.0) / (n + 2.0)) - 1.0
                
                # Compute p_sigma
                # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * 1/sigma * C^(-1/2) * (new_mean - mean)
                # C^(-1/2) = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (1.0 / sigma) * (C_inv_sqrt @ (new_mean - mean))
                
                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(2 * n) - 1))
                
                # Update C
                # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
                
                # Delta heuristic
                delta_h = (2 + min(mu_eff / (n + 2), 1)) * (1 - (1 - c_c) ** (2 * iter_ + 1))
                
                # Update C
                # C = (1 - c_c) * C + c_c * (p_c * p_c.T + (1 - (1 - c_c) ** 2) * delta_h * C) 
                # Wait, the standard update is:
                # C = (1 - c_cov) * C + c_cov * (1/mu_eff * sum w_i (y_i - z_bar)(y_i - z_bar).T + h_sigma * p_sigma p_sigma.T)
                # This is getting complex. Let's use a simplified rank-mu update.
                
                # For n=5, we can afford a simpler, more robust update.
                # Let's use the standard CMA-ES update formula.
                
                # y_sorted = (x_sorted - mean) / sigma
                y_sorted = (x_sorted - mean) / sigma
                # z_bar = sum w_i y_i
                z_bar = np.sum(w[:, None] * y_sorted[:lambda_//2], axis=0)
                
                # Rank-mu update
                # C = (1 - c_cov) * C + c_cov * (1/mu_eff * sum_{i=1}^{mu} w_i (y_i - z_bar)(y_i - z_bar).T)
                # Plus p_sigma term
                # C += c_cov * h_sigma * p_sigma p_sigma.T
                
                # h_sigma is a heuristic
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * iter_ + 1)) < (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff) else 0.0
                
                # Compute the outer products
                # This is O(mu * n^2)
                # For n=5, mu=4, this is small.
                
                # Initialize update matrix
                update_mat = np.zeros((n, n))
                for i in range(lambda_ // 2):
                    diff = y_sorted[i] - z_bar
                    update_mat += w[i] * np.outer(diff, diff)
                    
                update_mat /= mu_eff
                update_mat += h_sigma * np.outer(p_sigma, p_sigma)
                
                C = (1 - c_cov) * C + c_cov * update_mat
                
                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2.0
                # Regularize if needed
                eigvals_check, _ = np.linalg.eigh(C)
                if np.min(eigvals_check) < 1e-12:
                    C += 1e-12 * np.eye(n)
                    
                # Update mean
                mean = new_mean
                
                # Check for stagnation
                if iter_ > 50 and np.linalg.norm(new_mean - mean) < 1e-6:
                    break

        # Run CMA-ES from the best initial point
        # Allocate most of the budget to this
        # We have used ~40 calls. ~460 left.
        # CMA-ES with lambda=9 uses 9 calls per iteration.
        # 460 / 9 ~ 50 iterations.
        
        # Initialize CMA-ES state variables
        # These need to be in scope for run_cmaes
        # Let's define them here and pass or use nonlocal
        # Actually, run_cmaes is a closure, so it can access nonlocal variables if defined in __call__
        
        # Let's refactor run_cmaes to take initial state or define state in __call__
        
        # Re-defining run_cmaes with proper state initialization inside __call__ scope
        def run_cmaes_optimized(x_start, max_iter=100):
            nonlocal calls_made, best_x, best_f
            
            n = dim
            lambda_ = 4 + int(3 * np.log(n))  # ~9
            sigma = 2.0 
            mean = x_start.copy()
            C = np.eye(n)
            
            c_sigma = (lambda_ / 2.0 + 2) / ((n + 1.3) ** 2 + lambda_ / 2.0)
            c_c = (4 + lambda_ / 2.0 / n) / (n + 4 + 2 * lambda_ / 2.0 / n)
            c_cov = (2 + 2 * min(lambda_ / 2.0 / (n + 2), 1)) / \
                    ((n + 1) ** 2 + lambda_ / 2.0 + 2 * min(lambda_ / 2.0 / (n + 2), 1))
            
            weights = np.log(lambda_ / 2 + 1) - np.log(np.arange(1, lambda_ / 2 + 1))
            weights /= weights.sum()
            w = np.zeros(lambda_)
            w[:lambda_//2] = weights
            w /= w.sum()
            
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            lambda_half = lambda_ // 2
            d_sigma = 1.0 + 2.0 * max(0.0, (np.sqrt((lambda_half + 1) * (n + 0.5)) - 1.0) / (n + 2.0)) - 1.0
            
            for iter_ in range(max_iter):
                if not can_call():
                    break
                    
                z = rng.standard_normal((lambda_, n))
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                y = (eigvecs * np.sqrt(eigvals)) @ z.T
                y = y.T
                points = mean + sigma * y
                
                f_vals = []
                for p in points:
                    if not can_call():
                        return
                    fx = evaluate(p)
                    f_vals.append(fx)
                    
                f_vals = np.array(f_vals)
                sorted_idx = np.argsort(f_vals)
                f_sorted = f_vals[sorted_idx]
                x_sorted = points[sorted_idx]
                
                if f_sorted[0] < best_f:
                    best_f = f_sorted[0]
                    best_x = x_sorted[0].copy()
                    
                if np.std(f_sorted[:lambda_//2]) < 1e-12 or sigma < 1e-12:
                    break
                    
                new_mean = np.sum(w[:, None] * x_sorted[:lambda_//2], axis=0)
                
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * lambda_ / 2.0) * (1.0 / sigma) * (C_inv_sqrt @ (new_mean - mean))
                
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(2 * n) - 1))
                
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * lambda_ / 2.0) * (new_mean - mean) / sigma
                
                y_sorted = (x_sorted - mean) / sigma
                z_bar = np.sum(w[:, None] * y_sorted[:lambda_//2], axis=0)
                
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * iter_ + 1)) < (1.4 + 2 / (n + 1)) * np.sqrt(lambda_ / 2.0) else 0.0
                
                update_mat = np.zeros((n, n))
                for i in range(lambda_ // 2):
                    diff = y_sorted[i] - z_bar
                    update_mat += w[i] * np.outer(diff, diff)
                    
                update_mat /= lambda_ / 2.0
                update_mat += h_sigma * np.outer(p_sigma, p_sigma)
                
                C = (1 - c_cov) * C + c_cov * update_mat
                C = (C + C.T) / 2.0
                eigvals_check, _ = np.linalg.eigh(C)
                if np.min(eigvals_check) < 1e-12:
                    C += 1e-12 * np.eye(n)
                    
                mean = new_mean
                
                if iter_ > 50 and np.linalg.norm(new_mean - mean) < 1e-6:
                    break

        # Run from best
        run_cmaes_optimized(best_x, max_iter=100)
        
        # If budget remains, try a second start from the second best initial point
        if len(initial_results) > 1 and can_call() and budget - calls_made > 50:
            # Check if we improved significantly
            # If not, restart
            run_cmaes_optimized(initial_results[1][1], max_iter=50)
            
        return best_x

# EVOLVE-BLOCK-END
