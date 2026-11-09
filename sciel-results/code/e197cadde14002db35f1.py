"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Surrogate-based optimizer using local quadratic models and random initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Store recent points for surrogate modeling
        # We keep a sliding window of the last ~30 points
        history_x = []
        history_f = []
        max_history = 30
        
        def add_to_history(x, f):
            nonlocal best_x, best_f
            history_x.append(x)
            history_f.append(f)
            if len(history_x) > max_history:
                history_x.pop(0)
                history_f.pop(0)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse Random Search
        # Use 10% of the budget to find a good starting point
        random_calls = max(10, total_budget // 10)
        
        for i in range(random_calls):
            if time.time() >= deadline or calls_made >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            calls_made += 1
            add_to_history(x, f)
            
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(dim)
            f = func(x)
            calls_made += 1
            add_to_history(x, f)

        # Phase 2: Surrogate-based Local Search
        # Use the remaining budget to refine the best point found
        # We will fit a quadratic model to the recent history and minimize it
        
        while calls_made < total_budget:
            if time.time() >= deadline:
                break
                
            # Ensure we have enough points to fit a quadratic model
            # Quadratic in d dims has 1 + d + d*(d+1)/2 parameters
            # For d=5: 1 + 5 + 15 = 21 parameters. Need at least 21 points.
            if len(history_x) < 21:
                # If we don't have enough, do a small local random step
                x_new = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
                f_new = func(x_new)
                calls_made += 1
                add_to_history(x_new, f_new)
                continue

            X = np.array(history_x)
            F = np.array(history_f)
            
            # Fit a quadratic model: f(x) = c0 + c1^T x + x^T A x
            # Design matrix for least squares
            n = len(X)
            d = dim
            # Number of parameters: 1 (bias) + d (linear) + d*(d+1)/2 (quadratic)
            num_params = 1 + d + d * (d + 1) // 2
            
            Phi = np.zeros((n, num_params))
            for i in range(n):
                x = X[i]
                Phi[i, 0] = 1
                Phi[i, 1:1+d] = x
                # Quadratic terms
                idx = 1 + d
                for r in range(d):
                    for c in range(r, d):
                        Phi[i, idx] = x[r] * x[c]
                        idx += 1
                        
            try:
                # Solve least squares
                coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, F, rcond=None)
                
                # Extract coefficients
                c0 = coeffs[0]
                c1 = coeffs[1:1+d]
                # Reconstruct symmetric matrix A from upper triangular coefficients
                A = np.zeros((d, d))
                idx = 1 + d
                for r in range(d):
                    for c in range(r, d):
                        val = coeffs[idx]
                        A[r, c] = val
                        A[c, r] = val
                        idx += 1
                        
                # Ensure A is symmetric
                A = (A + A.T) / 2
                
                # Define surrogate function
                def surrogate(x):
                    return c0 + c1 @ x + x @ A @ x
                
                # Define surrogate gradient
                def surrogate_grad(x):
                    return c1 + 2 * A @ x
                
                # Minimize the surrogate
                # Start from the current best x
                res = minimize(surrogate, best_x, jac=surrogate_grad, 
                               method='L-BFGS-B', bounds=[(-5, 5)] * d)
                
                x_proposed = res.x
                f_proposed = surrogate(x_proposed)
                
                # If the surrogate minimum is very close to current best, 
                # we might be stuck in a local min of the surrogate.
                # Add a small perturbation or check if we should explore more.
                
                # Evaluate the actual function at the proposed point
                # Clip to bounds
                x_proposed = np.clip(x_proposed, -5, 5)
                
                # If the proposed point is too close to the current best, 
                # it might not improve things significantly. 
                # However, since we are using a local model, this is expected.
                
                f_actual = func(x_proposed)
                calls_made += 1
                add_to_history(x_proposed, f_actual)
                
                # If we didn't improve, we might need to restart the local search 
                # from a slightly different location or expand the search window.
                # For now, we just continue. The history window will shift.
                
            except Exception:
                # If fitting fails, do a random step
                x_new = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                f_new = func(x_new)
                calls_made += 1
                add_to_history(x_new, f_new)

        return best_x, best_f
# EVOLVE-BLOCK-END
