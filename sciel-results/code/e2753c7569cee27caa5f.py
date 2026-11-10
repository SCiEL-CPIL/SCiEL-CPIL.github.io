"""Multi-Objective Optimizer for RE21 using Augmented Tchebycheff Scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # 1. Initialization: Latin Hypercube Sampling
        # We use a portion of the budget for initial exploration to ensure the front is well-covered.
        # 40 points is a good balance for dim=4.
        n_init = 40
        X_init = self._latin_hypercube(n_init, self.dim, lo, hi)
        
        # Evaluate initial points
        F_init = np.zeros((n_init, self.n_obj))
        n_evals = 0
        for i in range(n_init):
            if n_evals >= self.budget or time.time() > deadline:
                break
            F_init[i] = problem(X_init[i])
            n_evals += 1
            
        X_archive = X_init[:n_evals].copy()
        F_archive = F_init[:n_evals].copy()
        
        # Helper to update archive
        def add_point(x, f):
            nonlocal X_archive, F_archive, n_evals
            X_archive = np.vstack([X_archive, x.reshape(1, -1)])
            F_archive = np.vstack([F_archive, f.reshape(1, -1)])
            n_evals += 1

        # 2. Augmented Tchebycheff Scalarization
        # Strategy:
        # - Generate a set of weight vectors w_k that cover the unit simplex.
        # - For each weight, minimize: max_i ( w_i * (f_i(x) - z_i^*) ) + rho * sum_i ( w_i * (f_i(x) - z_i^*) )
        # where z^* is the ideal point (min of each objective).
        # - Use L-BFGS-B for the scalar optimization, starting from the best initial point for that weight.
        # - This effectively performs a series of directed local searches that cover the front.
        
        # Get ideal point from initial archive
        if len(F_archive) == 0:
            return X_archive
        
        z_star = np.min(F_archive, axis=0)
        
        # Generate weight vectors
        # For 2 objectives, weights are (w, 1-w).
        # We want to spread them across the front.
        # Let's use 10 weight vectors.
        n_weights = 10
        weights = np.linspace(0.05, 0.95, n_weights)
        
        # For each weight, run a scalar optimization
        for w in weights:
            if n_evals >= self.budget or time.time() > deadline:
                break
            
            # Weight vector for 2 objectives
            w_vec = np.array([w, 1.0 - w])
            
            # Define scalar objective
            # Augmented Tchebycheff: max_i ( w_i * (f_i - z_i*) ) + rho * sum_i ( w_i * (f_i - z_i*) )
            # Note: Since we are minimizing, and z* is the ideal point, f_i - z_i* >= 0.
            # We want to minimize this scalar function.
            
            def scalar_obj(x):
                f = problem(x)
                # Check if x is valid (should be, due to bounds)
                # Calculate Tchebycheff part
                scaled = w_vec * (f - z_star)
                tcheb = np.max(scaled)
                # Calculate augmented part
                # rho is a parameter, typically 0.05
                rho = 0.05
                aug = rho * np.sum(scaled)
                return tcheb + aug

            # Define counted problem to track evaluations
            # We need to be careful: L-BFGS-B might call the objective many times.
            # We have a limited budget.
            # We will set a maxiter limit on L-BFGS-B to ensure we don't blow the budget.
            # Also, we will use a wrapper to count evals and stop if budget is exceeded.
            
            # Find best initial point for this weight from the archive
            # Score each archive point by the scalar objective value
            # We don't have the f values for all archive points in a vectorized way easily here,
            # but we can compute the scalar value for each archive point.
            
            # Compute scalar values for archive
            # F_archive shape (N, 2)
            # z_star shape (2,)
            # w_vec shape (2,)
            scaled_archive = w_vec * (F_archive - z_star)
            tcheb_archive = np.max(scaled_archive, axis=1)
            aug_archive = 0.05 * np.sum(scaled_archive, axis=1)
            scalar_vals = tcheb_archive + aug_archive
            
            # Pick the best (lowest) scalar value
            best_idx = np.argmin(scalar_vals)
            x0 = X_archive[best_idx].copy()
            
            # Run L-BFGS-B
            # maxiter: limit iterations. Each iteration might involve multiple function evals.
            # Let's allow up to 10 iterations.
            # gtol: gradient tolerance.
            
            # We need a wrapper to count evals and raise an exception if budget is exceeded,
            # or simply check inside the loop? L-BFGS-B doesn't support early stopping via callback easily
            # without modifying the objective to return a large value or raise an exception.
            
            # Let's use a callback to check time and budget.
            # Actually, simpler: just limit maxiter and rely on the fact that L-BFGS-B is fast.
            # But we must not exceed budget.
            
            # Let's calculate remaining budget
            remaining_budget = self.budget - n_evals
            if remaining_budget <= 0:
                break
            
            # Set maxiter to a small number, e.g., 5-10, to conserve budget for other weights.
            # With 10 weights, if we use 15 evals per weight, that's 150 evals.
            # 40 init + 150 = 190. Safe.
            # L-BFGS-B with maxiter=10 might use ~10-20 evals.
            
            max_iter = 5 # Conservative
            
            try:
                res = minimize(
                    scalar_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': max_iter, 'maxfun': 10, 'ftol': 1e-6}
                )
                
                # Evaluate the final point if it's different from x0
                x_opt = res.x
                # Check if x_opt is in archive (approximate)
                # Just evaluate and add
                f_opt = problem(x_opt)
                add_point(x_opt, f_opt)
                
            except Exception:
                # If optimization fails, we just skip this weight
                pass
        
        return X_archive

    def _latin_hypercube(self, n, dim, lo, hi):
        """Generate Latin Hypercube points."""
        pts = np.zeros((n, dim))
        for i in range(dim):
            # Random permutation of indices
            perm = np.random.permutation(n)
            # Generate uniform points in (0, 1)
            # For LHS, we divide [0,1] into n intervals and pick one point from each.
            # The point in the i-th interval (0-indexed) is (perm[i] + u_i) / n
            u = np.random.rand(n)
            pts[:, i] = (perm + u) / n
        return lo + pts * (hi - lo)

# EVOLVE-BLOCK-END
