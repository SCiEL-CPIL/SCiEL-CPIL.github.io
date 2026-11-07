"""Rewritten optimizer: IDW Surrogate + Adaptive Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """IDW Surrogate with Adaptive Uncertainty Estimation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim

        # Hyperparameters for IDW
        alpha = 2.0  # IDW exponent
        epsilon = 1e-6
        beta_start = 3.0
        beta_end = 0.5
        
        # --- Phase 1: Initial Sobol Sample ---
        n_init = 64
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        points_unit = sobol.random(n_init)
        points = -5.0 + 10.0 * points_unit
        X = np.array(points)
        Y = np.array([func(x) for x in X])
        
        calls_made = n_init
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]
        
        # Pre-generate a large pool of global candidates from Sobol
        sobol_global = qmc.Sobol(d=n, scramble=True, seed=123)
        global_pool_unit = sobol_global.random(10000)
        global_pool = -5.0 + 10.0 * global_pool_unit
        global_idx = 0
        pool_size = len(global_pool)

        # Main Loop
        while calls_made < budget:
            # 1. Update Global Variance
            y_var = np.var(Y)
            if y_var < 1e-10:
                y_var = 1e-10

            # 2. Generate Candidates
            n_global_cands = 50
            n_local_cands = 50
            
            # Refill pool if needed
            if global_idx + n_global_cands > pool_size:
                new_pool_unit = sobol_global.random(10000)
                global_pool = -5.0 + 10.0 * new_pool_unit
                global_idx = 0
                pool_size = len(global_pool)
            
            g_cands = global_pool[global_idx : global_idx + n_global_cands]
            global_idx += n_global_cands
            
            # Local candidates around best
            progress = calls_made / budget
            step_size = 2.0 * (1.0 - progress) + 0.5
            l_cands = best_x + step_size * np.random.randn(n_local_cands, n)
            l_cands = np.clip(l_cands, -5, 5)
            
            candidates = np.vstack([g_cands, l_cands])
            
            # 3. Predict Mean and Variance for Candidates using IDW
            X2 = np.sum(X**2, axis=1)
            C2 = np.sum(candidates**2, axis=1)
            D2 = C2[:, None] + X2[None, :] - 2 * candidates @ X.T
            D2 = np.maximum(D2, 0)
            
            D2_safe = D2 + epsilon
            weights = 1.0 / (D2_safe ** alpha)
            
            sum_w = np.sum(weights, axis=1)
            sum_w[sum_w == 0] = 1e-10
            
            weighted_sum = weights @ Y
            mu = weighted_sum / sum_w
            
            sum_w_sq = np.sum(weights**2, axis=1)
            var_estimate = y_var * (sum_w_sq / (sum_w * sum_w))
            std_estimate = np.sqrt(var_estimate)
            
            # 4. Acquisition Function: LCB
            beta = beta_start + (beta_end - beta_start) * progress
            acq = mu - beta * std_estimate
            
            # 5. Select Best Candidate
            best_cand_idx = np.argmin(acq)
            x_new = candidates[best_cand_idx]
            
            # 6. Evaluate
            y_new = func(x_new)
            calls_made += 1
            
            # 7. Update Data
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

        return best_x, best_y
# EVOLVE-BLOCK-END
