"""Improved Optimizer: IDW Mean + Nearest Neighbor Uncertainty + Multi-Modal Exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """IDW Surrogate with NN-Uncertainty and Multi-Modal Local Search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim

        # Hyperparameters
        alpha = 2.0  # IDW exponent
        epsilon = 1e-6
        # Beta controls the trade-off between exploitation (mu) and exploration (uncertainty)
        # Uncertainty here is distance to nearest neighbor
        beta_start = 1.5
        beta_end = 0.1
        
        # --- Phase 1: Initial Sobol Sample ---
        n_init = 128  # Increased initial sample for better global view
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
        global_pool_unit = sobol_global.random(5000)
        global_pool = -5.0 + 10.0 * global_pool_unit
        global_idx = 0
        pool_size = len(global_pool)

        # Main Loop
        while calls_made < budget:
            # 1. Generate Candidates
            n_global_cands = 30
            n_local_cands = 70  # More local candidates for refinement
            
            # Refill pool if needed
            if global_idx + n_global_cands > pool_size:
                new_pool_unit = sobol_global.random(5000)
                global_pool = -5.0 + 10.0 * new_pool_unit
                global_idx = 0
                pool_size = len(global_pool)
            
            g_cands = global_pool[global_idx : global_idx + n_global_cands]
            global_idx += n_global_cands
            
            # Local candidates: Perturb the top 3 best points to handle multi-modality
            progress = calls_made / budget
            # Adaptive step size: larger early, smaller late
            base_step = 1.5 * (1.0 - progress) + 0.2
            
            l_cands_list = []
            # Get indices of top 3 best points
            top_k = min(3, len(Y))
            top_indices = np.argsort(Y)[:top_k]
            
            for idx in top_indices:
                center = X[idx]
                # Generate local perturbations
                n_per_center = n_local_cands // top_k
                noise = np.random.randn(n_per_center, n) * base_step
                l_cands_part = center + noise
                l_cands_part = np.clip(l_cands_part, -5, 5)
                l_cands_list.append(l_cands_part)
            
            l_cands = np.vstack(l_cands_list)
            
            candidates = np.vstack([g_cands, l_cands])
            
            # 2. Predict Mean using IDW
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
            
            # 3. Estimate Uncertainty using Distance to Nearest Neighbor
            # Find min distance from each candidate to any point in X
            # D2 is (N_cands, N_data). We want min over data axis.
            nn_dist = np.sqrt(np.min(D2, axis=1))
            
            # Normalize uncertainty? 
            # The scale of distance depends on the box size (10).
            # We can use raw distance or normalize by box width.
            # Let's use raw distance, beta will scale it.
            
            # 4. Acquisition Function: LCB
            # We want to minimize mu - beta * uncertainty
            # Higher uncertainty (larger dist) -> lower score -> more likely to be picked
            beta = beta_start + (beta_end - beta_start) * progress
            acq = mu - beta * nn_dist
            
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
