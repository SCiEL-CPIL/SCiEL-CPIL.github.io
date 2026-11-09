"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Differential Evolution with Binomial Crossover."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s kill, stop at 320s
        deadline = 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # 1. Initialization: Latin Hypercube Sampling for good initial coverage
        # Use a significant portion of the budget for initialization to ensure diversity
        n_init = min(50, budget)
        
        # Generate LHS points
        # For each dimension, permute the indices and assign normalized values
        X_init = np.zeros((n_init, dim))
        for j in range(dim):
            idx = np.random.permutation(n_init)
            X_init[idx, j] = (np.arange(n_init) + 0.5) / n_init # Center of bins
        X_init = lo + X_init * (hi - lo)

        evaluated_X = []
        evaluated_F = []

        # Evaluate initial population
        for x in X_init:
            f = problem(x)
            evaluated_X.append(x)
            evaluated_F.append(f)
            
        used = n_init
        X = np.array(evaluated_X)
        F = np.array(evaluated_F)

        # Helper function to find non-dominated indices
        # Returns a boolean mask where True indicates non-dominated
        def get_non_dominated_mask(F_arr):
            n = F_arr.shape[0]
            if n == 0:
                return np.array([])
            # Matrix of dominance: D[i,j] is True if i dominates j
            # i dominates j if F[i] <= F[j] and F[i] != F[j]
            # Vectorized check:
            # F[i] <= F[j] for all objectives
            # At least one F[i] < F[j]
            
            # To avoid O(N^2) memory blowup for large N (though N=200 is small),
            # we can do it in a loop or vectorized. N=200 is fine for vectorized.
            
            # F_expanded: (N, 1, M) and F_arr: (1, N, M)
            # This creates a (N, N, M) array. 200*200*3 = 120,000 floats. Safe.
            F_i = F_arr[:, np.newaxis, :]  # (N, 1, M)
            F_j = F_arr[np.newaxis, :, :]  # (1, N, M)
            
            less_eq = np.all(F_i <= F_j, axis=2)  # (N, N)
            less_than = np.any(F_i < F_j, axis=2) # (N, N)
            
            dominates = less_eq & less_than  # (N, N)
            
            # A point i is non-dominated if no other point j dominates it
            # i.e., not any(dominates[j, i] for j != i)
            # Note: dominates[i, i] is False because F[i] < F[i] is false.
            
            is_dominated = np.any(dominates, axis=0) # True if any row j has dominates[j, i] = True
            return ~is_dominated

        # Main Loop
        F_de = 0.8 # Differential Evolution factor
        C_r = 0.5  # Crossover rate
        
        # We keep X and F as the full history to compute hypervolume over all points.
        # However, for parent selection, we only care about non-dominated ones.
        
        while used < budget:
            if time.time() > deadline:
                break

            # Get current non-dominated indices
            nd_mask = get_non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                # Fallback: Random point
                x_new = lo + np.random.rand(dim) * (hi - lo)
            else:
                # Select parents from non-dominated set
                # If we have fewer than 3 non-dominated points, sample with replacement
                if len(nd_indices) >= 3:
                    r1_idx, r2_idx, r3_idx = np.random.choice(nd_indices, 3, replace=False)
                else:
                    r1_idx = np.random.choice(nd_indices)
                    r2_idx = np.random.choice(nd_indices)
                    r3_idx = np.random.choice(nd_indices)
                
                x1 = X[r1_idx]
                x2 = X[r2_idx]
                x3 = X[r3_idx]
                
                # DE Mutation
                v = x1 + F_de * (x2 - x3)
                
                # Binomial Crossover
                # For each dimension, pick from v or x1 with prob C_r
                mask = np.random.rand(dim) < C_r
                # Ensure at least one gene is taken from v (standard DE practice)
                if not np.any(mask):
                    mask[np.random.randint(dim)] = True
                    
                x_new = np.where(mask, v, x1)
                
                # Add small perturbation to escape local minima
                x_new += 0.05 * (hi - lo) * np.random.randn(dim)

            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            used += 1
            
            # Append to history
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])

        return X
# EVOLVE-BLOCK-END
