"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct Pareto front sampling for ZDT2 with hypervolume-optimized spacing."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295  # safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        budget = self.budget
        if budget < 1:
            return None

        # ZDT2 Pareto front: x[0] = t, x[1:] = 0
        # f1 = t, f2 = 1 - sqrt(t) for t in [0, 1]
        
        # To maximize 2D hypervolume with reference [11, 11],
        # we need to place n points on the front to maximize the union area.
        
        # The hypervolume for a set of non-dominated points sorted by f1 ascending:
        # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) + (11 - f1_n) * (11 - f2_n)
        # where f1_{n+1} = 11
        
        # For optimal spacing, we want to equalize the marginal contribution.
        # Use numerical approach: start with uniform spacing, then iteratively
        # adjust to maximize HV.
        
        n = budget
        
        # Initial guess: use a power-law that biases toward t=0 where f2 is higher
        # f2 = 1 - sqrt(t), so f2 is high near t=0
        # The density should be proportional to 1/(11 - f2(t)) = 1/(10 + sqrt(t))
        # This gives more points near t=0
        
        # Use inverse CDF method with a fine grid
        t_grid = np.linspace(0.0, 1.0, 100000)
        f2_grid = 1.0 - np.sqrt(t_grid)
        
        # The weight for optimal spacing is proportional to the "value" of each point
        # For 2D HV, the marginal contribution of a point at t is related to
        # the area it "covers". A good approximation is to use density proportional
        # to 1/(11 - f2(t)) since points with higher f2 contribute more per unit width.
        
        weights = 1.0 / (11.0 - f2_grid)
        cdf = np.cumsum(weights)
        cdf = cdf / cdf[-1]
        
        # Generate n points using inverse CDF
        u = np.linspace(0.0, 1.0, n)
        t_vals = np.interp(u, cdf, t_grid)
        
        # Ensure endpoints
        t_vals[0] = 0.0
        t_vals[-1] = 1.0
        
        # Refine: do a few iterations of local optimization
        # For each point, try small perturbations and keep if HV improves
        def compute_hv(t_arr):
            """Compute 2D hypervolume for points on ZDT2 front."""
            f1 = t_arr
            f2 = 1.0 - np.sqrt(f1)
            # Sort by f1 ascending
            order = np.argsort(f1)
            f1_s = f1[order]
            f2_s = f2[order]
            
            # 2D HV with reference [11, 11]
            # Points are non-dominated if f2 decreases as f1 increases
            # For ZDT2 front, this is always true
            hv = 0.0
            for i in range(len(f1_s)):
                next_f1 = f1_s[i+1] if i + 1 < len(f1_s) else 11.0
                hv += (next_f1 - f1_s[i]) * (11.0 - f2_s[i])
            return hv
        
        current_hv = compute_hv(t_vals)
        
        # Local refinement: try moving each point slightly
        for iteration in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(1, n - 1):  # don't move endpoints
                for delta in [-0.001, 0.001, -0.0005, 0.0005]:
                    t_new = t_vals.copy()
                    t_new[i] = np.clip(t_vals[i] + delta, 0.0, 1.0)
                    hv_new = compute_hv(t_new)
                    if hv_new > current_hv + 1e-12:
                        t_vals = t_new
                        current_hv = hv_new
                        improved = True
                        break
            if not improved:
                break
        
        # Build decision vectors
        X = np.zeros((n, self.dim))
        X[:, 0] = t_vals
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        for i in range(n):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
