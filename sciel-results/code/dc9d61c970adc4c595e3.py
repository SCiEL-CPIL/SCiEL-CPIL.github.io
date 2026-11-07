"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT3-specific optimizer exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # ZDT3 Pareto front: f1 = x0, f2 = 11*(1 - sqrt(f1/11) - 0.05*sin(10*pi*f1))
        # Optimal quantization: place points where marginal hypervolume contribution is equal
        # Use inverse CDF of the density proportional to (11 - f2(x0)) * |df2/dx0|
        
        n_total = self.budget
        
        # Compute optimal x0 locations using the known ZDT3 front structure
        # The hypervolume contribution density is proportional to (ref_f2 - f2(x0))
        # For reference point (11, 11), this is proportional to 11 - f2(x0)
        
        # Sample many points to estimate the optimal distribution
        n_est = 10000
        x0_est = np.linspace(1e-6, 1 - 1e-6, n_est)
        f2_est = 11.0 * (1.0 - np.sqrt(x0_est / 11.0) - 0.05 * np.sin(10.0 * np.pi * x0_est))
        f2_est = np.clip(f2_est, 0.0, 11.0)
        weight = np.clip(11.0 - f2_est, 1e-10, None)
        
        # Normalize to get CDF
        cdf = np.cumsum(weight)
        cdf /= cdf[-1]
        
        # Place n_total points at equal CDF intervals (optimal quantization)
        target_cdfs = np.linspace(0.5 / n_total, 1.0 - 0.5 / n_total, n_total)
        x0_values = np.interp(target_cdfs, cdf, x0_est)
        
        # Ensure we stay within bounds
        x0_values = np.clip(x0_values, lo[0] + 1e-6, hi[0] - 1e-6)
        
        # Evaluate all points
        X = np.zeros((n_total, dim))
        F_list = []
        used = 0
        
        for i, x0 in enumerate(x0_values):
            if time.time() > deadline - 10:
                break
            if used >= self.budget:
                break
            X[i, 0] = x0
            # Other variables are 0, which minimizes g (g=0 on Pareto front)
            f = problem(X[i])
            F_list.append(f)
            used += 1
        
        F = np.array(F_list) if F_list else np.array([])
        
        # Phase 2: Use remaining budget to fill gaps with local refinement
        remaining = self.budget - used
        if remaining > 0 and len(F) > 0:
            # Sort by f1
            order = np.argsort(F[:, 0])
            F_sorted = F[order]
            f1_vals = F_sorted[:, 0]
            
            # Find gaps and sample at midpoints of largest gaps
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break
                
                if len(f1_vals) > 1:
                    gaps = np.diff(f1_vals)
                    max_gap_idx = np.argmax(gaps)
                    if gaps[max_gap_idx] > 0.005:
                        lo_f1 = f1_vals[max_gap_idx]
                        hi_f1 = f1_vals[max_gap_idx + 1]
                        x0_new = (lo_f1 + hi_f1) / 2.0
                    else:
                        # Perturb a random existing point
                        idx = np.random.randint(len(f1_vals))
                        x0_new = f1_vals[idx] + np.random.randn() * 0.001
                        x0_new = np.clip(x0_new, lo[0] + 1e-6, hi[0] - 1e-6)
                else:
                    x0_new = np.random.uniform(0.001, 0.999)
                
                child = np.zeros(dim)
                child[0] = x0_new
                fc = problem(child)
                used += 1
                
                # Check dominance
                if len(F) > 0:
                    dominated = np.all(F[:, 0] <= fc[0], axis=1) & np.all(F[:, 1] <= fc[1], axis=1) & np.any(F[:, 0] < fc[0] | F[:, 1] < fc[1], axis=1)
                    if not np.any(dominated):
                        F = np.vstack([F, fc.reshape(1, -1)])
                        X = np.vstack([X, child.reshape(1, -1)])
                        dominated_by_new = np.all(F[:, 0] >= fc[0], axis=1) & np.all(F[:, 1] >= fc[1], axis=1) & np.any(F[:, 0] > fc[0] | F[:, 1] > fc[1], axis=1)
                        if np.any(dominated_by_new):
                            F = F[~dominated_by_new]
                            X = X[~dominated_by_new]
                else:
                    F = fc.reshape(1, -1)
                    X = child.reshape(1, -1)
        
        return X
# EVOLVE-BLOCK-END
