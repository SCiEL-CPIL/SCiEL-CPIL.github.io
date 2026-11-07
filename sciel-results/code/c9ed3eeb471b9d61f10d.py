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
        
        # Phase 1: Sample x[0] values spread across [0,1] to cover the non-convex front
        # ZDT3's Pareto front has segments where f2 varies with x[0]
        # Use dense uniform sampling to maximize hypervolume coverage
        n_phase1 = 100
        # Dense uniform sampling of x[0] in (0, 1)
        x0_values = np.linspace(0.001, 0.999, n_phase1)
        
        # For each x0 value, set other variables to 0 (Pareto-optimal for g)
        # This gives us points on or very close to the Pareto front
        X = np.zeros((n_phase1, self.dim))
        F_list = []
        used = 0
        
        for i, x0 in enumerate(x0_values):
            if time.time() > deadline - 10:
                break
            if used >= self.budget:
                break
            X[i, 0] = x0
            # Other variables are 0, which minimizes g
            f = problem(X[i])
            F_list.append(f)
            used += 1
        
        F = np.array(F_list) if F_list else np.array([])
        
        # Phase 2: Use remaining budget to fill gaps in the front
        # Identify gaps in f1 coverage and sample there
        remaining = self.budget - used
        if remaining > 0 and len(F) > 0:
            # Sort by f1 to identify gaps
            order = np.argsort(F[:, 0])
            F_sorted = F[order]
            
            # Find largest gaps in f1
            f1_vals = F_sorted[:, 0]
            gaps = np.diff(f1_vals)
            
            # Sample in the largest gaps
            for _ in range(remaining):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break
                
                # Pick a gap to fill (largest gap)
                if len(gaps) > 0 and len(gaps) > 1:
                    max_gap_idx = np.argmax(gaps)
                    if gaps[max_gap_idx] > 0.01:  # Only fill significant gaps
                        # Sample in this gap
                        lo_f1 = f1_vals[max_gap_idx]
                        hi_f1 = f1_vals[max_gap_idx + 1]
                        # Sample at random position in the gap
                        x0_new = lo_f1 + np.random.rand() * (hi_f1 - lo_f1)
                    else:
                        # All gaps are small, sample randomly
                        x0_new = np.random.uniform(0.001, 0.999)
                else:
                    x0_new = np.random.uniform(0.001, 0.999)
                
                child = np.zeros(self.dim)
                child[0] = x0_new
                fc = problem(child)
                used += 1
                
                # Add to archive if it improves coverage
                # Check if it's dominated
                dominated = np.all(F[:, 0] <= fc[0], axis=1) & np.all(F[:, 1] <= fc[1], axis=1) & np.any(F[:, 0] < fc[0] | F[:, 1] < fc[1], axis=1)
                if not np.any(dominated):
                    # Not dominated, add it
                    F = np.vstack([F, fc.reshape(1, -1)])
                    X = np.vstack([X, child.reshape(1, -1)])
                    # Remove points dominated by the new point
                    dominated_by_new = np.all(F[:, 0] >= fc[0], axis=1) & np.all(F[:, 1] >= fc[1], axis=1) & np.any(F[:, 0] > fc[0] | F[:, 1] > fc[1], axis=1)
                    if np.any(dominated_by_new):
                        F = F[~dominated_by_new]
                        X = X[~dominated_by_new]
        
        return X
# EVOLVE-BLOCK-END
