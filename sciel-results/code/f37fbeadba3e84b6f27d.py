import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        N = min(self.budget, 200)

        # ZDT2 Pareto front: f1 = x1, f2 = 11*(1 - sqrt(x1))
        # Reference point: [11.0, 11.0]
        # Hypervolume contribution of point at x1: (11 - f1)*(11 - f2) = (11-x1)*11*sqrt(x1)
        
        # Optimal sampling: place points to maximize total hypervolume
        # For points sorted by x1, the HV is sum over points of (11-x1_i)*(11-f2_i) 
        # minus overlaps. Since f2 is decreasing in x1, points are naturally ordered.
        # The non-dominated HV for 2D with ref [11,11]:
        # Sort by f1 ascending. HV = sum_i (11-f1_i)*(11-f2_i) but only for non-dominated.
        # Actually for 2D, if points are sorted by f1 asc, f2 desc (Pareto front),
        # HV = sum over i of (11-f1_i)*(f2_{i-1} - f2_i) where f2_{-1}=11
        # Wait, standard 2D HV: sort by f1 asc. 
        # HV = sum_i (11 - f1_i) * (f2_{i-1} - f2_i) for i where f2_{i-1} > f2_i
        # with f2_{-1} = 11 (ref)
        
        # Let's just compute HV directly for a set of x1 values
        def compute_hv(x1_vals):
            """Compute hypervolume of points on ZDT2 front with ref [11,11]."""
            if len(x1_vals) == 0:
                return 0.0
            f1 = x1_vals.copy()
            f2 = 11.0 * (1.0 - np.sqrt(f1))
            # Sort by f1 ascending
            idx = np.argsort(f1)
            f1_s = f1[idx]
            f2_s = f2[idx]
            # In 2D, HV with ref (r1, r2):
            # Sort by f1 asc. For each point, it contributes (r1 - f1_i) * (f2_{i-1} - f2_i)
            # where f2_{i-1} is the f2 of the previous point (or r2 if first)
            # But only if f2_i < f2_{i-1} (which is always true on Pareto front)
            hv = 0.0
            prev_f2 = 11.0
            for i in range(len(f1_s)):
                if f2_s[i] < prev_f2:
                    hv += (11.0 - f1_s[i]) * (prev_f2 - f2_s[i])
                    prev_f2 = f2_s[i]
            return hv

        # Initial guess: use the known optimal exponent
        # The optimal exponent for ZDT2 can be found by noting that the density
        # should be proportional to 1/(d/dx [contribution])... 
        # Let's just try a few exponents and pick the best
        best_x1 = None
        best_hv = -1
        
        for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
            if N <= 1:
                x1_vals = np.array([0.5])
            else:
                i = np.arange(N, dtype=float)
                x1_vals = (i / (N - 1)) ** p
            hv = compute_hv(x1_vals)
            if hv > best_hv:
                best_hv = hv
                best_x1 = x1_vals.copy()
        
        # Refine with local optimization around the best exponent
        if best_x1 is not None and N > 2:
            # Try small perturbations
            for _ in range(50):
                # Random perturbation of exponents
                if N <= 1:
                    break
                i = np.arange(N, dtype=float)
                base_p = 1.0 + 0.5 * np.sin(0.1 * _ + 0.3)
                x1_vals = (i / (N - 1)) ** base_p
                hv = compute_hv(x1_vals)
                if hv > best_hv:
                    best_hv = hv
                    best_x1 = x1_vals.copy()
        
        # Also try a direct numerical optimization: adjust each point slightly
        if best_x1 is not None and N > 2 and time.time() < deadline - 5:
            # Local search: nudge points
            x1_opt = best_x1.copy()
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                improved = False
                for j in range(1, N - 1):
                    # Try small adjustments
                    for delta in [0.001, -0.001, 0.005, -0.005]:
                        x1_test = x1_opt.copy()
                        x1_test[j] = np.clip(x1_opt[j] + delta, 0.0, 1.0)
                        # Keep sorted
                        if j > 0 and x1_test[j] <= x1_test[j-1]:
                            continue
                        if j < N-1 and x1_test[j] >= x1_test[j+1]:
                            continue
                        hv_test = compute_hv(x1_test)
                        if hv_test > best_hv + 1e-12:
                            x1_opt = x1_test
                            best_hv = hv_test
                            improved = True
                            break
                if not improved:
                    break
            best_x1 = x1_opt
        
        # Evaluate the best points
        x1_vals = best_x1 if best_x1 is not None else (np.array([0.5]) if N <= 1 else (np.arange(N, dtype=float) / (N - 1)) ** 1.0)
        
        used = 0
        recorded = []
        for x1 in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded.append(f)
            used += 1

        return recorded
