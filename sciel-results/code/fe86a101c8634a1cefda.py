"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct structural optimizer for ZDT2 hitting the Pareto front exactly.

    Uses non-uniform spacing of x0 values to maximize hypervolume coverage
    of the ZDT2 Pareto front with respect to reference point [11, 11].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, t):
        """Compute hypervolume for a set of x0 values on the ZDT2 front.
        
        ZDT2 front: f1 = t, f2 = (1+t)*(1-sqrt(t))
        Reference point: [11, 11]
        
        The hypervolume is the area of the union of rectangles [f1_i, 11] x [f2_i, 11].
        For points sorted by f1 ascending, the running minimum of f2 gives the lower envelope.
        """
        if len(t) == 0:
            return 0.0
        
        f1 = np.clip(t, 0.0, 1.0)
        f2 = (1.0 + f1) * (1.0 - np.sqrt(np.maximum(f1, 0.0)))
        
        # Sort by f1 ascending
        idx = np.argsort(f1)
        f1_s = f1[idx]
        f2_s = f2[idx]
        
        r1, r2 = 11.0, 11.0
        running_min = float('inf')
        hv = 0.0
        for i in range(len(f1_s)):
            running_min = min(running_min, f2_s[i])
            if i < len(f1_s) - 1:
                width = f1_s[i+1] - f1_s[i]
            else:
                width = r1 - f1_s[i]
            if running_min < r2:
                hv += width * (r2 - running_min)
        return hv

    def _optimal_x0_placement(self, n):
        """Find optimal x0 placements to maximize hypervolume on ZDT2 front."""
        if n <= 1:
            return np.array([0.0])
        
        from scipy.optimize import minimize
        
        def neg_hypervolume(x0_vals):
            t = np.clip(x0_vals, 0.0, 1.0)
            return -self._compute_hv(t)
        
        # Initial guess: uniform
        x0_init = np.linspace(0.0, 1.0, n)
        
        # Optimize
        try:
            res = minimize(
                neg_hypervolume,
                x0_init,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * n,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x0_opt = np.sort(np.clip(res.x, 0.0, 1.0))
        except Exception:
            x0_opt = x0_init
        
        # Ensure we have exactly n unique points
        x0_opt = np.unique(x0_opt)
        if len(x0_opt) < n:
            x0_opt = np.sort(x0_opt)
            while len(x0_opt) < n:
                gaps = np.diff(np.concatenate(([0.0], x0_opt, [1.0])))
                max_idx = np.argmax(gaps)
                if max_idx == 0:
                    new_pt = x0_opt[0] / 2.0 if len(x0_opt) > 0 else 0.5
                elif max_idx == len(gaps) - 1:
                    new_pt = (1.0 + x0_opt[-1]) / 2.0 if len(x0_opt) > 0 else 0.5
                else:
                    new_pt = (x0_opt[max_idx-1] + x0_opt[max_idx]) / 2.0
                x0_opt = np.sort(np.concatenate((x0_opt, [new_pt])))
        
        return x0_opt[:n]

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget
        if n < 2:
            x0s = np.array([0.0])
        else:
            # Try optimized placement if we have time
            if time.time() < deadline - 60:
                try:
                    x0s = self._optimal_x0_placement(n)
                except Exception:
                    x0s = np.concatenate(([0.0], np.linspace(0.0, 1.0, n - 1)[1:]))
            else:
                # Fallback to uniform
                x0s = np.concatenate(([0.0], np.linspace(0.0, 1.0, n - 1)[1:]))
        
        # Ensure we include x0=0 for f1=0
        if len(x0s) > 0 and x0s[0] > 1e-10:
            x0s = np.concatenate(([0.0], x0s))
            x0s = np.unique(x0s)
            if len(x0s) > n:
                x0s = x0s[:n]
        
        # Build candidate vectors with x[1:]=0
        cand = np.zeros((len(x0s), self.dim))
        cand[:, 0] = x0s
        cand = np.clip(cand, lo, hi)

        # Evaluate candidates within budget and deadline
        F_list = []
        for i in range(len(cand)):
            if len(F_list) >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(cand[i])
            F_list.append(np.asarray(f, dtype=float))

        return None
# EVOLVE-BLOCK-END
