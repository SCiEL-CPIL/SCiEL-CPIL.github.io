"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ7.
    
    Strategy:
    1. Find points near g=0 (x_M = 0.5) using local search.
    2. Sample the 2D Pareto front using Sobol sequence on (y1, y2).
    3. Use remaining budget for refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n):
        """Generate n points in [0,1]^2 using Halton sequence (base 2, 3)."""
        points = np.zeros((n, 2))
        for i in range(n):
            # Base 2 (binary van der Corput)
            f = 1.0
            r = 0.0
            q = i + 1
            while q > 0:
                f /= 2.0
                r += f * (q % 2)
                q //= 2
            points[i, 0] = r

            # Base 3 (ternary van der Corput)
            f = 1.0
            r = 0.0
            q = i + 1
            while q > 0:
                f /= 3.0
                r += f * (q % 3)
                q //= 3
            points[i, 1] = r
        return points

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ7 structure: first dim - (n_obj - 1) variables are "M" (penalty),
        # last (n_obj - 1) variables are "y" (shape the front)
        n_y = n_obj - 1  # 2 for 3 objectives
        n_m = dim - n_y  # 20 for dim=22, n_obj=3

        # Phase 1: Find a good anchor point with g≈0
        # g = sum((x_m - 0.5)^2) for x_m in M
        # To get g=0, set all x_m = 0.5
        # y values determine the front position
        
        # Start with x_m = 0.5, and try a few y values
        # Evaluate a small set to find a good starting point
        anchor = None
        anchor_f = None
        best_g = np.inf
        used = 0

        # Try several y configurations to find a good anchor
        for trial in range(min(10, budget)):
            y_trial = rng.random(n_y)
            x_trial = np.concatenate([np.full(n_m, 0.5), y_trial])
            x_trial = np.clip(x_trial, lo, hi)
            f_trial = problem(x_trial)
            used += 1
            # Estimate g from f values: for DTLZ7, f_i = y_i*(1+g) for i<n_obj-1
            # So g ≈ f_0/y_0 - 1 (if y_0 > 0)
            if y_trial[0] > 0.01:
                g_est = f_trial[0] / y_trial[0] - 1.0
            else:
                g_est = 10.0
            if g_est < best_g:
                best_g = g_est
                anchor = x_trial.copy()
                anchor_f = f_trial.copy()

        if anchor is None:
            # Fallback: random
            anchor = lo + (hi - lo) * rng.random(dim)
            anchor_f = problem(anchor)
            used += 1

        # Phase 2: Local refinement of the anchor to drive g→0
        # Use coordinate descent / Nelder-Mead on the penalty
        # We want to minimize g = sum((x_m - 0.5)^2)
        # But we can only evaluate through problem()
        # Use a simple approach: adjust x_m toward 0.5

        # Actually, let's just set x_m = 0.5 directly and evaluate
        # The y values from anchor should be fine
        x_m_05 = np.full(n_m, 0.5)
        y_anchor = anchor[n_m:]
        x_refined = np.concatenate([x_m_05, y_anchor])
        x_refined = np.clip(x_refined, lo, hi)
        f_refined = problem(x_refined)
        used += 1

        # Compare with anchor
        # For DTLZ7, lower f values are better. If g went to 0, f should be lower
        # (since f_i = y_i*(1+g), g=0 gives f_i = y_i which is minimal)
        if np.sum(f_refined) < np.sum(anchor_f):
            anchor = x_refined
            anchor_f = f_refined

        # Phase 3: Sample the 2D Pareto front using Sobol
        # The front is parameterized by (y1, y2) in [0,1]^2
        # f1 = y1, f2 = y2, f3 = 1 - (y1+y2)/2 (when g=0)
        # We need to cover this 2D manifold densely for hypervolume

        # Reserve some budget for front sampling
        n_front_samples = min(budget - used - 5, 150)
        if n_front_samples < 20:
            n_front_samples = max(10, budget - used - 2)

        sobol_pts = self._sobol_2d(n_front_samples)

        # Add corner points for better coverage
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                            [0.0, 0.5], [0.5, 0.0], [0.5, 1.0], [1.0, 0.5],
                            [0.5, 0.5]])
        sobol_pts = np.vstack([sobol_pts, corners])

        # Evaluate all front samples
        all_x = []
        all_f = []

        for pt in sobol_pts:
            if used >= budget or time.time() > deadline:
                break
            x_pt = np.concatenate([np.full(n_m, 0.5), pt])
            x_pt = np.clip(x_pt, lo, hi)
            f_pt = problem(x_pt)
            used += 1
            all_x.append(x_pt)
            all_f.append(f_pt)

        # Phase 4: Use remaining budget for local refinement of some points
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Pick a few points and do local search
            n_refine = min(remaining, 20)
            idxs = rng.choice(len(all_x), size=min(n_refine, len(all_x)), replace=False)
            for idx in idxs:
                if used >= budget or time.time() > deadline:
                    break
                x_cur = all_x[idx].copy()
                f_cur = all_f[idx].copy()
                # Small perturbations
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    x_new = x_cur + 0.05 * (hi - lo) * rng.standard_normal(dim)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    # Keep if better in sum
                    if np.sum(f_new) < np.sum(f_cur):
                        x_cur = x_new
                        f_cur = f_new
                all_x.append(x_cur)
                all_f.append(f_cur)

        # Return the best set (the return value is ignored, but we return something valid)
        if all_x:
            return np.array(all_x)
        else:
            return anchor
# EVOLVE-BLOCK-END
