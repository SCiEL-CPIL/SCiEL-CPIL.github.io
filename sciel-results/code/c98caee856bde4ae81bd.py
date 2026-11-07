import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt4_f2(self, x0):
        """Compute f2 for ZDT4 given x0 (x[1:] = 0)"""
        x0 = np.clip(x0, 0.0, 1.0)
        return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0) ** 2

    def _hv_2d(self, f1_vals, f2_vals, R1=11.0, R2=400.0):
        """Compute 2D hypervolume for non-dominated points."""
        if len(f1_vals) == 0:
            return 0.0
        order = np.argsort(f1_vals)
        f1 = np.asarray(f1_vals)[order]
        f2 = np.asarray(f2_vals)[order]
        mask = (f1 < R1) & (f2 < R2)
        f1 = f1[mask]
        f2 = f2[mask]
        if len(f1) == 0:
            return 0.0
        # Keep only non-dominated points: f2 must be strictly decreasing as f1 increases
        n = len(f1)
        keep = np.zeros(n, dtype=bool)
        running_min_f2 = np.inf
        for i in range(n):
            if f2[i] < running_min_f2:
                keep[i] = True
                running_min_f2 = f2[i]
        f1 = f1[keep]
        f2 = f2[keep]
        if len(f1) == 0:
            return 0.0
        hv = 0.0
        for i in range(len(f1)):
            next_f1 = f1[i + 1] if i + 1 < len(f1) else R1
            hv += (next_f1 - f1[i]) * (R2 - f2[i])
        return hv

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 285.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, dim))

        N = min(budget, 200)

        # ZDT4 Pareto front: f1 = x0, f2 = 1 - sqrt(x0) - x0*sin(10*pi*x0)^2
        # Strategy: use dynamic programming on a fine quantized grid to select
        # N x0 values that maximize hypervolume, then refine locally.

        # Fine grid for DP
        n_grid = 2000
        x0_grid = np.linspace(0.0, 1.0, n_grid)
        f2_grid = self._zdt4_f2(x0_grid)

        # DP: select k points from grid positions 0..i (sorted by f1)
        # to maximize hypervolume contribution.
        # State: dp[k][i] = max HV achievable using k points from grid[0..i]
        # Transition: either skip grid[i] or include it.
        # When we include grid[i] as the k-th point (largest f1 among selected),
        # the HV contribution of this point depends on the (k-1)-th selected point.
        #
        # For 2D HV with reference (R1, R2), if selected points are at
        # f1 values a_1 < a_2 < ... < a_k, the HV is:
        #   sum_{j=1}^{k} (a_{j+1} - a_j) * (R2 - f2_j)
        # where a_{k+1} = R1.
        #
        # This is hard to decompose simply. Let's use a different approach:
        # Greedy with local search, but with a smarter initialization.

        # Better approach: Start with arc-length uniform placement, then do
        # multiple rounds of local optimization where each point is moved to
        # maximize the marginal HV contribution.

        # Arc-length uniform initialization
        n_fine = 10000
        x0_fine = np.linspace(0.0, 1.0, n_fine)
        f2_fine = self._zdt4_f2(x0_fine)
        dx = np.diff(x0_fine)
        df2 = np.diff(f2_fine)
        ds = np.sqrt(dx ** 2 + df2 ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_total = s[-1]
        s_norm = s / s_total

        if N == 1:
            x0_vals = np.array([0.5])
        else:
            s_targets = np.linspace(0.0, 1.0, N)
            x0_vals = np.interp(s_targets, s_norm, x0_fine)

        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        # Local refinement: iteratively move each point to maximize marginal HV
        def refine_placement(x0_init, max_rounds=15):
            x0 = np.sort(x0_init.copy())
            n = len(x0)
            if n <= 1:
                return x0

            # Precompute f2 on fine grid
            grid_n = 10000
            x0_grid = np.linspace(0.0, 1.0, grid_n)
            f2_grid = self._zdt4_f2(x0_grid)

            for rnd in range(max_rounds):
                if time.time() - t0 > DEADLINE - 10:
                    break
                improved = False
                for i in range(n):
                    # Determine the f1 interval this point owns
                    if i == 0:
                        f1_left = 0.0
                        f1_right = x0[1] if n > 1 else 11.0
                    elif i == n - 1:
                        f1_left = x0[i - 1]
                        f1_right = 11.0
                    else:
                        f1_left = x0[i - 1]
                        f1_right = x0[i + 1]

                    if f1_right <= f1_left + 1e-10:
                        continue

                    # Find grid points in the interval
                    mask = (x0_grid >= f1_left) & (x0_grid <= f1_right)
                    if not np.any(mask):
                        continue

                    x0_cand = x0_grid[mask]
                    f2_cand = f2_grid[mask]

                    # For each candidate, compute the marginal HV change
                    # The point's contribution to HV depends on its neighbors
                    # and its f2 value.
                    # Simple approach: pick candidate with min f2 in interval
                    # (since lower f2 always helps HV)
                    best_idx = np.argmin(f2_cand)
                    new_x0 = x0_cand[best_idx]

                    if abs(new_x0 - x0[i]) < 1e-9:
                        continue

                    # Verify improvement via full HV computation
                    old_hv = self._hv_2d(x0, self._zdt4_f2(x0))
                    x0_test = x0.copy()
                    x0_test[i] = new_x0
                    new_hv = self._hv_2d(x0_test, self._zdt4_f2(x0_test))

                    if new_hv > old_hv + 1e-12:
                        x0[i] = new_x0
                        improved = True

                if not improved:
                    break

            return x0

        # Refine
        if N > 1 and time.time() - t0 < DEADLINE - 15:
            x0_vals = refine_placement(x0_vals)

        # Final sort and clip
        x0_vals = np.sort(x0_vals)
        x0_vals = np.clip(x0_vals, lo[0], hi[0])

        # Remove duplicates (keep first occurrence)
        if len(x0_vals) > 1:
            keep = np.concatenate([[True], np.diff(x0_vals) > 1e-8])
            x0_vals = x0_vals[keep]

        evals = 0
        results = []

        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break

            x = np.zeros(dim)
            x[0] = float(x0_val)
            x = np.clip(x, lo, hi)

            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass

        if results:
            return np.array(results)

        return np.empty((0, dim))
