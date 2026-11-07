"""Multi-objective optimizer on ZDT6 using closed-form parametric front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-objective optimizer on ZDT6 using closed-form parametric front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s/330s kill
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        results = []

        def eval_and_record(x):
            nonlocal used
            if used >= budget:
                return
            if time.time() > deadline:
                return
            x = np.clip(x, lo, hi)
            try:
                fx = problem(x)
                fx = np.array(fx, dtype=float)
                if np.all(np.isfinite(fx)) and np.all(fx < 11.0):
                    results.append(fx)
                used += 1
            except Exception:
                used += 1

        # ZDT6 Analysis:
        # f1(x0) = 1 - exp(-4*x0*sin^2(10*pi*x0))
        # f2(x0) = 2*(1 - sqrt(f1(x0))) when x[1:]=0
        #
        # We want to sample the Pareto front efficiently.
        # Let u = f1(x0). Then f2 = 2*(1 - sqrt(u)).
        # The range of f1 for x0 in [0,1] is [0, 1].
        # Specifically, f1(0)=0, f1(1)=0, and it oscillates between 0 and 1.
        # The "Pareto front" in objective space is the curve (f1, f2) for x0 in [0,1].
        # Since f2 is monotonically decreasing in f1, and f1 is not monotonic in x0,
        # the curve loops. However, for hypervolume, we care about the non-dominated set.
        # Actually, any point on the curve is non-dominated by other points on the curve
        # IF we consider the full curve. But wait, if x0a < x0b and f1(x0a) < f1(x0b),
        # then f2(x0a) > f2(x0b). So (f1a, f2a) and (f1b, f2b) are incomparable.
        # Thus the entire curve is the Pareto front.
        #
        # To maximize hypervolume, we should distribute points along the curve.
        # The curve is defined by x0.
        # A uniform distribution in x0 does NOT give a uniform distribution in f1.
        # f1 changes slowly near the zeros of sin^2 and rapidly near the peaks.
        # 
        # Strategy: Sample u = f1 uniformly in [0, 1].
        # For each u, we need to find x0 such that f1(x0) = u.
        # Since f1(x0) = 1 - exp(-4*x0*sin^2(10*pi*x0)),
        # let v = -ln(1-u)/4. Then x0*sin^2(10*pi*x0) = v.
        # This equation has multiple solutions for x0 in [0,1] for most v.
        # The function h(x0) = x0*sin^2(10*pi*x0) oscillates.
        # We can solve for x0 using Newton's method or bisection for each u.
        # However, doing this for 200 points might be slow if not optimized.
        #
        # Alternative: Pre-compute a dense mapping from x0 to (f1, f2) and then
        # select points based on a metric that maximizes HV.
        # But the prompt suggests "closed-form parametric representation... bypassing root-finding".
        # Is there a closed form for x0 given u?
        # x0 sin^2(10 pi x0) = v. No simple closed form.
        #
        # However, we can parameterize by the "phase" of the sine wave.
        # The function h(x0) has 10 half-cycles in [0,1].
        # Let's just use a dense grid of x0 and select points that are well-spread
        # in the (f1, f2) space, which is equivalent to spreading in the curve's parameter.
        #
        # A more efficient approach:
        # 1. Generate a very dense set of x0 values (e.g., 10000).
        # 2. Compute f1, f2 for all.
        # 3. Sort by f1 (or arc length).
        # 4. Select every k-th point to get ~200 points.
        # 5. Evaluate these specific x0 values.
        #
        # To make it "closed-form" feeling, we can define the sampling directly on the
        # objective space if we can invert. Since we can't invert easily, we stick to
        # dense x0 sampling but optimize the selection.
        #
        # Let's try a different angle: The "inspiration" mentioned sampling u uniformly.
        # If we sample u uniformly, we get points (u, 2*(1-sqrt(u))).
        # We need to find x0 for each u.
        # We can pre-compute a lookup table for x0(u) by inverting the dense x0->u mapping.
        # This is effectively root-finding but done once for a grid.
        
        # Let's do the dense x0 grid approach but very efficiently.
        n_dense = 10000
        x0_grid = np.linspace(0.0, 1.0, n_dense)
        
        # Compute f1, f2
        sin_sq = np.sin(10.0 * math.pi * x0_grid) ** 2
        f1_grid = 1.0 - np.exp(-4.0 * x0_grid * sin_sq)
        f2_grid = 2.0 * (1.0 - np.sqrt(f1_grid))
        
        # We want to select ~budget points from this grid that maximize hypervolume.
        # Since the front is a 1D curve, and HV is maximized by spreading points,
        # a uniform selection in the parameter that corresponds to "arc length" or
        # "objective space distance" is good.
        #
        # Let's compute the arc length in objective space.
        df1 = np.diff(f1_grid)
        df2 = np.diff(f2_grid)
        ds = np.sqrt(df1**2 + df2**2)
        arc_len = np.concatenate([[0.0], np.cumsum(ds)])
        
        # Select points uniformly in arc length
        n_select = budget
        # Ensure we don't pick more than available
        n_select = min(n_select, n_dense)
        
        # Target arc lengths
        target_arcs = np.linspace(0.0, arc_len[-1], n_select)
        
        # Map target arcs to indices in the grid
        # np.searchsorted finds the index where target would be inserted
        # We want the grid point closest to the target arc.
        idxs = np.searchsorted(arc_len, target_arcs)
        idxs = np.clip(idxs, 0, n_dense - 1)
        
        # Remove duplicates (if target arcs map to same index)
        # Since arc_len is strictly increasing (mostly), duplicates are rare,
        # but can happen if ds=0.
        unique_idxs = []
        last_idx = -1
        for idx in idxs:
            if idx != last_idx:
                unique_idxs.append(idx)
                last_idx = idx
        
        # If we lost some due to duplicates, add more from the end or randomly?
        # Just take the first n_select unique ones.
        unique_idxs = unique_idxs[:n_select]
        
        # If we still have fewer than n_select, fill with random grid points
        if len(unique_idxs) < n_select:
            remaining = [i for i in range(n_dense) if i not in set(unique_idxs)]
            np.random.shuffle(remaining)
            for i in remaining:
                if len(unique_idxs) >= n_select:
                    break
                unique_idxs.append(i)
        
        # Evaluate the selected x0 values
        for idx in unique_idxs:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            x0_val = float(x0_grid[idx])
            x = np.zeros(dim)
            x[0] = x0_val
            eval_and_record(x)
            
        # If budget remains (shouldn't happen), use random
        while used < budget:
            if time.time() > deadline:
                break
            x = np.random.uniform(lo, hi)
            eval_and_record(x)

        return results
# EVOLVE-BLOCK-END
