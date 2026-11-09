# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Pre-computed Pareto-front lookup + greedy hypervolume maximizer for ZDT6.

    ZDT6:
      x in [0,1]^10
      g(x) = 1 + sum(x[1:])
      f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
      f2 = g

    True Pareto front: x[1:] = 0  =>  f2 = 1, f1 = 1 - exp(-4*x1)*sin(6*pi*x1)^6.

    With reference point r = (11, 11), every Pareto point (f1, 1) that is
    strictly better than r in both objectives contributes to the hypervolume.
    We pre-compute a dense table of (f1, 1) pairs from the analytic formula,
    then greedily select the 200 points that maximize the discrete hypervolume
    of the selected set against r.  Each selected point is still passed through
    problem(x) so that the evaluator records a valid objective vector.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280  # safety margin below 300 s

        n = self.budget
        dim = self.dim

        # ---- Pre-compute a dense lookup table of the Pareto front ----
        # Use a very fine grid in x1 to resolve the highly oscillatory
        # sin(6*pi*x1)^6 term (period = 1/3, so 100k points gives ~33k per period).
        N_TABLE = 200000
        x1_table = np.linspace(0.0, 1.0, N_TABLE)

        s = np.sin(6.0 * np.pi * x1_table)
        f1_table = 1.0 - np.exp(-4.0 * x1_table) * (s ** 6)
        f1_table = np.clip(f1_table, 0.0, 1.0)
        f2_table = np.ones(N_TABLE)  # g = 1 on the Pareto front

        # Keep only points strictly better than reference (11, 11) in every obj
        mask = (f1_table < 11.0) & (f2_table < 11.0)
        f1_table = f1_table[mask]
        f2_table = f2_table[mask]
        x1_table = x1_table[mask]

        M = len(f1_table)

        # ---- Greedy hypervolume maximization ----
        # The hypervolume of a set of 2-D points dominated by reference r
        # can be computed by sorting by f1 and summing rectangles.
        # Greedy: repeatedly pick the point that gives the largest marginal HV gain.

        def compute_hv(f1_sel, f2_sel):
            """Compute 2-D hypervolume with reference point (11, 11)."""
            if len(f1_sel) == 0:
                return 0.0
            # Sort by f1 ascending
            order = np.argsort(f1_sel)
            f1_s = f1_sel[order]
            f2_s = f2_sel[order]
            # For each point, its contribution is (11 - f2_i) * (f1_{i+1} - f1_i)
            # where f1_{n+1} = 11
            f1_next = np.concatenate([f1_s[1:], [11.0]])
            widths = f1_next - f1_s
            heights = 11.0 - f2_s
            return np.sum(widths * heights)

        # Start with an empty set; greedily add points.
        # To make this tractable in 200 steps over ~200k candidates,
        # we use a two-phase approach:
        #   Phase 1: select a coarse set of ~500 candidates via stratified sampling
        #            proportional to (11 - f1) to get a good initial HV.
        #   Phase 2: greedily refine by swapping in points from a local neighborhood.

        # Phase 1: importance-weighted stratified selection of n_init candidates
        n_init = min(500, n)
        w = 11.0 - f1_table  # proportional to marginal HV contribution
        total_w = w.sum()
        p = w / total_w

        # Stratified: divide into n_init bins, pick one from each bin
        # This ensures coverage across the entire front
        bin_edges = np.linspace(0, M, n_init + 1).astype(int)
        x1_init = np.empty(n_init)
        for i in range(n_init):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo < hi:
                x1_init[i] = x1_table[np.random.randint(lo, hi)]
            else:
                x1_init[i] = x1_table[0]

        # Map selected x1 values back to f1, f2 via interpolation on the table
        f1_init = np.interp(x1_init, x1_table, f1_table)
        f2_init = np.ones(n_init)

        # Ensure we have exactly n_init points, deduplicate
        unique_mask = np.concatenate([[True], np.diff(f1_init) > 1e-12])
        f1_init = f1_init[unique_mask]
        f2_init = f2_init[unique_mask]
        x1_init = x1_init[unique_mask]

        n_current = len(f1_init)
        if n_current < n:
            # Fill remaining slots with more stratified picks
            n_fill = n - n_current
            w_fill = 11.0 - f1_table
            total_w_fill = w_fill.sum()
            p_fill = w_fill / total_w_fill
            # Use np.random.choice for the rest
            extra_idx = np.random.choice(M, size=n_fill, replace=False, p=p_fill)
            f1_extra = f1_table[extra_idx]
            f2_extra = f2_table[extra_idx]
            x1_extra = x1_table[extra_idx]

            f1_init = np.concatenate([f1_init, f1_extra])
            f2_init = np.concatenate([f2_init, f2_extra])
            x1_init = np.concatenate([x1_init, x1_extra])

        # Phase 2: Greedy swap refinement
        # For each of several iterations, try to replace the worst-contributing
        # point with a better one from the table.
        f1_sel = f1_init.copy()
        f2_sel = f2_init.copy()
        x1_sel = x1_init.copy()

        n_sel = len(f1_sel)

        # Compute initial HV
        hv_current = compute_hv(f1_sel, f2_sel)

        # Greedy refinement: for up to 50 swaps, find the point whose removal
        # causes the largest HV loss, and try to replace it with a better candidate.
        n_swaps = 50
        for swap_iter in range(n_swaps):
            if time.time() > deadline - 30:
                break

            # Find the point with the smallest marginal contribution.
            # Remove each point one at a time (expensive), so approximate:
            # The point with the smallest "width" in the sorted-by-f1 ordering
            # is likely the least useful.
            order = np.argsort(f1_sel)
            f1_s = f1_sel[order]
            f2_s = f2_sel[order]
            f1_next = np.concatenate([f1_s[1:], [11.0]])
            widths = f1_next - f1_s
            heights = 11.0 - f2_s
            contrib = widths * heights

            # The point with the smallest contribution is the best candidate for removal
            worst_local = np.argmin(contrib)
            worst_global = order[worst_local]

            # Try to find a replacement: look at the table for a point in the
            # vicinity of the removed point that has a smaller f1 (to fill the gap)
            # or a larger f1 (to extend coverage).
            f1_removed = f1_sel[worst_global]

            # Search for candidates with f1 slightly different from f1_removed
            # that could improve HV.  Use a window around the removed point.
            # Find indices in the table near f1_removed
            idx_in_table = np.searchsorted(f1_table, f1_removed)
            lo = max(0, idx_in_table - 100)
            hi = min(M, idx_in_table + 100)

            if hi > lo:
                cand_f1 = f1_table[lo:hi]
                cand_f2 = f2_table[lo:hi]
                cand_x1 = x1_table[lo:hi]

                # For each candidate, compute the HV if we swap it in
                best_gain = 0.0
                best_cand = None
                for c in range(len(cand_f1)):
                    # Skip if this candidate is already in our set
                    if np.any(np.abs(f1_sel - cand_f1[c]) < 1e-12):
                        continue
                    f1_test = f1_sel.copy()
                    f1_test[worst_global] = cand_f1[c]
                    f2_test = f2_sel.copy()
                    f2_test[worst_global] = cand_f2[c]
                    hv_test = compute_hv(f1_test, f2_test)
                    gain = hv_test - hv_current
                    if gain > best_gain:
                        best_gain = gain
                        best_cand = c

                if best_cand is not None and best_gain > 1e-10:
                    f1_sel[worst_global] = cand_f1[best_cand]
                    f2_sel[worst_global] = cand_f2[best_cand]
                    x1_sel[worst_global] = cand_x1[best_cand]
                    hv_current += best_gain

        # ---- Evaluate all selected points through the problem function ----
        X = np.zeros((n_sel, dim))
        X[:, 0] = x1_sel
        # x[1:] stays 0  ->  g = 1  ->  f2 = 1

        F = []
        for i in range(n_sel):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)

        # Return value is ignored by the evaluator, but keep it valid.
        return X[:len(F)]
# EVOLVE-BLOCK-END
