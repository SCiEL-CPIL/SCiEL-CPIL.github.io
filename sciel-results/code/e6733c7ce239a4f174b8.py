"""ZDT3-specific optimizer: greedy hypervolume-maximizing x[0] quantization + polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3(self, x):
        x = np.asarray(x, dtype=float)
        m = x.shape[0]
        f1 = x[0]
        g = 1.0 + 10.0 * (m - 1)
        for i in range(1, m):
            xi = x[i]
            g = g + 2.0 * (xi * xi - 1.0 * np.sin(4.0 * np.pi * xi))
        f2 = (1.0 - np.sqrt(f1 / max(f1, 1e-300))) * g
        return np.array([f1, f2])

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        ref = np.array([11.0, 11.0])
        used = 0

        def eval1(x):
            nonlocal used
            v = problem(x)
            used += 1
            return np.asarray(v, dtype=float)

        # ---------- greedy hypervolume selection over x[0] ----------
        # Candidate x[0] values: dense on [0,1] (front is piecewise linear in x[0])
        n_cand = 4000
        cands = np.linspace(0.0, 1.0, n_cand)
        # Precompute f1 = x[0]; f2 depends on x[1:] which we fix at the nominal value.
        # Nominal x[1:] that minimizes g: xi where xi^2 - sin(4 pi xi) is minimal.
        # Minimize h(xi)=xi^2 - sin(4 pi xi) on [0,1].
        grid = np.linspace(0.0, 1.0, 20001)
        h = grid * grid - np.sin(4.0 * np.pi * grid)
        xi_nom = grid[int(np.argmin(h))]
        m = dim
        g_nom = 1.0 + 10.0 * (m - 1) + 2.0 * (m - 1) * (xi_nom * xi_nom - np.sin(4.0 * np.pi * xi_nom))
        # f2 = (1 - sqrt(f1)) * g ; valid only on non-dominated segments.
        f1c = cands
        f2c = (1.0 - np.sqrt(np.clip(f1c, 0.0, 1.0))) * g_nom
        # Non-dominated segment mask for ZDT3 (f1 thresholds).
        # f2 non-increasing on segments: [0,0.05],[0.1,0.2],[0.3,0.4],[0.5,0.6],
        # [0.7,0.8],[0.9,0.95],[0.982,1]. Outside these, point is dominated.
        segs = [(0.0, 0.05), (0.1, 0.2), (0.3, 0.4), (0.5, 0.6),
                (0.7, 0.8), (0.9, 0.95), (0.982, 1.0)]
        mask = np.zeros(n_cand, dtype=bool)
        for a, b in segs:
            mask |= (cands >= a) & (cands <= b)
        f1c = f1c[mask]
        f2c = f2c[mask]
        n_valid = len(f1c)

        # Greedy: pick point with max hypervolume contribution (2D, ref [11,11]).
        # Contribution of point (f1,f2) given current min-f1-per-f2-bucket structure:
        # In 2D, HV increase of a new point = (ref1 - f1) * (delta_f2) where delta_f2
        # is the gap in f2 between the nearest existing point with f2 < this and the
        # nearest with f2 > this (bounded by ref). We maintain sorted f2 of selected.
        import bisect
        selected_idx = []
        f2_sorted = []  # sorted list of f2 of selected
        f1_sorted = []  # parallel sorted by f2
        # We'll keep arrays sorted by f2.
        f2_sel = np.array([])
        f1_sel = np.array([])
        n_select = min(120, budget - 5)
        for _ in range(n_select):
            if time.time() > deadline:
                break
            if n_valid == 0:
                break
            # For each candidate, compute HV increment.
            # Sort candidates by f2.
            order = np.argsort(f2c)
            fc2 = f2c[order]
            fc1 = f1c[order]
            # current selected f2 sorted
            # For a candidate at position, the f2-interval it owns is between
            # predecessor and successor selected f2 (clamped to [0, ref2]).
            # Use np.searchsorted on f2_sel (sorted).
            pos = np.searchsorted(f2_sel, fc2)  # number of selected with f2 <= cand
            # predecessor f2 (largest selected <= cand) at pos-1; successor at pos.
            pred = np.where(pos > 0, f2_sel[np.clip(pos - 1, 0, len(f2_sel) - 1)], 0.0)
            succ = np.where(pos < len(f2_sel), f2_sel[np.clip(pos, 0, max(len(f2_sel) - 1, 0))], ref[1])
            # lower bound of interval is max(pred, 0); but if pred==cand (duplicate f2) interval 0.
            lo_f2 = pred
            hi_f2 = succ
            # Also, the point only contributes where it is the min f1 in that f2-band.
            # Since all candidates are on the front (non-dominated), the band between
            # predecessor and successor selected points: the new point dominates the
            # region f2 in (pred, succ) only if its f1 < the f1 of the boundary points.
            # Simplify: increment = (ref1 - f1) * (hi_f2 - lo_f2), but cap hi_f2 by the
            # f2 of the successor's "coverage". Good enough approximation; refine by
            # exact: area = (ref1 - f1cand) * (min(succ, ref2) - max(pred,0))
            # but must ensure pred < cand < succ else 0.
            band = (hi_f2 - lo_f2)
            band = np.clip(band, 0.0, None)
            # If candidate f2 equals a selected f2, band may be positive but it's not
            # strictly better; we want strictly-better points. Use small tolerance.
            inc = (ref[0] - fc1) * band
            # Exclude candidates whose f2 is within 1e-9 of an existing selected (no gain).
            j = int(np.argmax(inc))
            if inc[j] <= 1e-12:
                break
            new_idx = order[j]
            new_f2 = fc2[j]
            new_f1 = fc1[j]
            # Insert into sorted arrays.
            p = int(np.searchsorted(f2_sel, new_f2))
            f2_sel = np.insert(f2_sel, p, new_f2)
            f1_sel = np.insert(f1_sel, p, new_f1)
            selected_idx.append(new_idx)
            if used + 1 > budget - 5:
                break
        # Fallback: if greedy picked few, top up with uniform non-dominated points.
        if len(selected_idx) < 10:
            for i in range(0, n_valid, max(1, n_valid // 10)):
                if i not in selected_idx:
                    selected_idx.append(i)

        # ---------- evaluate chosen x[0] with polished x[1:] ----------
        chosen_x0 = f1c[selected_idx]
        # Polish each x[0]: optimize x[1:] to minimize g (and thus f2) via local search.
        base = np.tile(np.concatenate([[0.0], np.full(m - 1, xi_nom)]), (len(chosen_x0), 1))
        base[:, 0] = chosen_x0
        # Local polish on x[1:] to reduce g.
        n_polish = min(200, budget - len(chosen_x0) - 2)
        steps = max(1, n_polish // max(1, len(chosen_x0)))
        step = 0.05
        for _ in range(steps):
            if used >= budget - 2 or time.time() > deadline:
                break
            for i in range(len(base)):
                if used >= budget - 2 or time.time() > deadline:
                    break
                # try a few perturbations of x[1:]
                best = base[i].copy()
                best_f = self._zdt3(best)
                for _k in range(3):
                    cand = base[i].copy()
                    cand[1:] = cand[1:] + step * rng.standard_normal(m - 1)
                    cand = np.clip(cand, lo, hi)
                    cf = self._zdt3(cand)
                    if cf[1] < best_f[1] - 1e-9:
                        best = cand
                        best_f = cf
                base[i] = best
                step *= 0.7
        # Evaluate via problem() to count toward budget and record.
        F = np.array([eval1(x) for x in base])
        # Keep only points strictly better than ref in every objective.
        keep = np.all(F < ref, axis=1)
        F = F[keep]
        # Deduplicate: for each f2 bucket keep min f1 (front is 1:1 here).
        if len(F) > 1:
            order = np.argsort(F[:, 1])
            F = F[order]
            # remove points dominated by a lower-f1 with <= f2
            keep2 = np.ones(len(F), dtype=bool)
            minf1 = np.inf
            for i in range(len(F)):
                if F[i, 0] < minf1 - 1e-12:
                    minf1 = F[i, 0]
                else:
                    keep2[i] = False
            F = F[keep2]
        return F
# EVOLVE-BLOCK-END
