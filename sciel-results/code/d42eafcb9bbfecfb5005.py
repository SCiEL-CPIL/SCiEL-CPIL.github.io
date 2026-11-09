"""Multi-objective optimizer on RE21: HypE-style HV-contribution selection + directed coordinate refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective optimizer.

    Maintains a non-dominated archive and selects parents by marginal
    hypervolume contribution (HypE). Each selected point is then refined
    with a deterministic coordinate-descent local search, so the 200
    evaluation budget is spent on high-yield, low-overhead refinement of
    the Pareto front rather than broad stochastic exploration.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # explicit safety margin before the 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        # ---- Archive: list of [x, f], kept non-dominated and sorted by f[0] ----
        archive = []
        used = 0

        def time_ok():
            return time.time() < deadline

        def is_valid(f):
            return f[0] < ref[0] and f[1] < ref[1]

        def dominates(a, b):
            # True if a strictly dominates b (a better in at least one, no worse in any)
            return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])

        def add_point(x, f):
            """Add a point to the archive if non-dominated; drop points it dominates."""
            f = np.asarray(f, dtype=np.float64)
            if not is_valid(f):
                return
            x = np.asarray(x, dtype=np.float64)
            for _, af in archive:
                if dominates(af, f):
                    return  # dominated by existing point
            # remove archive points dominated by f
            survivors = [item for item in archive if not dominates(f, item[1])]
            survivors.append([x, f])
            survivors.sort(key=lambda it: it[1][0])
            archive[:] = survivors

        def hv_2d(pts):
            """Exact 2D hypervolume of a list of f-vectors vs ref (pts sorted by f[0])."""
            hv = 0.0
            prev_y = ref[1]
            for f in pts:
                if f[1] >= ref[1]:
                    continue
                width = ref[0] - f[0]
                height = prev_y - f[1]
                if width > 0 and height > 0:
                    hv += width * height
                if f[1] < prev_y:
                    prev_y = f[1]
            return hv

        def marginal_contribs():
            """Marginal hypervolume contribution of each archive point (O(n^2), n small)."""
            n = len(archive)
            if n == 0:
                return np.array([])
            pts = [it[1] for it in archive]
            total = hv_2d(pts)
            contribs = np.zeros(n)
            for i in range(n):
                hv_without = hv_2d(pts[:i] + pts[i + 1:])
                contribs[i] = total - hv_without
            return contribs

        def sample_uniform():
            return lo + span * np.random.random(self.dim)

        def sample_gaussian(parent, sigma):
            child = parent + sigma * span * np.random.standard_normal(self.dim)
            return np.clip(child, lo, hi)

        # ---- Phase 1: stratified (LHS-like) initialization for good spread ----
        n_init = min(24, self.budget // 3)
        for i in range(n_init):
            if used >= self.budget or not time_ok():
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bin_start = i / n_init
                bin_end = (i + 1) / n_init
                x[d] = lo[d] + span[d] * (bin_start + np.random.random() * (bin_end - bin_start))
            f = problem(x)
            used += 1
            add_point(x, f)

        # ---- Phase 2: HypE selection + directed coordinate refinement ----
        sigma = 0.25
        while used < self.budget and time_ok():
            if len(archive) == 0:
                x = sample_uniform()
                f = problem(x)
                used += 1
                add_point(x, f)
                continue

            if len(archive) >= 2:
                contribs = marginal_contribs()
                contribs = np.maximum(contribs, 1e-12)
                probs = contribs / contribs.sum()
                idx = int(np.random.choice(len(archive), p=probs))
            else:
                idx = 0

            parent_x = archive[idx][0]
            parent_f = archive[idx][1]

            # --- Directed coordinate-descent refinement of the selected point ---
            # Try a small set of fractional steps along each coordinate; keep the
            # best non-dominated improvement, then optionally continue.
            cur_x = parent_x.copy()
            cur_f = np.asarray(parent_f, dtype=np.float64)
            improved = True
            max_rounds = 2
            for _round in range(max_rounds):
                if used >= self.budget or not time_ok():
                    break
                best_x = cur_x
                best_f = cur_f
                best_gain = 0.0
                # candidate step fractions (coarse -> fine), scaled to the span
                steps = [0.5, 0.25, 0.125, 0.0625]
                for d in range(self.dim):
                    if used >= self.budget or not time_ok():
                        break
                    if span[d] < 1e-14:
                        continue
                    for s in steps:
                        if used >= self.budget or not time_ok():
                            break
                        for sign in (1.0, -1.0):
                            if used >= self.budget or not time_ok():
                                break
                            cand = cur_x.copy()
                            cand[d] = cur_x[d] + sign * s * span[d]
                            cand[d] = np.clip(cand[d], lo[d], hi[d])
                            if np.array_equal(cand, cur_x):
                                continue
                            cf = np.asarray(problem(cand), dtype=np.float64)
                            used += 1
                            if not is_valid(cf):
                                continue
                            # accept if it improves hypervolume of the local set {cur, cand}
                            # (i.e. cand is not dominated by cur and is not identical)
                            if dominates(cur_f, cf):
                                continue
                            # measure gain: does cand extend the front relative to cur?
                            local_hv_before = hv_2d([cur_f])
                            local_hv_after = hv_2d(sorted([cur_f, cf], key=lambda p: p[0]))
                            gain = local_hv_after - local_hv_before
                            if gain > best_gain + 1e-12:
                                best_gain = gain
                                best_x = cand
                                best_f = cf
                if best_gain <= 1e-12:
                    improved = False
                    break
                cur_x = best_x
                cur_f = best_f
            # commit the refined point
            add_point(cur_x, cur_f)

            # --- occasional global move to escape local basins ---
            if used < self.budget and time_ok():
                r = np.random.random()
                if r < 0.25:
                    x = sample_uniform()
                else:
                    x = sample_gaussian(parent_x, sigma)
                f = problem(x)
                used += 1
                add_point(x, f)
            sigma = max(0.02, sigma * 0.97)

        return archive
# EVOLVE-BLOCK-END
