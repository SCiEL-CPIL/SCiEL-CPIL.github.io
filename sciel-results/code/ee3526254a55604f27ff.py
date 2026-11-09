"""Rewritten: Augmented Epsilon-Constraint Method + L-BFGS-B polish for RE21."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin before 330s wall-clock

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        eval_count = 0
        max_evals = self.budget
        archive = []  # list of (x, f)

        def clip(x):
            return np.clip(x, lo, hi)

        def time_ok():
            return time.time() < deadline

        def record(x, f):
            nonlocal eval_count
            archive.append((np.asarray(x, dtype=float), np.asarray(f, dtype=float)))

        # ---- 1. Initial Latin-hypercube sampling ----
        n_init = 45
        n_init = min(n_init, max_evals)
        rng = np.random.RandomState(0)
        for i in range(n_init):
            x = lo + span * (rng.rand(self.dim) + i) / n_init
            x = clip(x)
            f = np.asarray(problem(x), dtype=float)
            record(x, f)
            eval_count += 1
            if not time_ok():
                break

        if eval_count >= max_evals or not time_ok():
            return archive

        # ---- 2. Epsilon-Constraint decomposition ----
        # f1 primary, f2 constrained: f2 <= eps
        def obj_and_constraint(eps):
            def fun(x):
                f = np.asarray(problem(x), dtype=float)
                eval_count += 1
                # augmented epsilon-constraint penalty
                penalty = 0.0
                if f[1] > eps:
                    penalty = 100.0 * (f[1] - eps)
                return f[0] + penalty
            return fun

        # Determine eps range from initial archive
        f2_vals = np.array([f[1] for (_, f) in archive])
        f2_min = float(np.min(f2_vals))
        f2_max = float(np.max(f2_vals))
        f1_vals = np.array([f[0] for (_, f) in archive])
        f1_min = float(np.min(f1_vals))
        f1_max = float(np.max(f1_vals))

        # Sweep eps from f2_max down to f2_min
        n_sweep = 10
        if f2_max > f2_min:
            eps_list = np.linspace(f2_max, f2_min, n_sweep)
        else:
            eps_list = np.array([f2_min])

        for eps in eps_list:
            if eval_count >= max_evals or not time_ok():
                break
            # Start from the archive point with f2 closest to eps (and low f1)
            best_start = None
            best_score = None
            for (x, f) in archive:
                score = abs(f[1] - eps) + 0.01 * f[0]
                if best_score is None or score < best_score:
                    best_score = score
                    best_start = x
            if best_start is None:
                best_start = lo + 0.5 * span
            x0 = clip(best_start)

            fun = obj_and_constraint(eps)
            res = minimize(
                fun, x0, method='L-BFGS-B',
                bounds=list(zip(lo, hi)),
                options={'maxiter': 15, 'gtol': 1e-6}
            )
            if res.x is not None:
                xf = clip(res.x)
                ff = np.asarray(problem(xf), dtype=float)
                eval_count += 1
                record(xf, ff)

        if eval_count >= max_evals or not time_ok():
            return archive

        # ---- 3. Polish: tighter L-BFGS-B on top HV points ----
        # Compute a quick hypervolume contribution ranking (2D)
        ref = np.array([3000.0, 0.06])
        pts = []
        for (x, f) in archive:
            if f[0] < ref[0] and f[1] < ref[1]:
                pts.append(f)
        pts = np.array(pts) if pts else np.array([])

        if len(pts) > 0:
            # Sort by f1; compute contribution
            order = np.argsort(pts[:, 0])
            pts_sorted = pts[order]
            contribs = np.zeros(len(pts_sorted))
            for i in range(len(pts_sorted)):
                f1 = pts_sorted[i, 0]
                f2 = pts_sorted[i, 1]
                # width in f1: from f1 to next f1 (or ref)
                f1_next = pts_sorted[i + 1, 0] if i + 1 < len(pts_sorted) else ref[0]
                # height in f2: from f2 to min of all previous f2 (or ref)
                f2_prev = np.min(pts_sorted[:i, 1]) if i > 0 else ref[1]
                w = max(0.0, f1_next - f1)
                h = max(0.0, f2_prev - f2)
                contribs[i] = w * h
            # Map back to archive indices
            # Find top-3 by contribution
            top_idx = np.argsort(-contribs)[:3]
            for idx in top_idx:
                if eval_count >= max_evals or not time_ok():
                    break
                f = pts_sorted[idx]
                eps = f[1]
                # Start from the archive point
                # Find the archive point matching this f
                start_x = None
                for (x, af) in archive:
                    if np.allclose(af, f, atol=1e-12):
                        start_x = x
                        break
                if start_x is None:
                    start_x = lo + 0.5 * span

                def fun_polish(x):
                    f = np.asarray(problem(x), dtype=float)
                    eval_count += 1
                    penalty = 0.0
                    if f[1] > eps:
                        penalty = 100.0 * (f[1] - eps)
                    return f[0] + penalty

                res = minimize(
                    fun_polish, clip(start_x), method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 100, 'gtol': 1e-8}
                )
                if res.x is not None:
                    xf = clip(res.x)
                    ff = np.asarray(problem(xf), dtype=float)
                    eval_count += 1
                    record(xf, ff)

        # ---- 4. A few extra eps values if budget remains ----
        if eval_count < max_evals and time_ok():
            # Try midpoints between existing eps values
            if len(eps_list) > 1:
                extra_eps = []
                for i in range(len(eps_list) - 1):
                    extra_eps.append((eps_list[i] + eps_list[i + 1]) / 2.0)
                rng2 = np.random.RandomState(1)
                rng2.shuffle(extra_eps)
                for eps in extra_eps:
                    if eval_count >= max_evals or not time_ok():
                        break
                    best_start = None
                    best_score = None
                    for (x, f) in archive:
                        score = abs(f[1] - eps) + 0.01 * f[0]
                        if best_score is None or score < best_score:
                            best_score = score
                            best_start = x
                    if best_start is None:
                        best_start = lo + 0.5 * span
                    x0 = clip(best_start)

                    def fun_extra(x):
                        f = np.asarray(problem(x), dtype=float)
                        eval_count += 1
                        penalty = 0.0
                        if f[1] > eps:
                            penalty = 100.0 * (f[1] - eps)
                        return f[0] + penalty

                    res = minimize(
                        fun_extra, x0, method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 10, 'gtol': 1e-6}
                    )
                    if res.x is not None:
                        xf = clip(res.x)
                        ff = np.asarray(problem(xf), dtype=float)
                        eval_count += 1
                        record(xf, ff)

        return archive
# EVOLVE-BLOCK-END
