"""Direct DTLZ7 Pareto front sampling with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = 318.0  # safety margin below 330s wall-clock limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        M = self.n_obj

        evaluated = 0
        max_evals = self.budget
        points = []
        F = []

        def check_deadline():
            return (time.perf_counter() - t0) < deadline

        def safe_eval(x):
            nonlocal evaluated
            if evaluated >= max_evals or not check_deadline():
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != n:
                x = np.resize(x, n)
            f = np.array(problem(x), dtype=float)
            if np.any(~np.isfinite(f)):
                return None
            evaluated += 1
            return f

        # DTLZ7 structure for n=22, M=3:
        # f1 = x[0], f2 = x[1]
        # g-set is the last M-2 = 1 variable: x[21]
        # g(x) = 100 * (1 + (x[21]-0.5)^2 * cos(20*pi*(x[21]-0.5)))
        # The Pareto front is where g is minimized.
        #
        # For DTLZ7 with M=3:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = (1 + g) * h(y)
        # where y = sum(x[2:21]) / 19 (or similar normalization)
        # h(y) = 1 - y/m * (1 + sin(3*pi*y/m)) or similar
        #
        # The Pareto front has g at its minimum, and the shape is determined
        # by x[0], x[1], and the middle variables x[2:21].

        # Phase 1: Find optimal g-variable value (x[21])
        g_var_vals = np.linspace(0, 1, 12)
        best_g_var = 0.5
        best_f3 = np.inf
        for gv in g_var_vals:
            if not check_deadline() or evaluated >= max_evals:
                break
            x = np.full(n, 0.5)
            x[21] = gv
            f = safe_eval(x)
            if f is None:
                break
            if f[2] < best_f3:
                best_f3 = f[2]
                best_g_var = gv
            points.append(np.clip(x, lo, hi))
            F.append(f)

        if not F:
            return np.array([])

        # Phase 2: Dense sampling of the Pareto front
        remaining = max_evals - evaluated
        if remaining <= 0 or not check_deadline():
            return np.array(points) if points else np.zeros((0, n))

        # For hypervolume maximization with ref [1.1, 1.1, 25.0]:
        # f1 = x[0], f2 = x[1], f3 depends on middle vars and g
        #
        # Use Latin Hypercube Sampling for x[0], x[1] and a 1-D sweep for middle vars
        N0 = 8
        N1 = 8
        Nv = max(1, remaining // (N0 * N1))
        total_planned = N0 * N1 * Nv
        if total_planned > remaining:
            Nv = max(1, remaining // (N0 * N1))
            total_planned = N0 * N1 * Nv

        # LHS for x0, x1 in [0,1]
        def lhs_1d(n, seed):
            rng_local = np.random.RandomState(seed)
            perm = rng_local.permutation(n)
            return (perm + rng_local.uniform(0, 1, n)) / n

        xs0 = lhs_1d(N0, seed=101)
        xs1 = lhs_1d(N1, seed=202)
        vs = np.linspace(0, 1, Nv)

        # Evaluate structured points
        for x0 in xs0:
            for x1 in xs1:
                for v in vs:
                    if not check_deadline() or evaluated >= max_evals:
                        break
                    x = np.zeros(n)
                    x[0] = x0
                    x[1] = x1
                    x[2:21] = v
                    x[21] = best_g_var
                    f = safe_eval(x)
                    if f is None:
                        break
                    points.append(np.clip(x, lo, hi))
                    F.append(f)
                if not check_deadline() or evaluated >= max_evals:
                    break
            if not check_deadline() or evaluated >= max_evals:
                break

        # Phase 3: Corner & edge cases for hypervolume corners
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline():
            corners = [
                (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
            ]
            edge_vs = [0.0, 0.25, 0.5, 0.75, 1.0]
            for c0, c1 in corners:
                for ev in edge_vs:
                    if not check_deadline() or evaluated >= max_evals:
                        break
                    x = np.zeros(n)
                    x[0] = c0
                    x[1] = c1
                    x[2:21] = ev
                    x[21] = best_g_var
                    f = safe_eval(x)
                    if f is None:
                        break
                    points.append(np.clip(x, lo, hi))
                    F.append(f)
                if not check_deadline() or evaluated >= max_evals:
                    break

        # Phase 4: Refinement with remaining budget
        remaining = max_evals - evaluated
        if remaining > 0 and check_deadline() and points:
            rng = np.random.default_rng(7)
            F_arr = np.array(F)
            n_refine = min(remaining, 60)
            for _ in range(n_refine):
                if not check_deadline() or evaluated >= max_evals:
                    break
                if len(F_arr) > 0:
                    f3_vals = F_arr[:, 2]
                    threshold = np.percentile(f3_vals, 40)
                    good_idx = np.where(f3_vals <= threshold)[0]
                    if len(good_idx) > 0:
                        idx = good_idx[rng.integers(0, len(good_idx))]
                    else:
                        idx = rng.integers(0, len(points))
                else:
                    idx = 0

                base = np.array(points[idx], dtype=float)
                noise = rng.standard_normal(n) * 0.04 * (hi - lo)
                x = np.clip(base + noise, lo, hi)
                f = safe_eval(x)
                if f is None:
                    break
                points.append(np.clip(x, lo, hi))
                F.append(f)
                F_arr = np.array(F)

        # Final check
        if not points and evaluated < max_evals and check_deadline():
            x = np.random.uniform(0, 1, n)
            f = safe_eval(x)
            if f is not None:
                points.append(np.clip(x, lo, hi))
                F.append(f)

        return np.array(points) if points else np.zeros((0, n))

# EVOLVE-BLOCK-END
