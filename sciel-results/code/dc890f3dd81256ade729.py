# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search with epsilon-constraint scalarization for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        local_rng = np.random.default_rng(12345)
        eval_count = [0]  # mutable counter

        def eval_problem(x):
            """Evaluate with counting."""
            eval_count[0] += 1
            return problem(x)

        # Phase 1: broad random sampling
        n_init = 25
        X_init = lo + (hi - lo) * local_rng.random((n_init, dim))
        F_init = np.array([eval_problem(x) for x in X_init])

        f_min = F_init.min(axis=0)
        f_max = F_init.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-10] = 1.0

        best_points = []
        best_f = []

        def add_point(x, f):
            f = np.array(f, dtype=float)
            if np.any(f >= [1.1, 1.3, 1.1]):  # reference point check
                return
            for bf in best_f:
                if np.all(bf <= f) and np.any(bf < f):
                    return
            i = 0
            while i < len(best_f):
                if np.all(f <= best_f[i]) and np.any(f < best_f[i]):
                    best_f.pop(i)
                    best_points.pop(i)
                else:
                    i += 1
            best_f.append(f)
            best_points.append(x.copy())

        def nelder_mead(f_obj, x0, max_iter=50, tol=1e-8):
            """Nelder-Mead with eval counting via f_obj."""
            n = len(x0)
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.05 * (hi[i] - lo[i])

            f_vals = [f_obj(simplex[i]) for i in range(n + 1)]

            for _ in range(max_iter):
                if eval_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                if abs(f_vals[-1] - f_vals[0]) < tol:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(centroid + 1.0 * (centroid - simplex[-1]), lo, hi)
                fr = f_obj(xr)

                if fr < f_vals[0]:
                    xe = np.clip(centroid + 2.0 * (centroid - simplex[-1]), lo, hi)
                    fe = f_obj(xe)
                    if fe < fr:
                        simplex[-1], f_vals[-1] = xe, fe
                    else:
                        simplex[-1], f_vals[-1] = xr, fr
                elif fr < f_vals[-2]:
                    simplex[-1], f_vals[-1] = xr, fr
                else:
                    if fr < f_vals[-1]:
                        xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), lo, hi)
                        fc = f_obj(xc)
                        if fc <= fr:
                            simplex[-1], f_vals[-1] = xc, fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f_vals[i] = f_obj(simplex[i])
                    else:
                        xc = np.clip(centroid - 0.5 * (centroid - simplex[-1]), lo, hi)
                        fc = f_obj(xc)
                        if fc < f_vals[-1]:
                            simplex[-1], f_vals[-1] = xc, fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f_vals[i] = f_obj(simplex[i])

            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx]

        # Phase 2: epsilon-constraint scalarization
        start_points = list(X_init)
        extra = lo + (hi - lo) * local_rng.random((8, dim))
        start_points.extend(extra)

        for c in range(n_obj):
            if eval_count[0] >= budget or time.time() > deadline:
                break
            others = [o for o in range(n_obj) if o != c]
            o1, o2 = others

            f_c_vals = F_init[:, c]
            eps_levels = np.quantile(f_c_vals, [0.85, 0.7, 0.55, 0.4, 0.25])
            eps_levels = np.maximum(eps_levels, f_min[c] + 1e-6)

            weights = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.85, 0.15)]

            for eps in eps_levels:
                if eval_count[0] >= budget or time.time() > deadline:
                    break
                for w1, w2 in weights:
                    if eval_count[0] >= budget or time.time() > deadline:
                        break

                    def make_scalar(w1, w2, eps, c, o1, o2):
                        def scalar(x):
                            f = eval_problem(x)
                            base = w1 * f[o1] + w2 * f[o2]
                            if f[c] > eps:
                                base += 5.0 * (f[c] - eps)
                            return base
                        return scalar

                    scalar_obj = make_scalar(w1, w2, eps, c, o1, o2)

                    # 2 starting points per config
                    for s in range(2):
                        if eval_count[0] >= budget or time.time() > deadline:
                            break
                        if s == 0:
                            x0 = start_points[(c * 4 + int(eps_levels.tolist().index(eps)) + s) % len(start_points)]
                        else:
                            x0 = local_rng.random(dim) * (hi - lo) + lo

                        x_best, _ = nelder_mead(scalar_obj, x0, max_iter=35)
                        f_actual = eval_problem(x_best)
                        add_point(x_best, f_actual)

        # Phase 3: local refinement of Pareto points
        if best_points and eval_count[0] < budget and time.time() < deadline:
            F_arr = np.array(best_f)
            norm_F = (F_arr - f_min) / f_range
            sums = norm_F.sum(axis=1)
            order = np.argsort(sums)
            K = min(8, len(order))
            for idx in order[:K]:
                if eval_count[0] >= budget or time.time() > deadline:
                    break
                x0 = best_points[idx]

                def tcheb(x):
                    f = eval_problem(x)
                    return float(max(f))

                x_best, _ = nelder_mead(tcheb, x0, max_iter=25)
                f_actual = eval_problem(x_best)
                add_point(x_best, f_actual)

        if best_points:
            return np.array(best_points)
        return X_init

# EVOLVE-BLOCK-END
