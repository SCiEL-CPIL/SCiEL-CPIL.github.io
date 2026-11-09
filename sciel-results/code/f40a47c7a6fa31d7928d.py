"""Multi-objective optimizer for RE21 using Epsilon-Constraint Method with L-BFGS-B and local refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []  # List of (x, f) tuples
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        def add_to_archive(x, f):
            new_archive = []
            for x_old, f_old in self.archive:
                if np.all(f <= f_old) and np.any(f < f_old):
                    continue
                new_archive.append((x_old, f_old))
            new_archive.append((x, f))
            self.archive = new_archive

        # Phase 1: Initialization
        n_init = min(30, self.budget // 2)
        X_init = np.random.random((n_init, self.dim))
        X_init = lo + span * X_init

        for x in X_init:
            if self.eval_count >= self.budget:
                break
            f = problem(x)
            self.eval_count += 1
            add_to_archive(x, f)

        # Find approximate optima for each objective individually
        if self.eval_count < self.budget - 10:
            def obj1(x):
                if self.eval_count >= self.budget:
                    return 0
                f = problem(x)
                self.eval_count += 1
                return f[0]

            res1 = minimize(obj1, np.random.uniform(lo, hi), method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 10, 'maxfun': 10})
            if self.eval_count < self.budget:
                f1 = problem(res1.x)
                self.eval_count += 1
                add_to_archive(res1.x, f1)

        if self.eval_count < self.budget - 10:
            def obj2(x):
                if self.eval_count >= self.budget:
                    return 0
                f = problem(x)
                self.eval_count += 1
                return f[1]

            res2 = minimize(obj2, np.random.uniform(lo, hi), method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 10, 'maxfun': 10})
            if self.eval_count < self.budget:
                f2 = problem(res2.x)
                self.eval_count += 1
                add_to_archive(res2.x, f2)

        if len(self.archive) == 0:
            return self.archive

        f_vals = np.array([f for _, f in self.archive])
        f1_min, f1_max = f_vals[:, 0].min(), f_vals[:, 0].max()
        f2_min, f2_max = f_vals[:, 1].min(), f_vals[:, 1].max()

        # Phase 2: Epsilon-Constraint Method
        remaining = self.budget - self.eval_count
        n_steps = max(10, remaining // 8)

        epsilons = np.linspace(f2_min, f2_max, n_steps)

        for eps in epsilons:
            if self.eval_count >= self.budget - 5:
                break
            if time.time() > deadline:
                break

            def obj_func(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return f[0]

            def constr_func(x):
                if self.eval_count >= self.budget:
                    return 0.0
                f = problem(x)
                self.eval_count += 1
                return eps - f[1]

            x0 = np.random.uniform(lo, hi)

            try:
                res = minimize(
                    obj_func,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constr_func}],
                    options={'maxiter': 10, 'maxfun': 15}
                )
                if self.eval_count < self.budget:
                    f_res = problem(res.x)
                    self.eval_count += 1
                    if f_res[1] <= eps + 1e-3:
                        add_to_archive(res.x, f_res)
            except Exception:
                pass

        # Phase 3: Local refinement on non-dominated points
        if time.time() < deadline - 30 and self.eval_count < self.budget:
            # Find non-dominated points
            non_dom = []
            for i, (_, f_i) in enumerate(self.archive):
                is_nd = True
                for _, f_j in self.archive:
                    if np.all(f_j <= f_i) and np.any(f_j < f_i):
                        is_nd = False
                        break
                if is_nd:
                    non_dom.append(i)

            if non_dom:
                f_vals = np.array([f for _, f in self.archive])
                f_min = f_vals.min(axis=0)
                f_range = f_vals.max(axis=0) - f_min
                f_range[f_range == 0] = 1
                F_norm = (f_vals - f_min) / f_range

                non_dom_sorted = sorted(non_dom, key=lambda i: np.sum(F_norm[i]))
                n_refine = min(8, len(non_dom_sorted), (self.budget - self.eval_count) // 3)

                for idx in non_dom_sorted[:n_refine]:
                    if self.eval_count >= self.budget:
                        break
                    if time.time() > deadline:
                        break

                    x0 = self.archive[idx][0].copy()

                    def scalar_obj(x):
                        if self.eval_count >= self.budget:
                            return 0.0
                        f = problem(x)
                        self.eval_count += 1
                        return f[0] + f[1]

                    try:
                        res = minimize(scalar_obj, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 15, 'xatol': 1e-6, 'fatol': 1e-6})
                        if self.eval_count < self.budget:
                            f_new = problem(res.x)
                            self.eval_count += 1
                            add_to_archive(res.x, f_new)
                    except Exception:
                        pass

        return self.archive
# EVOLVE-BLOCK-END
