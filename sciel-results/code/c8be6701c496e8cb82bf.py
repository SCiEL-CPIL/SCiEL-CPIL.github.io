"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class BudgetExceeded(Exception):
    pass


class Optimizer:
    """Multi-objective optimizer for RE21 using LHS init, epsilon-constraint with L-BFGS-B,
    and Nelder-Mead polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        used = 0

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                raise BudgetExceeded()
            if not check_time():
                raise BudgetExceeded()
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            used += 1
            return f

        # Phase 1: Latin Hypercube Sampling for diverse initial points
        n_init = 25
        # LHS: divide each dimension into n_init equal intervals, permute
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(n_init)
            lhs_points[np.arange(n_init), d] = (perm + rng.random(n_init)) / n_init
        X_init = lo + span * lhs_points
        F_init = []
        for x in X_init:
            try:
                F_init.append(evaluate(x))
            except BudgetExceeded:
                break
        if not F_init:
            return mid
        F_init = np.array(F_init)

        f1_min = F_init[:, 0].min()
        f1_max = F_init[:, 0].max()
        f2_min = F_init[:, 1].min()
        f2_max = F_init[:, 1].max()

        archive_X = list(X_init[:len(F_init)])
        archive_F = list(F_init)

        # Phase 2: Find individual optima
        x_opt1 = None
        x_opt2 = None
        try:
            def obj_f1(x):
                return evaluate(x)[0]
            res1 = minimize(obj_f1, mid, method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 8, 'ftol': 1e-8, 'gtol': 1e-6})
            if res1.success:
                x_opt1 = np.clip(res1.x, lo, hi)
                f_opt1 = evaluate(x_opt1)
                archive_X.append(x_opt1)
                archive_F.append(f_opt1)
                f1_min = min(f1_min, f_opt1[0])
        except BudgetExceeded:
            pass
        except Exception:
            pass

        try:
            def obj_f2(x):
                return evaluate(x)[1]
            res2 = minimize(obj_f2, mid, method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 8, 'ftol': 1e-8, 'gtol': 1e-6})
            if res2.success:
                x_opt2 = np.clip(res2.x, lo, hi)
                f_opt2 = evaluate(x_opt2)
                archive_X.append(x_opt2)
                archive_F.append(f_opt2)
                f2_min = min(f2_min, f_opt2[1])
        except BudgetExceeded:
            pass
        except Exception:
            pass

        # Phase 3: Epsilon-constraint method to trace Pareto front
        # Use 12 epsilon values for better coverage
        n_eps = 12
        if used > self.budget - 60:
            n_eps = max(4, n_eps - 4)
        if used > self.budget - 30:
            n_eps = max(2, n_eps - 4)

        # Epsilon values for f2 constraint, from high to low
        eps_values = np.linspace(f2_max * 1.1, f2_min * 0.99, n_eps)
        eps_values = np.sort(eps_values)[::-1]  # descending

        for i in range(n_eps):
            if not check_time() or used >= self.budget - 8:
                break
            eps = eps_values[i]

            # Starting point: interpolate between optima
            if x_opt1 is not None and x_opt2 is not None:
                w = i / max(1, n_eps - 1)
                x0 = w * x_opt1 + (1 - w) * x_opt2
            else:
                x0 = mid
            x0 = np.clip(x0, lo, hi)

            # Augmented epsilon-constraint: minimize f1 + mu*(f2 - eps)^+
            mu = 50.0

            def objective(x):
                x_clipped = np.clip(x, lo, hi)
                f = evaluate(x_clipped)
                f1, f2 = f[0], f[1]
                penalty = mu * max(0.0, f2 - eps) ** 2
                return f1 + penalty

            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                  bounds=list(zip(lo, hi)),
                                  options={'maxiter': 15, 'ftol': 1e-10, 'gtol': 1e-8})
                if result.success:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = evaluate(x_best)
                    archive_X.append(x_best)
                    archive_F.append(f_best)
            except BudgetExceeded:
                break
            except Exception:
                pass

        # Phase 4: Nelder-Mead polish on non-dominated points
        if check_time() and used < self.budget - 5 and archive_F:
            F_arr = np.array(archive_F)
            # Find non-dominated points
            n = len(F_arr)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]:
                    continue
                for j in range(n):
                    if i == j or not nondom[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        nondom[i] = False
                        break

            # Polish top non-dominated points with Nelder-Mead
            nd_indices = np.where(nondom)[0]
            # Limit polish to at most 6 points to save budget
            max_polish = min(6, len(nd_indices))
            if max_polish > 0:
                # Sort by hypervolume contribution proxy: distance to reference
                ref = np.array([3000.0, 0.06])
                hv_contrib = np.zeros(len(nd_indices))
                for k, idx in enumerate(nd_indices):
                    # Approximate: points closer to (0,0) in objective space are better
                    hv_contrib[k] = -(F_arr[idx, 0] + F_arr[idx, 1] * 50000)  # weight f2 heavily
                top_idx = nd_indices[np.argsort(hv_contrib)[:max_polish]]

                for idx in top_idx:
                    if not check_time() or used >= self.budget - 2:
                        break
                    x_start = np.clip(archive_X[idx], lo, hi)
                    try:
                        def nm_obj(x):
                            x_c = np.clip(x, lo, hi)
                            f = evaluate(x_c)
                            # Minimize a weighted sum to stay near Pareto front
                            return f[0] + f[1] * 50000
                        nm_res = minimize(nm_obj, x_start, method='Nelder-Mead',
                                          options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10})
                        if nm_res.success:
                            x_nm = np.clip(nm_res.x, lo, hi)
                            f_nm = evaluate(x_nm)
                            archive_X.append(x_nm)
                            archive_F.append(f_nm)
                    except BudgetExceeded:
                        break
                    except Exception:
                        pass

        # Phase 5: Final cleanup
        if archive_F:
            F_final = np.array(archive_F)
            n_f = len(F_final)
            nondom_mask = np.ones(n_f, dtype=bool)
            for i in range(n_f):
                if not nondom_mask[i]:
                    continue
                for j in range(n_f):
                    if i == j or not nondom_mask[j]:
                        continue
                    if np.all(F_final[j] <= F_final[i]) and np.any(F_final[j] < F_final[i]):
                        nondom_mask[i] = False
                        break

        return archive_X[0] if archive_X else mid
# EVOLVE-BLOCK-END
