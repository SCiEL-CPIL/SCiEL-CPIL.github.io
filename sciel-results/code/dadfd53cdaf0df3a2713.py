"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Epsilon-Constraint Method with careful budget tracking."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        mid = (lo + hi) / 2.0

        eval_count = 0

        def eval(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            eval_count += 1
            return f

        def time_ok():
            return time.time() < deadline and eval_count < self.budget

        # Phase 1: Initial diverse sampling
        n_init = 16
        X_init = lo + span * rng.random((n_init, self.dim))
        F_init = np.array([eval(x) for x in X_init])

        f1_min, f2_min = F_init[:, 0].min(), F_init[:, 1].min()
        f1_max, f2_max = F_init[:, 0].max(), F_init[:, 1].max()

        # Find individual optima using a few gradient steps
        # Minimize f1
        x_opt1 = mid.copy()
        for _ in range(8):
            if not time_ok():
                break
            f = eval(x_opt1)
            f1_val = f[0]
            # Numerical gradient of f1
            grad = np.zeros(self.dim)
            h = 1e-4
            for j in range(self.dim):
                xp = x_opt1.copy(); xp[j] += h
                xm = x_opt1.copy(); xm[j] -= h
                fp = eval(np.clip(xp, lo, hi))
                fm = eval(np.clip(xm, lo, hi))
                grad[j] = (fp[0] - fm[0]) / (2 * h)
            # Gradient descent step
            lr = 0.1
            x_opt1 = np.clip(x_opt1 - lr * grad, lo, hi)

        # Minimize f2
        x_opt2 = mid.copy()
        for _ in range(8):
            if not time_ok():
                break
            f = eval(x_opt2)
            f2_val = f[1]
            grad = np.zeros(self.dim)
            h = 1e-4
            for j in range(self.dim):
                xp = x_opt2.copy(); xp[j] += h
                xm = x_opt2.copy(); xm[j] -= h
                fp = eval(np.clip(xp, lo, hi))
                fm = eval(np.clip(xm, lo, hi))
                grad[j] = (fp[1] - fm[1]) / (2 * h)
            lr = 0.1
            x_opt2 = np.clip(x_opt2 - lr * grad, lo, hi)

        f1_opt1 = eval(x_opt1)[0]
        f2_opt2 = eval(x_opt2)[1]
        f1_min = min(f1_min, f1_opt1)
        f2_min = min(f2_min, f2_opt2)

        # Phase 2: Epsilon-constraint tracing
        n_eps = 12
        eps_range = f2_max - f2_min
        eps_values = np.linspace(f2_max + 0.1 * eps_range, f2_min, n_eps)

        for i, eps in enumerate(eps_values):
            if not time_ok():
                break

            # Starting point: interpolate between optima
            t = i / max(1, n_eps - 1)
            x0 = (1 - t) * x_opt1 + t * x_opt2
            x0 = np.clip(x0, lo, hi)

            # Augmented epsilon constraint: minimize f1 + mu*(f2 - eps)^+
            mu = 5.0
            x_cur = x0.copy()

            for step in range(5):
                if not time_ok():
                    break
                f = eval(x_cur)
                f1, f2 = f[0], f[1]
                penalty = mu * max(0, f2 - eps) ** 2
                obj_val = f1 + penalty

                # Numerical gradient
                grad = np.zeros(self.dim)
                h = 1e-4
                for j in range(self.dim):
                    xp = x_cur.copy(); xp[j] += h
                    xm = x_cur.copy(); xm[j] -= h
                    fp = eval(np.clip(xp, lo, hi))
                    fm = eval(np.clip(xm, lo, hi))
                    obj_p = fp[0] + mu * max(0, fp[1] - eps) ** 2
                    obj_m = fm[0] + mu * max(0, fm[1] - eps) ** 2
                    grad[j] = (obj_p - obj_m) / (2 * h)

                # Step
                lr = 0.05
                x_new = np.clip(x_cur - lr * grad, lo, hi)
                if np.allclose(x_new, x_cur):
                    break
                x_cur = x_new

            # Record the result
            f_final = eval(x_cur)
            # Add to archive (we'll filter later)

        # Phase 3: Local refinement
        if time_ok():
            # Try perturbations around the best points
            best_candidates = [x_opt1, x_opt2]
            # Add some points from the epsilon tracing
            for _ in range(min(4, self.budget - eval_count - 10)):
                if not time_ok():
                    break
                base = best_candidates[rng.integers(len(best_candidates))]
                pert = 0.05 * span * rng.standard_normal(self.dim)
                x_try = np.clip(base + pert, lo, hi)
                eval(x_try)

        return mid
# EVOLVE-BLOCK-END
