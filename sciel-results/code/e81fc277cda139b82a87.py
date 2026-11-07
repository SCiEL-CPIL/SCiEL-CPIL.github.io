"""Nelder-Mead simplex with Latin Hypercube initialization for BBOB f4 (Büche-Rastrigin)."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        call_count = 0
        best_x = np.zeros(dim, dtype=float)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                raise ValueError("Budget exceeded")
            call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return time.time() < deadline

        def lhm_sample(n, dim, rng, bounds=(-4.5, 4.5)):
            """Latin Hypercube sampling for global coverage."""
            pts = []
            for _ in range(n):
                p = np.zeros(dim)
                for d in range(dim):
                    p[d] = rng.uniform(bounds[0], bounds[1])
                pts.append(p)
            return pts

        def nelder_mead(x0, max_calls=50, initial_step=0.5, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
            """Nelder-Mead simplex with adaptive vertex scaling."""
            nonlocal call_count, best_f, best_x
            x = np.asarray(x0, dtype=float).copy()
            f = float(do_call(x))
            calls_used = 1

            # Initialize simplex
            simplex = [x.copy()]
            simple_vals = [f]
            for i in range(dim):
                xi = x.copy()
                if xi[i] + initial_step <= 4.95:
                    xi[i] += initial_step
                else:
                    xi[i] -= initial_step
                fi = do_call(xi)
                calls_used += 1
                simplex.append(xi)
                simple_vals.append(fi)

            max_iter = max_calls // (dim + 1)
            for it in range(max_iter):
                if calls_used >= max_calls or call_count >= budget - 5 or not time_ok():
                    break

                # Sort simplex
                order = np.argsort(simple_vals)
                simplex = [simplex[i] for i in order]
                simple_vals = [simple_vals[i] for i in order]

                f_best = simple_vals[0]
                f_worst = simple_vals[-1]
                f_second_worst = simple_vals[-2]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -4.95, 4.95)
                fr = do_call(xr)
                calls_used += 1

                if fr < f_best:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -4.95, 4.95)
                    fe = do_call(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simple_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        simple_vals[-1] = fr
                elif fr < f_second_worst:
                    simplex[-1] = xr
                    simple_vals[-1] = fr
                else:
                    # Contraction
                    if fr < f_worst:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -4.95, 4.95)
                        fc = do_call(xc)
                        calls_used += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            simple_vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -4.95, 4.95)
                                fi = do_call(simplex[i])
                                calls_used += 1
                                simple_vals[i] = fi
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -4.95, 4.95)
                        fc = do_call(xc)
                        calls_used += 1
                        if fc < f_worst:
                            simplex[-1] = xc
                            simple_vals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -4.95, 4.95)
                                fi = do_call(simplex[i])
                                calls_used += 1
                                simple_vals[i] = fi

                # Update best
                if simple_vals[0] < best_f:
                    best_f = simple_vals[0]
                    best_x = simplex[0].copy()

            return simplex[0], simple_vals[0]

        # ---------- Phase 0: Latin Hypercube initial exploration ----------
        n_lhm = 15
        lhm_pts = lhm_sample(n_lhm, dim, rng)
        for p in lhm_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)

        # Add origin and axis points for better coverage
        extra_pts = [np.zeros(dim)]
        for axis in range(dim):
            for s in (-2.0, 2.0):
                p = np.zeros(dim)
                p[axis] = s
                extra_pts.append(p)

        for p in extra_pts:
            if call_count >= budget or not time_ok():
                break
            p = np.clip(p, -4.95, 4.95)
            f = do_call(p)

        # ---------- Phase 1: Nelder-Mead from top initial points ----------
        # Collect all evaluated points
        evaluated = []
        for p in lhm_pts + extra_pts:
            if call_count > 0:  # Only if we actually called
                pass  # We don't track which were called, so we'll just use best_x

        # Run Nelder-Mead from best point and a few others
        nm_starts = [best_x.copy()]
        # Add a few random starts near best
        for _ in range(3):
            if call_count >= budget - 20 or not time_ok():
                break
            perturbation = rng.normal(0, 0.5, dim)
            start = best_x + perturbation
            start = np.clip(start, -4.95, 4.95)
            nm_starts.append(start)

        for start in nm_starts:
            if call_count >= budget - 10 or not time_ok():
                break
            try:
                nelder_mead(start, max_calls=40, initial_step=0.3)
            except Exception:
                pass

        # ---------- Phase 2: Final high-precision polish ----------
        if call_count < budget - 1 and time_ok():
            try:
                res = minimize(
                    do_call,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'initial_simplex': None}
                )
            except Exception:
                pass

        return best_x, best_f
