"""Optimizer for BBOB-style f17 (Schaffers F7) using CMA-ES with adaptive restarts."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES based optimizer with adaptive initial sampling and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_left():
            return deadline - time.time()

        def can_continue():
            return calls_used < budget and time_left() > 1.0

        # Phase 1: Initial sampling - Halton sequence for quasi-random coverage
        def halton_sequence(n, dim):
            """Generate Halton sequence points in [0,1]^dim."""
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, 2 + dim):
                    f = 1.0
                    r = 0.0
                    i_temp = i
                    while i_temp > 0:
                        f /= base
                        r += f * (i_temp % base)
                        i_temp //= base
                    point.append(r)
                points.append(point)
            return np.array(points)

        n_initial = 25
        halton_pts = halton_sequence(n_initial, dim)
        
        # First few points: center and corners for quick orientation
        special_points = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0] if dim == 5 else [2.0] * dim),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0] if dim == 5 else [-2.0] * dim),
        ]
        
        for i in range(min(5, n_initial)):
            if not can_continue():
                break
            x = np.clip(special_points[i], -5, 5)
            f = func(x)
            calls_used += 1
            update_best(x, f)
        
        # Then Halton points scaled to [-4, 4] (minimizer is in [-4,4]^5)
        for i in range(5, n_initial):
            if not can_continue():
                break
            x = -4 + 8 * halton_pts[i]
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            update_best(x, f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: CMA-ES with adaptive restarts
        def run_cmaes(x_start, sigma_init, max_calls):
            nonlocal calls_used, best_x, best_f

            # CMA-ES parameters for 5D - tuned for faster convergence
            lam = 12  # Population size
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)

            # Step size adaptation
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (2 + mueff / dim) / (dim + cc + 2)
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1) - 1)) + cc

            # Covariance matrix adaptation
            c1 = (mueff - 2 + 1 / dim) / ((dim + 2) ** 2 - mueff)
            cmu = min(1 - cc, 2 / ((dim + 1.3) ** 2 + mueff))
            cmu = min(cmu, (1 - 2 * c1) / 2)

            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # State
            x_mean = x_start.copy()
            sigma = sigma_init
            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            no_improve = 0
            best_local_f = best_f
            local_calls = 0

            while local_calls < max_calls:
                if not can_continue():
                    break

                # Sample new points
                ys = rng.standard_normal((lam, dim))
                xs = np.array([x_mean + sigma * (ys[i] @ C_sqrt) for i in range(lam)])
                xs = np.clip(xs, -5, 5)

                # Evaluate
                fs = np.zeros(lam)
                for i in range(lam):
                    if not can_continue():
                        break
                    fs[i] = func(xs[i])
                    calls_used += 1
                    local_calls += 1
                    update_best(xs[i], fs[i])

                if not can_continue():
                    break

                # Sort
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]
                ys = ys[order]

                # Check improvement
                if fs[0] < best_local_f - 1e-10:
                    no_improve = 0
                    best_local_f = fs[0]
                else:
                    no_improve += 1
                    if no_improve > 12:  # Restart sooner
                        # Restart with larger sigma
                        x_mean = best_x.copy()
                        sigma = max(sigma * 1.5, 0.3)
                        C = np.eye(dim)
                        C_sqrt = np.eye(dim)
                        pc = np.zeros(dim)
                        ps = np.zeros(dim)
                        no_improve = 0
                        continue

                # Update mean
                x_mean_new = np.sum(weights[:, None] * xs[:mu], axis=0)

                # Update step size
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((x_mean_new - x_mean) / sigma))
                sigma = sigma * np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))
                sigma = np.clip(sigma, 1e-7, 8.0)

                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (x_mean_new - x_mean) / sigma

                # Update covariance
                ys_parents = ys[:mu]
                S_w = np.sum(weights[:, None, None] * (ys_parents[:, :, None] * ys_parents[:, None, :]), axis=0)
                C_new = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C) + cmu * S_w
                C_new = (C_new + C_new.T) / 2.0

                try:
                    C_sqrt = np.linalg.cholesky(C_new)
                except np.linalg.LinAlgError:
                    C_new += 1e-8 * np.eye(dim)
                    try:
                        C_sqrt = np.linalg.cholesky(C_new)
                    except:
                        C_sqrt = np.eye(dim)
                        break

                C = C_new
                x_mean = x_mean_new

            return local_calls

        # Run main CMA-ES
        remaining = budget - calls_used
        run_cmaes(best_x.copy(), 1.0, remaining)

        # Phase 3: Nelder-Mead polish for fine convergence
        if best_x is not None and can_continue() and calls_used < budget - 10:
            def nm_func(x):
                nonlocal calls_used
                x = np.clip(x, -5, 5)
                f = func(x)
                calls_used += 1
                return f

            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

        # Phase 4: Additional CMA-ES refinement if budget remains
        if can_continue() and best_x is not None:
            remaining = budget - calls_used
            if remaining > 20:
                run_cmaes(best_x.copy(), 0.1, remaining)

        # Phase 5: Final fine-tune with very small perturbations
        if best_x is not None and can_continue():
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            for _ in range(20):
                if not can_continue():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    update_best(x, fx)
                else:
                    sigma *= 0.8
                    if sigma < 1e-7:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            if calls_used < budget:
                best_f = func(best_x)
                calls_used += 1

        return best_x, best_f
