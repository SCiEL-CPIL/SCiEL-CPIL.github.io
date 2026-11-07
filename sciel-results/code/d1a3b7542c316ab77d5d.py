"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with momentum and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Track best solution
        best_x = None
        best_fx = np.inf
        calls = 0

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float).copy()

        # Phase 1: (1+1)-ES with adaptive step size and momentum
        # Use ~75% of budget for ES
        es_budget = int(budget * 0.75)

        # Initialize at center
        x = np.zeros(d)
        fx = func(x)
        calls += 1
        record(x, fx)

        sigma = 3.0
        # Momentum from recent successful directions
        recent_dirs = []
        window = 10

        for i in range(es_budget):
            if not time_ok() or calls >= budget - 50:
                break

            # Generate direction with momentum
            direction = rng.standard_normal(d)
            if recent_dirs:
                avg_dir = np.mean(recent_dirs, axis=0)
                norm_avg = np.linalg.norm(avg_dir)
                if norm_avg > 1e-10:
                    avg_dir = avg_dir / norm_avg
                    direction = 0.7 * direction + 0.3 * avg_dir
                    norm_d = np.linalg.norm(direction)
                    if norm_d > 1e-10:
                        direction = direction / norm_d

            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls += 1
            record(y, fy)

            if fy < fx:
                # Success
                fx = fy
                x = y.copy()
                if len(recent_dirs) >= window:
                    recent_dirs.pop(0)
                recent_dirs.append(direction.copy())
                # Increase step size on success
                sigma *= 1.12
            else:
                # Failure
                sigma *= 0.88

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 8.0)

        # Phase 2: Nelder-Mead polish with remaining budget
        remaining = budget - calls
        if remaining > 20 and time_ok():
            # Build initial simplex around best_x
            nm_sigma = max(sigma * 0.5, 0.05)
            initial_simplex = [best_x.copy()]
            for i in range(d):
                perturbation = best_x.copy()
                perturbation[i] += nm_sigma
                perturbation[i] = np.clip(perturbation[i], -5, 5)
                initial_simplex.append(perturbation)

            nm_calls = [0]

            def nm_func(x_vec):
                nonlocal calls
                if not time_ok() or calls >= budget:
                    return best_fx
                x_vec = np.clip(x_vec, -5, 5)
                val = func(x_vec)
                calls += 1
                nm_calls[0] += 1
                record(x_vec, val)
                return val

            try:
                # Manual Nelder-Mead implementation to have full control over calls
                simplex = [np.array(p, dtype=float) for p in initial_simplex]
                fvals = [nm_func(p) for p in simplex]

                max_iter = min(remaining - 10, 200)
                for iteration in range(max_iter):
                    if not time_ok() or calls >= budget - 5:
                        break

                    # Sort simplex by function value
                    order = np.argsort(fvals)
                    simplex = [simplex[i].copy() for i in order]
                    fvals = [fvals[i] for i in order]

                    # Compute centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    alpha = 1.0
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = nm_func(x_ref)

                    if f_ref < fvals[0]:
                        # Expansion
                        gamma = 2.0
                        x_exp = centroid + gamma * (centroid - simplex[-1])
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = nm_func(x_exp)

                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    elif f_ref < fvals[-2]:
                        # Just accept reflection
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                    else:
                        # Contraction
                        rho = 0.5
                        x_con = centroid + rho * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = nm_func(x_con)

                        if f_con < fvals[-1]:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = nm_func(simplex[i])
                                if not time_ok() or calls >= budget - 5:
                                    break
                            if not time_ok() or calls >= budget - 5:
                                break

                    # Check convergence
                    if len(fvals) > 0:
                        f_range = max(fvals) - min(fvals)
                        if f_range < 1e-10:
                            break

            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
