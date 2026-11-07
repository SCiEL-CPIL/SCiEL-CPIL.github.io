import numpy as np
import time
import math


class Optimizer:
    """Global basin search + Rastrigin-aware local refinement for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0

        def time_left():
            return deadline - time.time()

        def do_eval(x):
            nonlocal eval_count, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def rastrigin_local(x0, max_evals, seed):
            """Local search that exploits Rastrigin periodicity."""
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time_left() < 2.0:
                return

            rng = np.random.default_rng(seed)
            x = np.clip(x0.copy(), -5, 5)
            fx = do_eval(x)
            local_evals = 1
            best_local_f = fx
            best_local_x = x.copy()

            # Try to snap to nearest Rastrigin minimum (multiples of 2*pi)
            # First, try a few perturbations to escape local minima
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

            for step in step_sizes:
                if local_evals >= max_evals or time_left() < 2.0 or eval_count >= budget:
                    break

                # Try moving each coordinate toward nearest multiple of 2*pi
                improved = False
                for coord in range(dim):
                    if local_evals >= max_evals or time_left() < 2.0 or eval_count >= budget:
                        break

                    # Current value
                    x_curr = x[coord]

                    # Nearest multiples of 2*pi
                    pi2 = 2 * math.pi
                    k = round(x_curr / pi2)
                    target = k * pi2

                    # Try moving toward target
                    delta = target - x_curr
                    if abs(delta) > 1e-10:
                        x_new = x.copy()
                        x_new[coord] = target
                        x_new = np.clip(x_new, -5, 5)
                        f_new = do_eval(x_new)
                        local_evals += 1

                        if f_new < best_local_f - 1e-10:
                            best_local_f = f_new
                            best_local_x = x_new.copy()
                            x = x_new.copy()
                            improved = True

                # If no improvement with snapping, try random perturbations
                if not improved:
                    for _ in range(3):
                        if local_evals >= max_evals or time_left() < 2.0 or eval_count >= budget:
                            break
                        x_new = x + step * rng.standard_normal(dim)
                        x_new = np.clip(x_new, -5, 5)
                        f_new = do_eval(x_new)
                        local_evals += 1
                        if f_new < best_local_f - 1e-10:
                            best_local_f = f_new
                            best_local_x = x_new.copy()
                            x = x_new.copy()
                            break

                # If we've converged (very small step and no improvement), stop
                if step < 0.005 and not improved:
                    break

            # Final: try to snap all coordinates to nearest 2*pi multiple
            if time_left() > 2.0 and eval_count < budget:
                pi2 = 2 * math.pi
                x_snapped = np.round(best_local_x / pi2) * pi2
                x_snapped = np.clip(x_snapped, -5, 5)
                f_snapped = do_eval(x_snapped)
                if f_snapped < best_local_f - 1e-10:
                    best_local_f = f_snapped
                    best_local_x = x_snapped.copy()

            return best_local_x, best_local_f

        def nelder_mead(x0, max_evals, seed):
            """Nelder-Mead local optimizer."""
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time_left() < 2.0:
                return

            rng = np.random.default_rng(seed)
            n = dim

            # Initialize simplex
            simplex = np.empty((n + 1, n))
            simplex[0] = np.clip(x0.copy(), -5, 5)
            for i in range(1, n + 1):
                simplex[i] = simplex[0].copy()
                simplex[i, i - 1] += 0.1

            values = np.empty(n + 1)
            for i in range(n + 1):
                values[i] = do_eval(simplex[i])
                if eval_count >= budget or time_left() < 2.0:
                    return

            local_evals = n + 1
            best_local_f = np.min(values)
            best_local_x = simplex[np.argmin(values)].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            max_iter = max_evals // (n + 1)
            for _ in range(max_iter):
                if local_evals >= max_evals or time_left() < 2.0 or eval_count >= budget:
                    break

                # Sort simplex
                order = np.argsort(values)
                simplex = simplex[order]
                values = values[order]

                if values[0] < best_local_f - 1e-10:
                    best_local_f = values[0]
                    best_local_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = do_eval(x_ref)
                local_evals += 1

                if f_ref < values[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = do_eval(x_exp)
                    local_evals += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        values[-1] = f_ref
                elif f_ref < values[-2]:
                    simplex[-1] = x_ref
                    values[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = do_eval(x_con)
                    local_evals += 1
                    if f_con < values[-1]:
                        simplex[-1] = x_con
                        values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            values[i] = do_eval(simplex[i])
                            local_evals += 1
                            if eval_count >= budget or time_left() < 2.0:
                                return

            return best_local_x, best_local_f

        # Phase 1: Coarse global search - sample many points
        rng = np.random.default_rng(42)
        n_global = min(200, budget // 3)
        for i in range(n_global):
            if eval_count >= budget or time_left() < 10.0:
                break
            x = rng.uniform(-5, 5, dim)
            do_eval(x)

        # Phase 2: For the best few points, do Rastrigin-aware local search
        # Re-evaluate a small set of diverse points to find good basins
        candidates = []
        for i in range(min(20, budget - eval_count)):
            if eval_count >= budget or time_left() < 10.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = do_eval(x)
            candidates.append((fx, x))

        # Sort by fitness and take top few
        candidates.sort(key=lambda t: t[0])
        top_candidates = [x for _, x in candidates[:5]]

        # Add the overall best
        if best_x not in top_candidates:
            top_candidates.append(best_x.copy())

        # Phase 3: Local refinement on each candidate
        for i, x0 in enumerate(top_candidates):
            if eval_count >= budget or time_left() < 5.0:
                break
            remaining = budget - eval_count
            max_local = min(remaining, max(30, budget // (len(top_candidates) + 1)))
            rastrigin_local(x0, max_local, seed=100 + i)

        # Phase 4: Final Nelder-Mead refinement on best
        if eval_count < budget and time_left() > 5.0:
            remaining = budget - eval_count
            nelder_mead(best_x, min(remaining, 100), seed=200)

        # Phase 5: One more Rastrigin snap attempt
        if eval_count < budget and time_left() > 3.0:
            pi2 = 2 * math.pi
            x_snapped = np.round(best_x / pi2) * pi2
            x_snapped = np.clip(x_snapped, -5, 5)
            do_eval(x_snapped)

        return best_x, best_f
