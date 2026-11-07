"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive optimizer for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return func(x)

        def check_time():
            return calls_made < budget and time.time() < deadline - 5

        def coordinate_descent(x, fx, max_evals=30):
            """Efficient adaptive coordinate descent."""
            nonlocal best_f, best_x
            step = np.ones(dim) * 1.0
            for _ in range(max_evals):
                if not check_time():
                    break
                improved = False
                for d in range(dim):
                    if not check_time():
                        break
                    # Try positive
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] + step[d], -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = safe_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            step[d] = min(step[d] * 1.5, 4.0)
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            continue
                    # Try negative
                    x_trial = x.copy()
                    x_trial[d] = np.clip(x_trial[d] - step[d], -5, 5)
                    if not np.allclose(x_trial, x):
                        f_trial = safe_func(x_trial)
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            step[d] = min(step[d] * 1.5, 4.0)
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                if not improved:
                    step *= 0.5
                    if np.max(step) < 1e-4:
                        break
            return x, fx

        def nelder_mead(x0, fx0, max_evals=40):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_f, best_x
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [fx0]
            for d in range(dim):
                x_new = x0.copy()
                x_new[d] = np.clip(x_new[d] + 0.5, -5, 5)
                simplex.append(x_new)
                fvals.append(safe_func(x_new))
                if not check_time():
                    return x0, fx0

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_simplex_iters = max_evals // (2 * dim + 1)

            for _ in range(max_simplex_iters):
                if not check_time():
                    break
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                centroid = np.clip(centroid, -5, 5)

                # Reflection
                x_ref = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                f_ref = safe_func(x_ref)
                if not check_time():
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = np.clip(centroid + gamma * (x_ref - centroid), -5, 5)
                    f_exp = safe_func(x_exp)
                    if not check_time():
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                    f_con = safe_func(x_con)
                    if not check_time():
                        break
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                            fvals[i] = safe_func(simplex[i])
                            if not check_time():
                                break

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            # Return best point from simplex
            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        # Phase 1: Generate diverse starting points
        n_starts = 10
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))

        # Phase 2: For each start, do coordinate descent then Nelder-Mead
        for s_idx, x0 in enumerate(starts):
            if not check_time():
                break

            x = np.clip(x0.copy(), -5, 5)
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive coordinate descent
            remaining_evals = budget - calls_made
            cd_evals = min(25, max(5, remaining_evals // (n_starts - s_idx + 1)))
            x, fx = coordinate_descent(x, fx, max_evals=cd_evals)

            # Nelder-Mead refinement
            if check_time():
                nm_evals = min(30, budget - calls_made)
                x, fx = nelder_mead(x, fx, max_evals=nm_evals)

        # Phase 3: Final fine-grained polish around the best point
        if best_x is not None and check_time():
            x = best_x.copy()
            fx = best_f
            # Very fine coordinate search
            for scale in [0.1, 0.01, 0.001]:
                if not check_time():
                    break
                step = np.ones(dim) * scale
                for _ in range(5):
                    if not check_time():
                        break
                    improved = False
                    for d in range(dim):
                        if not check_time():
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[d] = np.clip(x_trial[d] + direction * step[d], -5, 5)
                            if not np.allclose(x_trial, x):
                                f_trial = safe_func(x_trial)
                                if f_trial < fx:
                                    x = x_trial
                                    fx = f_trial
                                    improved = True
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                    break
                    if not improved:
                        break

        # Phase 4: If budget remains, do a few more random restarts with quick local search
        if check_time():
            remaining = budget - calls_made
            n_extra = min(remaining // 8, 5)
            for _ in range(n_extra):
                if not check_time():
                    break
                x_trial = rng.uniform(-5, 5, dim)
                f_trial = safe_func(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    # Quick local refinement
                    x = x_trial.copy()
                    fx = f_trial
                    remaining_evals = budget - calls_made
                    if remaining_evals > 10:
                        x, fx = coordinate_descent(x, fx, max_evals=min(15, remaining_evals // 2))
                        if check_time():
                            x, fx = nelder_mead(x, fx, max_evals=min(15, budget - calls_made))

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
