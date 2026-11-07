"""Rewritten program: Powell's method with Nelder-Mead polish for BBOB f22."""
import numpy as np


class Optimizer:
    """Deterministic gradient-free trust-region method using Powell's conjugate
    direction search with Nelder-Mead polish for rapid local exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        calls_made = 0

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return float(func(np.asarray(x, dtype=float)))

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # ---- Phase 1: Coarse multi-start with small Nelder-Mead ----
        rng = np.random.default_rng(12345)

        # Choose a few starting points
        start_points = [np.zeros(dim)]
        for _ in range(4):
            start_points.append(rng.uniform(-4, 4, dim))

        nm_budget_per_start = max(30, budget // (len(start_points) + 2))

        for sp in start_points:
            if calls_made >= budget:
                break
            f_sp = wrapped_func(sp)
            update_best(sp, f_sp)
            if calls_made >= budget:
                break

            # Small Nelder-Mead to get a local minimum
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = sp.copy()
            for i in range(dim):
                simplex[i + 1] = sp.copy()
                simplex[i + 1][i] += 0.3

            f_vals = np.zeros(dim + 1)
            f_vals[0] = f_sp
            for i in range(1, dim + 1):
                f_vals[i] = wrapped_func(simplex[i])
                update_best(simplex[i], f_vals[i])
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for _ in range(nm_budget_per_start):
                if calls_made >= budget:
                    break

                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]

                if f_vals[-1] - f_vals[0] < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = wrapped_func(x_r)
                update_best(x_r, f_r)
                if calls_made >= budget:
                    break

                if f_vals[0] <= f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                elif f_r < f_vals[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = wrapped_func(x_e)
                    update_best(x_e, f_e)
                    if calls_made >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                else:
                    if f_r < f_vals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = wrapped_func(x_c)
                    update_best(x_c, f_c)
                    if calls_made >= budget:
                        break
                    if f_c < min(f_r, f_vals[-1]):
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_vals[i] = wrapped_func(simplex[i])
                            update_best(simplex[i], f_vals[i])
                            if calls_made >= budget:
                                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        # ---- Phase 2: Powell's conjugate direction method ----
        def powell_method(x0, f0, max_calls):
            """Powell's method with adaptive direction updates."""
            x = np.array(x0, dtype=float)
            f = f0
            # Initial directions: standard basis
            dirs = [np.eye(dim, dtype=float)[i].copy() for i in range(dim)]
            n_powell = 0

            while n_powell < max_calls:
                if calls_made >= budget:
                    break

                x_old = x.copy()
                f_old = f
                min_decrease = 0.0
                min_dir_idx = -1

                # Line search along each direction
                for i in range(dim):
                    if calls_made >= budget:
                        break
                    d = dirs[i]
                    # Find optimal step by golden section or simple line search
                    # Use a simple adaptive line search
                    # First, try a small step in positive and negative direction
                    step = 0.1
                    # Determine direction of descent
                    f_plus = wrapped_func(x + step * d)
                    if calls_made >= budget:
                        break
                    f_minus = wrapped_func(x - step * d)
                    if calls_made >= budget:
                        break

                    if f_plus < f_minus:
                        # Descent in +d
                        f_left = f_minus
                        f_right = f_plus
                        a, b = -step, step
                        # Expand if still decreasing
                        while f_right < f and calls_made < budget:
                            a = b
                            f_left = f_right
                            b = 2 * b
                            f_right = wrapped_func(x + b * d)
                            if not np.isfinite(f_right):
                                b = b / 2
                                f_right = f_left
                                break
                        # Golden section in [a, b]
                        if b > a:
                            gr = (np.sqrt(5) - 1) / 2
                            c = b - gr * (b - a)
                            dpt = a + gr * (b - a)
                            f_c = wrapped_func(x + c * d)
                            f_d = wrapped_func(x + dpt * d)
                            for _ in range(15):
                                if calls_made >= budget:
                                    break
                                if f_c < f_d:
                                    b = dpt
                                    f_b = f_d
                                    dpt = c
                                    f_d = f_c
                                    c = b - gr * (b - a)
                                    f_c = wrapped_func(x + c * d)
                                else:
                                    a = c
                                    f_a = f_c
                                    c = dpt
                                    f_c = f_d
                                    dpt = a + gr * (b - a)
                                    f_d = wrapped_func(x + dpt * d)
                                if abs(b - a) < 1e-10:
                                    break
                            # Best point
                            if f_c < f_d:
                                t_opt = c
                                f_opt = f_c
                            else:
                                t_opt = dpt
                                f_opt = f_d
                        else:
                            t_opt = 0.0
                            f_opt = f
                    else:
                        # Descent in -d
                        f_left = f_plus
                        f_right = f_minus
                        a, b = -step, step
                        while f_right < f and calls_made < budget:
                            b = a
                            f_right = f_left
                            a = 2 * a
                            f_left = wrapped_func(x + a * d)
                            if not np.isfinite(f_left):
                                a = a / 2
                                f_left = f_right
                                break
                        if a < b:
                            gr = (np.sqrt(5) - 1) / 2
                            c = b - gr * (b - a)
                            dpt = a + gr * (b - a)
                            f_c = wrapped_func(x + c * d)
                            f_d = wrapped_func(x + dpt * d)
                            for _ in range(15):
                                if calls_made >= budget:
                                    break
                                if f_c < f_d:
                                    b = dpt
                                    f_b = f_d
                                    dpt = c
                                    f_d = f_c
                                    c = b - gr * (b - a)
                                    f_c = wrapped_func(x + c * d)
                                else:
                                    a = c
                                    f_a = f_c
                                    c = dpt
                                    f_c = f_d
                                    dpt = a + gr * (b - a)
                                    f_d = wrapped_func(x + dpt * d)
                                if abs(b - a) < 1e-10:
                                    break
                            if f_c < f_d:
                                t_opt = c
                                f_opt = f_c
                            else:
                                t_opt = dpt
                                f_opt = f_d
                        else:
                            t_opt = 0.0
                            f_opt = f

                    if f_opt < f:
                        x = x + t_opt * d
                        f = f_opt
                        decrease = f_old - f
                        if decrease > min_decrease:
                            min_decrease = decrease
                            min_dir_idx = i
                    else:
                        f_opt = f  # no change

                if calls_made >= budget:
                    break

                # Update directions
                if min_dir_idx >= 0:
                    # Replace the direction that gave the most decrease with the
                    # direction from x_old to x
                    new_dir = x - x_old
                    # Shift directions: move the best direction to the end
                    dirs = dirs[:min_dir_idx] + dirs[min_dir_idx + 1:] + [new_dir]
                else:
                    # No improvement, stop
                    break

                # Check convergence
                if min_decrease < 1e-12:
                    break

                n_powell += 1

            return x, f

        remaining = budget - calls_made
        if remaining > 2 * (dim + 1):
            x_pw, f_pw = powell_method(best_x.copy(), best_f, remaining)
            if f_pw < best_f:
                best_f = f_pw
                best_x = x_pw.copy()

        # ---- Phase 3: Final Nelder-Mead polish ----
        if calls_made < budget and best_x is not None:
            polish_budget = budget - calls_made
            if polish_budget >= 2 * (dim + 1):
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = best_x.copy()
                for i in range(dim):
                    simplex[i + 1] = best_x.copy()
                    simplex[i + 1][i] += 0.1

                f_vals = np.zeros(dim + 1)
                f_vals[0] = best_f
                for i in range(1, dim + 1):
                    f_vals[i] = wrapped_func(simplex[i])
                    update_best(simplex[i], f_vals[i])
                    if calls_made >= budget:
                        break

                alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                for _ in range(polish_budget):
                    if calls_made >= budget:
                        break

                    order = np.argsort(f_vals)
                    simplex = simplex[order]
                    f_vals = f_vals[order]

                    if f_vals[-1] - f_vals[0] < 1e-14:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    f_r = wrapped_func(x_r)
                    update_best(x_r, f_r)
                    if calls_made >= budget:
                        break

                    if f_vals[0] <= f_r < f_vals[-2]:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    elif f_r < f_vals[0]:
                        x_e = centroid + gamma * (x_r - centroid)
                        f_e = wrapped_func(x_e)
                        update_best(x_e, f_e)
                        if calls_made >= budget:
                            break
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_vals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_vals[-1] = f_r
                    else:
                        if f_r < f_vals[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        f_c = wrapped_func(x_c)
                        update_best(x_c, f_c)
                        if calls_made >= budget:
                            break
                        if f_c < min(f_r, f_vals[-1]):
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_vals[i] = wrapped_func(simplex[i])
                                update_best(simplex[i], f_vals[i])
                                if calls_made >= budget:
                                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
