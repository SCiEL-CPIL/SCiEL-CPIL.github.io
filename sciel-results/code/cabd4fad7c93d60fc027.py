"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based multi-start optimizer for BBOB f9 (rotated Rosenbrock) in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        evals = 0

        def evaluate(x):
            """Evaluate function and track best."""
            nonlocal best_f, best_x, evals
            x = np.clip(x, -5, 5)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def gradient_descent(x0, max_iters):
            """Variable-step gradient descent with central differences and backtracking."""
            nonlocal best_f, best_x, evals

            x = x0.copy()
            f = evaluate(x)

            # Initial step size
            step = 0.5
            min_step = 1e-8
            max_step = 4.0

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if evals >= budget:
                    break

                # Compute gradient via central differences
                if evals + 2 * n > budget:
                    break

                grad = np.zeros(n)
                for i in range(n):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += step * 0.01
                    x_minus[i] -= step * 0.01
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    evals += 2
                    grad[i] = (f_plus - f_minus) / (2 * step * 0.01)

                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-12:
                    break

                # Normalize gradient direction
                direction = -grad / grad_norm

                # Backtracking line search
                alpha = step
                c1 = 1e-4  # Armijo condition
                rho = 0.5

                # Try current step
                x_trial = x + alpha * direction
                x_trial = np.clip(x_trial, -5, 5)
                if evals >= budget:
                    break
                f_trial = func(x_trial)
                evals += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()

                # Armijo condition: f_trial <= f + c1 * alpha * grad_norm * (-1)
                # i.e., f_trial <= f - c1 * alpha * grad_norm
                if f_trial <= f - c1 * alpha * grad_norm:
                    # Accept step
                    x = x_trial
                    f = f_trial
                    # Increase step slightly if we made good progress
                    if f_trial < f - 0.5 * c1 * alpha * grad_norm:
                        step = min(step * 1.1, max_step)
                else:
                    # Backtrack
                    while alpha > min_step:
                        alpha *= rho
                        x_trial = x + alpha * direction
                        x_trial = np.clip(x_trial, -5, 5)
                        if evals >= budget:
                            break
                        f_trial = func(x_trial)
                        evals += 1
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()
                        if f_trial <= f - c1 * alpha * grad_norm:
                            x = x_trial
                            f = f_trial
                            break
                    else:
                        # Couldn't find a good step, reduce step size
                        step *= 0.5
                        if step < min_step:
                            break

            return x, f

        def nelder_mead(func, x0, budget_remaining, deadline, rng):
            """Run Nelder-Mead from x0 with given budget."""
            nonlocal best_f, best_x, evals

            # Initialize simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i, i - 1] = 0.1

            # Evaluate simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                if evals >= budget:
                    break
                simplex[i] = np.clip(simplex[i], -5, 5)
                fvals[i] = func(simplex[i])
                evals += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho_c = 0.5
            sigma = 0.5
            max_iter = budget_remaining // (n + 1) + 10
            no_improve = 0
            threshold = 1e-10

            for it in range(max_iter):
                if time.time() > deadline:
                    break
                if evals >= budget:
                    break

                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                if np.max(fvals) - np.min(fvals) < threshold:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                if evals >= budget:
                    break
                f_ref = func(x_ref)
                evals += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals[0]:
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if evals >= budget:
                        break
                    f_exp = func(x_exp)
                    evals += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
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
                    if f_ref < fvals[-1]:
                        x_con = centroid + rho_c * (x_ref - centroid)
                    else:
                        x_con = centroid - rho_c * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    if evals >= budget:
                        break
                    f_con = func(x_con)
                    evals += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if evals >= budget:
                                break
                        if evals >= budget:
                            break
                        for i in range(1, n + 1):
                            fvals[i] = func(simplex[i])
                            evals += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                if no_improve > 20:
                    break
                no_improve += 1

            return best_f

        # Multi-start strategy
        start_points = []

        # Origin
        start_points.append(np.zeros(n))

        # Random points
        for _ in range(10):
            pt = rng.uniform(-4, 4, size=n)
            start_points.append(pt)

        # Points along axes
        for axis in range(n):
            for val in [2.0, -2.0, 3.0, -3.0]:
                pt = np.zeros(n)
                pt[axis] = val
                start_points.append(pt)

        # Shuffle
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx]

        # Evaluate all starts
        start_fvals = []
        for pt in start_points:
            if evals >= budget:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            evals += 1
            start_fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        if len(start_fvals) < len(start_points):
            start_points = start_points[:len(start_fvals)]
            start_fvals = start_fvals[:len(start_fvals)]

        if len(start_fvals) > 0:
            order = np.argsort(start_fvals)
            start_points = [start_points[i] for i in order]
            start_fvals = [start_fvals[i] for i in order]

        # Allocate remaining budget
        remaining = budget - evals
        if remaining > 0:
            num_starts = len(start_points)
            if num_starts > 0:
                # Use gradient descent on top starts, Nelder-Mead on others
                # Top 3 get gradient descent, rest get Nelder-Mead
                gd_count = min(3, num_starts)
                nm_count = num_starts - gd_count

                # Allocate budget: gradient descent gets ~60% if available
                gd_budget = int(remaining * 0.6)
                nm_budget = remaining - gd_budget

                # Run gradient descent on best starts
                for i in range(gd_count):
                    if evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    if start_fvals[i] > best_f + 500:
                        continue
                    max_iters = max(1, gd_budget // (2 * n + 1))
                    gradient_descent(start_points[i], max_iters)

                # Run Nelder-Mead on remaining starts
                if nm_count > 0 and nm_budget > 0:
                    weights = np.array([1.0 / (i + 1) for i in range(nm_count)])
                    weights /= weights.sum()
                    budgets = np.maximum(1, (weights * nm_budget).astype(int))
                    while budgets.sum() > nm_budget:
                        budgets[np.argmax(budgets)] -= 1
                    while budgets.sum() < nm_budget and budgets.max() > 0:
                        budgets[np.argmin(budgets)] += 1

                    for i in range(gd_count, num_starts):
                        if evals >= budget:
                            break
                        if time.time() > deadline:
                            break
                        if budgets[i - gd_count] <= 0:
                            continue
                        if start_fvals[i] > best_f + 1000:
                            continue
                        nelder_mead(func, start_points[i], budgets[i - gd_count], deadline, rng)

        return best_x, best_f
# EVOLVE-BLOCK-END
