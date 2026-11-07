"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-fidelity evolutionary strategy with gradient descent for rotated Rosenbrock (f9)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def estimate_gradient(x, h=None):
            """Estimate gradient using central finite differences with adaptive h."""
            nonlocal calls
            if h is None:
                h = max(1e-5, min(1e-3, 0.01 * max(1.0, abs(best_f))))
            grad = np.zeros(n)
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    return None
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                f_plus = func(x_plus)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    return None
                f_minus = func(x_minus)
                calls += 1
                grad[i] = (f_plus - f_minus) / (2 * h)
            return grad

        def gradient_descent(x0, f0, max_iter=5000, h_start=None):
            """Variable-step gradient descent with backtracking line search."""
            nonlocal calls, best_x, best_f

            x = x0.copy()
            f = f0
            step_size = 0.1
            max_step = 1.0
            min_step = 1e-8
            c1 = 1e-4
            rho = 0.5

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                h = h_start
                if h is None:
                    h = max(1e-5, min(1e-3, 0.01 * max(1.0, abs(f))))
                grad = estimate_gradient(x, h=h)
                if grad is None:
                    break

                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break

                direction = -grad / grad_norm

                alpha = step_size
                f_new = None
                for _ in range(20):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_trial = x + alpha * direction
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = func(x_trial)
                    calls += 1

                    if f_trial < f - c1 * alpha * grad_norm:
                        f_new = f_trial
                        break
                    alpha *= rho

                if f_new is None:
                    step_size *= 0.5
                    if step_size < min_step:
                        break
                    continue

                x = x_trial
                f = f_new

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

                if f < f0 - 0.01 * abs(f0):
                    step_size = min(step_size * 1.5, max_step)
                else:
                    step_size *= 0.8

                f0 = f

            return x, f

        def multi_fidelity_es(x0, f0, max_gen=50, sigma_init=1.0):
            """Multi-fidelity population-based evolutionary strategy with adaptive step size."""
            nonlocal calls, best_x, best_f

            pop_size = 12
            elite_count = 3
            improvement_rate = 1.0
            prev_best = f0

            # Initialize population around x0
            population = [(x0.copy(), f0)]
            for i in range(1, pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                x_try = x0 + sigma_init * rng.standard_normal(n)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1
                population.append((x_try, f_try))
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()

            for gen in range(max_gen):
                if time.time() > deadline or calls >= budget:
                    break

                # Sort population by fitness
                population.sort(key=lambda x: x[1])

                # Update improvement rate
                current_best = population[0][1]
                if prev_best > 1e-10:
                    improvement_rate = max(0.1, min(1.0, 1 - current_best / prev_best))
                prev_best = current_best

                # Adaptive sigma: scale by improvement rate
                sigma = sigma_init * (0.5 + 0.5 * improvement_rate)
                sigma = max(sigma, 0.01)

                # Keep elites, replace rest with new candidates
                elites = population[:elite_count]
                new_population = elites[:]

                for i in range(elite_count, pop_size):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Coarse-to-fine: mix exploration and exploitation
                    if rng.random() < 0.3:
                        # Exploitation: perturb an elite
                        parent_x, parent_f = elites[rng.integers(0, elite_count)]
                        x_try = parent_x + sigma * rng.standard_normal(n)
                    else:
                        # Exploration: perturb a random non-elite or use global restart
                        if rng.random() < 0.5:
                            parent_idx = rng.integers(elite_count, len(population))
                            parent_x, parent_f = population[parent_idx]
                        else:
                            parent_x = rng.uniform(-4, 4, n)
                            parent_f = func(parent_x)
                            calls += 1
                            if calls >= budget or time.time() > deadline:
                                break
                        x_try = parent_x + sigma * rng.standard_normal(n)

                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    calls += 1
                    new_population.append((x_try, f_try))
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()

                population = new_population

                # Convergence check
                if population[-1][1] - population[0][1] < 1e-10:
                    break

            return population[0][0].copy(), population[0][1]

        # Phase 1: Evaluate origin and a few random points
        f_origin = func(np.zeros(n))
        calls += 1
        best_x = np.zeros(n).copy()
        best_f = f_origin

        for i in range(5):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            calls += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-fidelity ES for exploration (use ~50% of remaining budget)
        remaining = budget - calls
        es_budget = int(0.5 * remaining)
        max_gen = max(15, es_budget // 12)
        x_es, f_es = multi_fidelity_es(best_x.copy(), best_f, max_gen=max_gen, sigma_init=1.5)

        if f_es < best_f:
            best_f = f_es
            best_x = x_es.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Gradient descent for fast local convergence
        x_grad, f_grad = gradient_descent(best_x.copy(), best_f, max_iter=2000)

        if f_grad < best_f:
            best_f = f_grad
            best_x = x_grad.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3b: High-precision polish with tighter gradient tolerance
        if budget - calls > 20:
            x_polish, f_polish = gradient_descent(best_x.copy(), best_f, max_iter=500, h_start=1e-6)
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Nelder-Mead for final polish
        def nelder_mead(x0, f0, max_iter=200):
            nonlocal calls, best_x, best_f

            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append(xi)
                fvals.append(fi)

            for it in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break

                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[-1] - fvals[0] < 1e-10:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or time.time() > deadline:
                                return simplex[0], fvals[0]

                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

            min_idx = np.argmin(fvals)
            return simplex[min_idx], fvals[min_idx]

        x_nm, f_nm = nelder_mead(best_x.copy(), best_f, max_iter=200)

        if f_nm < best_f:
            best_f = f_nm
            best_x = x_nm.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4b: Final high-precision gradient polish after Nelder-Mead
        if budget - calls > 10:
            x_final, f_final = gradient_descent(best_x.copy(), best_f, max_iter=200, h_start=1e-7)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 5: If budget remains, do another multi-fidelity ES restart from best
        if budget - calls > 50:
            x_restart = best_x + 0.3 * rng.standard_normal(n)
            x_restart = np.clip(x_restart, -5, 5)
            f_restart = func(x_restart)
            calls += 1
            if f_restart < best_f:
                best_f = f_restart
                best_x = x_restart.copy()

            x_es2, f_es2 = multi_fidelity_es(x_restart, f_restart, max_gen=25, sigma_init=0.5)
            if f_es2 < best_f:
                best_f = f_es2
                best_x = x_es2.copy()

            if calls < budget and time.time() < deadline:
                x_grad2, f_grad2 = gradient_descent(best_x.copy(), best_f, max_iter=500)
                if f_grad2 < best_f:
                    best_f = f_grad2
                    best_x = x_grad2.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
