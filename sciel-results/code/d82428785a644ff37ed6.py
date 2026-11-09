"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with quadratic basin-jumping for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_quadratic(center, points, values):
            """Fit a full quadratic model and return the predicted minimizer."""
            m = len(points)
            if m < n + 1:
                return None

            # Build design matrix for quadratic: [1, d_i, d_i*d_j for i<=j]
            X = np.ones((m, 1 + n + n * (n + 1) // 2))
            for k, (p, v) in enumerate(zip(points, values)):
                d = p - center
                col = 1
                for i in range(n):
                    X[k, col] = d[i]
                    col += 1
                for i in range(n):
                    for j in range(i, n):
                        X[k, col] = d[i] * d[j]
                        col += 1
            y = np.array(values)

            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                return None

            # Extract gradient and Hessian
            g = coeffs[1:n + 1].copy()
            H = np.zeros((n, n))
            idx = n + 1
            for i in range(n):
                for j in range(i, n):
                    hval = coeffs[idx]
                    idx += 1
                    if i == j:
                        H[i, j] = 2.0 * hval
                    else:
                        H[i, j] = hval
                        H[j, i] = hval

            # Check if H is positive definite
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.any(eigvals <= 0):
                    return None
            except:
                return None

            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                except:
                    return None

            # Verify the step reduces the model
            model_center = coeffs[0]
            model_new = coeffs[0] + g @ dx + 0.5 * dx @ H @ dx
            if model_new >= model_center:
                return None

            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            return new_x

        def do_es_around(x, fx, budget_limit, sigma_init=0.5):
            """Local (1+λ)-ES around a point."""
            nonlocal calls_made, best_x, best_f
            sigma = sigma_init
            lam = 4
            no_improve = 0
            max_no_improve = 15

            while calls_made < budget_limit and time.time() < deadline:
                if no_improve >= max_no_improve:
                    break

                offspring = []
                offspring_f = []
                for i in range(lam):
                    if calls_made >= budget_limit or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = np.clip(x + sigma * z, -5, 5)
                    fy = evaluate(y)
                    offspring.append(y)
                    offspring_f.append(fy)

                if not offspring:
                    break

                best_idx = np.argmin(offspring_f)
                best_y = offspring[best_idx]
                best_yf = offspring_f[best_idx]

                if best_yf < fx:
                    x = best_y
                    fx = best_yf
                    no_improve = 0
                    sigma *= 1.15
                else:
                    no_improve += 1
                    sigma *= 0.85

                sigma = np.clip(sigma, 1e-6, 3.0)

            return x, fx

        # Phase 1: Differential Evolution for global exploration
        pop_size = 12
        F = 0.75
        CR = 0.85

        # Initialize population with some structure
        pop = np.zeros((pop_size, n))
        pop[0] = np.zeros(n)
        for i in range(1, pop_size):
            pop[i] = rng.uniform(-5, 5, n)

        pop_f = np.array([evaluate(x) for x in pop])
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE loop
        max_de_calls = int(total_budget * 0.6)
        while calls_made < max_de_calls and time.time() < deadline:
            best_idx = np.argmin(pop_f)
            current_best_f = pop_f[best_idx]

            for i in range(pop_size):
                if calls_made >= max_de_calls or time.time() > deadline:
                    break

                # Select three distinct random indices different from i
                a, b, c = rng.choice(pop_size, size=3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = rng.choice(pop_size, size=3, replace=False)

                # DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover
                trial = np.copy(pop[i])
                j_rand = rng.integers(0, n)
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                trial_f = evaluate(trial)

                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

            best_idx = np.argmin(pop_f)
            if pop_f[best_idx] < best_f:
                best_f = pop_f[best_idx]
                best_x = pop[best_idx].copy()

        # Phase 2: Quadratic basin-jumping around the best point
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # Collect points for quadratic fit around best
            h = 1.0
            center = best_x.copy()
            points = [center]
            values = [best_f]

            for i in range(n):
                if calls_made >= total_budget or time.time() > deadline:
                    break
                x_plus = np.clip(center + h * np.eye(n)[i], -5, 5)
                x_minus = np.clip(center - h * np.eye(n)[i], -5, 5)
                fp = evaluate(x_plus)
                fm = evaluate(x_minus)
                points.extend([x_plus, x_minus])
                values.extend([fp, fm])

            # Fit quadratic and jump
            if len(points) >= n + 1:
                jump_x = fit_quadratic(center, points, values)
                if jump_x is not None:
                    jump_f = evaluate(jump_x)
                    if jump_f < best_f:
                        best_f = jump_f
                        best_x = jump_x.copy()

                        # Do local ES around the new best
                        es_budget = min(total_budget, calls_made + 80)
                        best_x, best_f = do_es_around(best_x, best_f, es_budget, sigma_init=0.3)

        # Phase 3: Final polishing with local ES
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            remaining = total_budget - calls_made
            if remaining > 20:
                best_x, best_f = do_es_around(best_x, best_f, total_budget, sigma_init=0.2)

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
