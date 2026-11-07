"""Combined optimizer for BBOB f12 (Bent cigar) with population-based Hessian estimation."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)

        # --- Initialization: sample several points to find a good starting region ---
        # For Bent Cigar, the valley is narrow in one direction and wide in others.
        # We do a brief random search to find a promising area.
        
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        # Initial random search: 30 evaluations to find a decent starting point
        init_budget = min(30, budget // 10)
        for _ in range(init_budget):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Population-based Hessian estimation with Newton-like steps ---
        # We maintain a population of points around the current best.
        # We estimate the Hessian using finite differences on the population.
        # Then we take a damped Newton step toward the predicted minimum.

        sigma = 1.0  # initial perturbation scale
        sigma_min = 1e-4
        sigma_max = 10.0

        # Population size: enough to estimate Hessian (dim*(dim+1)/2 + 1 = 16 for dim=5)
        # We use a slightly larger population for robustness
        pop_size = 20

        # Generate initial population around best_x
        population = [best_x.copy()]
        pop_f = [best_f]

        # Generate additional population points
        for _ in range(pop_size - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            f = float(func(x))
            calls += 1
            population.append(x.copy())
            pop_f.append(f)

        # Update best if any population point is better
        for i in range(len(population)):
            if pop_f[i] < best_f:
                best_f = pop_f[i]
                best_x = population[i].copy()

        # --- Main loop ---
        for _ in range(budget - calls):
            if calls >= budget or time.time() > deadline:
                break

            # Estimate Hessian from population using finite differences
            # We use the population points to build a local quadratic model
            X = np.array(population)
            F = np.array(pop_f)

            # Fit a quadratic model: f(x) ≈ c + b^T x + x^T A x
            # Design matrix
            n = dim
            m = n * (n + 1) // 2  # number of quadratic terms
            n_params = 1 + n + m
            M = np.ones((len(X), n_params))
            M[:, 1:n + 1] = X
            col = n + 1
            for i in range(n):
                for j in range(i, n):
                    M[:, col] = X[:, i] * X[:, j]
                    col += 1

            # Solve least squares
            try:
                coeffs, _, rank, _ = np.linalg.lstsq(M, F, rcond=None)
            except np.linalg.LinAlgError:
                coeffs = None

            if coeffs is not None and rank >= n_params - 1:
                # Extract gradient and Hessian
                c = coeffs[0]
                b = coeffs[1:1 + n]

                # Reconstruct Hessian A from quadratic coefficients
                A = np.zeros((n, n))
                idx = 1 + n
                for i in range(n):
                    for j in range(i, n):
                        val = coeffs[idx]
                        if i == j:
                            A[i, i] = val
                        else:
                            A[i, j] = val / 2.0
                            A[j, i] = val / 2.0
                        idx += 1

                # Find minimizer of the quadratic model
                # grad = b + 2*A*x = 0 => x* = -0.5 * A^{-1} * b
                try:
                    # Add small regularization for stability
                    A_reg = A + 1e-6 * np.eye(n)
                    x_star = -0.5 * np.linalg.solve(A_reg, b)
                    x_star = np.clip(x_star, -5, 5)

                    # Take a damped step toward x_star
                    step = x_star - best_x
                    step_norm = np.linalg.norm(step)
                    if step_norm > 0:
                        # Limit step size to sigma for stability
                        if step_norm > sigma:
                            step = step / step_norm * sigma
                        x_new = np.clip(best_x + step, -5, 5)
                        f_new = float(func(x_new))
                        calls += 1

                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                            # Increase sigma slightly on success
                            sigma = min(sigma * 1.2, sigma_max)
                        else:
                            # Decrease sigma on failure
                            sigma = max(sigma * 0.7, sigma_min)
                except np.linalg.LinAlgError:
                    pass

            # --- Update population: keep best point, add new random perturbations ---
            # Replace the worst few points in the population with new random perturbations
            # Sort population by function value
            order = np.argsort(pop_f)
            # Keep the best half, replace the worst half
            half = len(population) // 2
            for i in range(half):
                if calls >= budget or time.time() > deadline:
                    break
                # Replace the worst point
                worst_idx = order[-(i + 1)]
                x_new = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                f_new = float(func(x_new))
                calls += 1
                population[worst_idx] = x_new.copy()
                pop_f[worst_idx] = f_new

                # Update best if this new point is better
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            # Re-sort for next iteration
            order = np.argsort(pop_f)
            pop_f = [pop_f[i] for i in order]
            population = [population[i] for i in order]

        return best_x, best_f
