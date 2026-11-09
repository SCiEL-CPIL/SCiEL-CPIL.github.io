"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead simplex ensemble for Rastrigin-style minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # safety margin before 300s limit

        # Track best value and point
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Generate diverse starting points: center + corners + random
        rng = np.random.default_rng(42)
        start_points = [np.zeros(dim)]  # center

        # Add corners of the box
        for mask in range(2 ** dim):
            corner = np.array([(-5.0 if (mask >> i) & 1 else 5.0) for i in range(dim)])
            start_points.append(corner)

        # Add random points
        for _ in range(8):
            start_points.append(rng.uniform(-5, 5, dim))

        # Shuffle start points for diversity
        rng.shuffle(start_points)

        # Run Nelder-Mead from multiple starting points
        # Each simplex gets a portion of the budget
        n_starts = min(len(start_points), 12)
        budget_per_start = max(10, (budget - n_starts * (dim + 1)) // n_starts)

        for start_idx in range(n_starts):
            if calls_made >= budget - 1:
                break
            if time.time() > deadline:
                break

            x0 = start_points[start_idx].copy()
            f0 = evaluate(x0)

            # Initialize simplex: x0 plus dim perturbed points
            simplex = [x0.copy()]
            simplex_f = [f0]

            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(xi)
                simplex_f.append(evaluate(xi))

            if calls_made >= budget - 1:
                break
            if time.time() > deadline:
                break

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_iter = budget_per_start
            for it in range(max_iter):
                if calls_made >= budget - 1:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[-1] - simplex_f[0]) < 1e-10:
                    break

                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < simplex_f[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    if f_c < min(f_r, simplex_f[-1]):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_f[i] = evaluate(simplex[i])

        return best_x, best_f
# EVOLVE-BLOCK-END
