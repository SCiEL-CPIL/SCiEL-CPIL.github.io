"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """LHS initialization + Nelder-Mead refinement for Step ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 280.0  # safety margin before 300s kill
        d = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # Use a reasonable number of points for initial exploration
        n_lhs = min(100, budget // 5)
        if n_lhs < 10:
            n_lhs = 10

        # Generate LHS points in [0, 1]^d
        sampler = qmc.LatinHypercube(d=d, seed=42)
        lhs_samples = sampler.random(n_lhs)  # shape (n_lhs, d)

        # Map to [-5, 5]^d
        lhs_points = lhs_samples * 10.0 - 5.0

        # Evaluate all LHS points
        for i in range(n_lhs):
            if time.time() - start_time > timeout:
                break
            if calls >= budget:
                break
            eval_point(lhs_points[i])

        if best_x is None:
            # Fallback: single random point
            x0 = np.random.default_rng(0).uniform(-5, 5, d)
            eval_point(x0)
            return best_x, best_f

        # Phase 2: Nelder-Mead refinement from the best LHS point
        # Allocate remaining budget to NM
        remaining_budget = budget - calls

        if remaining_budget <= d + 1:
            return best_x, best_f

        # NM simplex size: start with a moderate step
        # For step functions, we want to explore different regions
        x0 = best_x.copy()

        # Create initial simplex
        simplex = [x0.copy()]
        for i in range(d):
            perturbed = x0.copy()
            # Use a step size that covers a reasonable fraction of the box
            step = 2.0
            if i % 2 == 0:
                perturbed[i] += step
            else:
                perturbed[i] -= step
            perturbed = np.clip(perturbed, -5.0, 5.0)
            simplex.append(perturbed)

        # Evaluate initial simplex
        simplex_f = []
        for i, si in enumerate(simplex):
            if time.time() - start_time > timeout:
                break
            if calls >= budget:
                break
            simplex_f.append(eval_point(si))

        # If we couldn't evaluate the full simplex, just return best
        if len(simplex_f) < d + 1:
            return best_x, best_f

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        # Main NM loop
        while calls < budget and time.time() - start_time < timeout:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            x_best = simplex[0]
            f_best = simplex_f[0]

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = eval_point(x_r)

            if f_r < f_best:
                # Expansion
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = eval_point(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - x_worst)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = eval_point(x_c)
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = eval_point(simplex[i])

            # Check for convergence: if the range of function values is tiny
            if len(simplex_f) > 1:
                f_range = max(simplex_f) - min(simplex_f)
                if f_range < 1e-10:
                    break

            # For step functions, if we're stuck, try a larger perturbation
            # Detect stagnation: if worst hasn't changed much, perturb more
            if len(simplex_f) > 1:
                f_range = max(simplex_f) - min(simplex_f)
                if f_range < 1e-8:
                    # Perturb the worst point significantly
                    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
                    x_worst = best_x + rng.normal(0, 1.0, d)
                    x_worst = np.clip(x_worst, -5.0, 5.0)
                    simplex[-1] = x_worst
                    simplex_f[-1] = eval_point(x_worst)

        return best_x, best_f
# EVOLVE-BLOCK-END
