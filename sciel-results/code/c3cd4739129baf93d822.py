# EVOLVE-BLOCK-START
"""
Nelder-Mead + gradient-based final polish for BBOB Sphere (f1).

Strategy:
1. Use Nelder-Mead simplex method for the bulk of the budget (robust, derivative-free,
   works well on smooth convex functions like Sphere).
2. Once near convergence, switch to a gradient-based line search using finite-difference
   gradients to polish the solution to high precision.
3. Adaptive step sizing in both phases based on improvement.

This keeps the same entry point: Optimizer(budget, dim) and __call__(func).
Returns (best_x, best_f).
"""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # --- Nelder-Mead setup ---
        # Initial simplex: center at origin, small spread
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        best_f = f0
        best_x = x0.copy()

        # Initial simplex points
        simplex = [x0.copy()]
        for i in range(dim):
            pt = x0.copy()
            pt[i] += 0.5
            simplex.append(pt.copy())

        # Evaluate remaining simplex points
        for i in range(1, dim + 1):
            if calls >= budget or time.time() > deadline:
                break
            f_i = func(simplex[i])
            calls += 1
            if f_i < best_f:
                best_f = f_i
                best_x = simplex[i].copy()

        # Sort simplex by function value
        def sort_simplex():
            nonlocal simplex
            idx = np.argsort([func(simplex[i]) if i == 0 else _simplex_f[i] for i in range(len(simplex))])
            # Actually we need stored function values
            pass

        # Store function values for simplex
        simplex_f = [best_f] if best_x.tolist() == simplex[0].tolist() else [f0]
        # Re-evaluate to be safe
        simplex_f = []
        for pt in simplex:
            if calls >= budget or time.time() > deadline:
                break
            simplex_f.append(func(pt))
            calls += 1
            if simplex_f[-1] < best_f:
                best_f = simplex_f[-1]
                best_x = pt.copy()

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink

        nm_iter = 0
        max_nm_iters = 150  # limit NM iterations to leave budget for polish

        while calls < budget and time.time() < deadline and nm_iter < max_nm_iters:
            nm_iter += 1
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            # Check convergence
            f_range = simplex_f[-1] - simplex_f[0]
            if f_range < 1e-12:
                break

            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)

            if calls >= budget or time.time() > deadline:
                break
            f_r = func(x_r)
            calls += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                if calls >= budget or time.time() > deadline:
                    break
                f_e = func(x_e)
                calls += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
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
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                if calls >= budget or time.time() < deadline and time.time() > deadline:
                    break
                if calls >= budget or time.time() > deadline:
                    break
                f_c = func(x_c)
                calls += 1
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls >= budget or time.time() > deadline:
                            break
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
                    if calls >= budget or time.time() > deadline:
                        break

        # --- Gradient-based polish phase ---
        # Use finite-difference gradient with line search
        x = best_x.copy()
        fx = best_f
        h_fd = 1e-4  # finite difference step

        while calls < budget and time.time() < deadline:
            # Estimate gradient
            grad = np.zeros(dim)
            for i in range(dim):
                if calls + 1 >= budget or time.time() > deadline:
                    break
                x_plus = x.copy()
                x_plus[i] += h_fd
                x_plus = np.clip(x_plus, -5, 5)
                f_plus = func(x_plus)
                calls += 1
                grad[i] = (f_plus - fx) / h_fd

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break

            # Line search: try step in -grad direction
            direction = -grad / grad_norm
            step = 0.1
            improved = False
            for _ in range(5):  # up to 5 line search steps
                if calls >= budget or time.time() > deadline:
                    break
                x_new = x + direction * step
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1
                if fx_new < fx - 1e-12:
                    improved = True
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    break
                else:
                    step *= 0.5

            if not improved:
                # Try a small random perturbation to escape
                if calls >= budget or time.time() > deadline:
                    break
                jitter = np.random.default_rng(hash(x.tobytes()) & 0xFFFFFFFF).uniform(-0.01, 0.01, dim)
                x_new = x + jitter
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls += 1
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
