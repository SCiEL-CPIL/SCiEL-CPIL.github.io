"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Nelder-Mead simplex with local Hessian Newton steps for Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Track best point and value
        best_x = None
        best_f = np.inf
        call_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Nelder-Mead simplex method for basin finding ---
        # Initialize simplex
        x0 = np.zeros(dim)
        simplex = [x0.copy()]
        simplex_f = [evaluate(x0)]

        # Create simplex around x0
        delta = 1.0
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            simplex_f.append(evaluate(xi))

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_shrink = 0.5  # shrink

        nm_calls = 0
        nm_max_calls = min(300, budget - 2 * dim - 10)

        while nm_calls < nm_max_calls and call_count < budget and time.time() < deadline:
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in order]
            simplex_f = [simplex_f[i] for i in order]

            if time.time() >= deadline or call_count >= budget:
                break

            # Check convergence
            f_range = simplex_f[-1] - simplex_f[0]
            if f_range < 1e-12 * max(1.0, abs(simplex_f[0])):
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = evaluate(xr)
            nm_calls += 1

            if fr < simplex_f[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = evaluate(xe)
                nm_calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                fc = evaluate(xc)
                nm_calls += 1

                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        simplex_f[i] = evaluate(simplex[i])
                        nm_calls += 1

        # --- Phase 2: Newton's method with local Hessian approximation ---
        # Start from best point found
        x = best_x.copy()
        fx = best_f

        # Hessian fitting interval
        hessian_interval = 50
        newton_calls = 0
        newton_max_calls = min(150, budget - call_count - 10)

        # Estimate Hessian via finite differences
        def estimate_hessian(xc, fc):
            """Estimate Hessian matrix using central finite differences."""
            n = dim
            H = np.zeros((n, n))
            h = 1e-5

            for i in range(n):
                ei = np.zeros(n)
                ei[i] = h
                xp = xc + ei
                xm = xc - ei
                fpp = evaluate(xp)
                fpm = evaluate(xm)
                H[i, i] = (fpp - 2 * fc + fpm) / (h * h)

                for j in range(i + 1, n):
                    ej = np.zeros(n)
                    ej[j] = h
                    xpp = xc + ei + ej
                    xpm = xc + ei - ej
                    xmp = xc - ei + ej
                    xmm = xc - ei - ej
                    fppp = evaluate(xpp)
                    fpmm = evaluate(xpm)
                    fmpm = evaluate(xmp)
                    fmmp = evaluate(xmm)
                    H[i, j] = (fppp - fpmm - fmpm + fmmp) / (4 * h * h)
                    H[j, i] = H[i, j]

            return H

        # Newton iterations
        while newton_calls < newton_max_calls and call_count < budget and time.time() < deadline:
            # Estimate Hessian at current point
            H = estimate_hessian(x, fx)
            newton_calls += dim * dim  # approximate call count for Hessian

            # Check if Hessian is positive definite
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < 1e-10:
                    # Add regularization
                    H += 1e-4 * np.eye(dim)
                    eigvals = np.linalg.eigvalsh(H)
                    if np.min(eigvals) < 1e-10:
                        break  # Can't make progress
            except np.linalg.LinAlgError:
                break

            # Compute gradient via finite differences
            grad = np.zeros(dim)
            h = 1e-5
            for i in range(dim):
                ei = np.zeros(dim)
                ei[i] = h
                grad[i] = (evaluate(x + ei) - evaluate(x - ei)) / (2 * h)
            newton_calls += 2 * dim

            # Newton step: solve H * dx = -grad
            try:
                dx = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                break

            # Limit step size
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * 5.0 / step_norm

            x_new = np.clip(x + dx, -5, 5)
            fx_new = evaluate(x_new)
            newton_calls += 1

            if fx_new < fx:
                x = x_new
                fx = fx_new
            else:
                # Damped Newton step
                damp = 0.5
                for _ in range(10):
                    x_try = np.clip(x + damp * dx, -5, 5)
                    fx_try = evaluate(x_try)
                    newton_calls += 1
                    if fx_try < fx:
                        x = x_try
                        fx = fx_try
                        break
                    damp *= 0.5

            # Check convergence
            if np.linalg.norm(grad) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
