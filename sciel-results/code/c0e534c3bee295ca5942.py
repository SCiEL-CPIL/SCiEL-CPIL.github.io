"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf

        # Phase 1: Nelder-Mead
        # Initialize simplex
        rng = np.random.default_rng(42)
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x, best_f = x0.copy(), f0

        # Create simplex
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            # Standard Nelder-Mead initialization
            if f0 > 0:
                xi[i] = 0.05
            else:
                xi[i] = 0.05
            simplex.append(xi.copy())

        # Evaluate simplex
        fvals = [f0]
        for i in range(1, dim + 1):
            fvals.append(func(simplex[i]))

        # Track calls
        calls_used = dim + 1

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Main Nelder-Mead loop
        max_nm_iterations = 5000
        nm_iter = 0
        for nm_iter in range(max_nm_iterations):
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Check budget
            if calls_used >= budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Check convergence
            if max(fvals) - min(fvals) < 1e-10 * max(1.0, abs(min(fvals))):
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1

                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)

                fc = func(xc)
                calls_used += 1

                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1

                        if calls_used >= budget or time.time() > deadline - 5.0:
                            break
                    if calls_used >= budget or time.time() > deadline - 5.0:
                        break

        # Phase 2: If budget remains, do local refinement with small steps
        if calls_used < budget and time.time() < deadline - 10.0:
            # Use remaining budget for fine-tuning around best_x
            sigma = 0.1
            x = best_x.copy()
            fx = best_f

            for _ in range(budget - calls_used):
                if time.time() > deadline - 2.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.9

                    # If sigma gets too small, try a different direction
                    if sigma < 1e-8:
                        # Try a random jump to escape potential local minima
                        # (though for Sphere this shouldn't happen much)
                        jump = rng.standard_normal(dim) * 0.5
                        y = np.clip(x + jump, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < fx:
                            x, fx = y, fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            sigma = 0.1

        return best_x, best_f
# EVOLVE-BLOCK-END
