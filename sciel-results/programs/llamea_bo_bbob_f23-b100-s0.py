"""Combined optimizer for BBOB f23 (Katsuura): Multi-start Nelder-Mead + Adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive ES refinement for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        # Phase 1: Multi-start Nelder-Mead
        # Katsuura is highly multimodal with fractal structure; use multiple starting points
        # distributed across the box to cover different local optima
        n_starts = min(30, B // 20)  # at least ~20 calls per start for NM
        remaining_budget = B - n_starts * 5

        # Generate spread-out starting points using a lattice + jitter
        rng = np.random.default_rng(12345)

        # Create a grid of starting points
        n_per_dim = max(2, int(round(n_starts ** (1.0 / d))))
        grid_coords = np.linspace(-4.5, 4.5, n_per_dim)

        starts = []
        for idx in range(n_starts):
            if idx < n_per_dim ** d:
                coords = np.unravel_index(idx, [n_per_dim] * d)
                x0 = np.array([grid_coords[c] for c in coords], dtype=np.float64)
            else:
                x0 = rng.uniform(-4.5, 4.5, d)
            x0 = x0 + rng.normal(0, 0.1, d)
            x0 = np.clip(x0, -5.0, 5.0)
            starts.append(x0)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        initial_simplex_size = 0.5

        for start in starts:
            if calls >= B - 80 or time.time() > deadline - 30:
                break

            # Initialize simplex
            x0, f0 = safe_call(start)

            # Build initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            for i in range(d):
                xi = x0.copy()
                xi[i] += initial_simplex_size
                xi = np.clip(xi, -5.0, 5.0)
                xi, fi = safe_call(xi)
                simplex.append(xi)
                simplex_f.append(fi)

            # Nelder-Mead iterations
            max_nm_iters = 100
            for it in range(max_nm_iters):
                if calls >= B - 80 or time.time() > deadline - 30:
                    break

                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                xr, fr = safe_call(xr)

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    xe, fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    xc, fc = safe_call(xc)

                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            xi, fi = safe_call(simplex[i])
                            simplex[i] = xi
                            simplex_f[i] = fi

        # Phase 2: Adaptive (1+1)-ES polish around the best point
        if best_x is not None and calls < B - 10 and time.time() < deadline - 10:
            # Adaptive (1+1)-ES with step-size adaptation
            sigma = 0.1  # initial step size
            x = best_x.copy()
            fx = best_f
            success_counter = 0.0
            chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))  # expected norm of standard normal in d dims

            while calls < B - 5 and time.time() < deadline - 5:
                # Generate candidate
                z = rng.standard_normal(d)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                x_trial, f_trial = safe_call(x_trial)

                # Update
                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    success_counter += 1.0
                else:
                    success_counter -= 1.0

                # Step-size adaptation (1/5 rule)
                sigma *= np.exp(0.2 * (success_counter / 5 - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)
                success_counter *= 0.84  # decay to avoid over-adaptation

        return best_x, best_f
# EVOLVE-BLOCK-END
