"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive (1+1)-ES polish for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        calls = [0]

        def safe_call(x):
            calls[0] += 1
            return func(np.clip(x, -5.0, 5.0))

        def time_ok():
            return (time.time() - t0) < deadline and calls[0] < budget - 5

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start sampling to find promising basins
        n_samples = 12
        samples = []
        for _ in range(n_samples):
            if not time_ok():
                break
            x = rng.uniform(-5.0, 5.0, n)
            fx = safe_call(x)
            samples.append((fx, x.copy()))

        samples.sort(key=lambda t: t[0])
        top_k = 2
        best_fx = samples[0][0]
        best_x = samples[0][1].copy()

        # Phase 2: Nelder-Mead on the top-k most promising starts
        for k in range(min(top_k, len(samples))):
            if not time_ok():
                break
            start_x = samples[k][1].copy()
            start_fx = samples[k][0]

            # Initialize simplex: start point + n perturbations
            simplex = [start_x.copy()]
            simplex_fx = [start_fx]
            alpha = 0.3  # Initial step size for simplex vertices
            for i in range(n):
                v = start_x.copy()
                v[i] += alpha
                v = np.clip(v, -5.0, 5.0)
                fv = safe_call(v)
                simplex.append(v)
                simplex_fx.append(fv)

            # Nelder-Mead iterations
            max_nm_iters = 200
            for _ in range(max_nm_iters):
                if not time_ok():
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_fx)
                simplex = [simplex[i] for i in order]
                simplex_fx = [simplex_fx[i] for i in order]

                # Convergence check
                if abs(simplex_fx[-1] - simplex_fx[0]) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fxr = safe_call(xr)

                if fxr < simplex_fx[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fxe = safe_call(xe)
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_fx[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_fx[-1] = fxr
                elif fxr < simplex_fx[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_fx[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_fx[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5.0, 5.0)
                        fxc = safe_call(xc)
                        if fxc <= fxr:
                            simplex[-1] = xc
                            simplex_fx[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_fx[i] = safe_call(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                        xc = np.clip(xc, -5.0, 5.0)
                        fxc = safe_call(xc)
                        if fxc < simplex_fx[-1]:
                            simplex[-1] = xc
                            simplex_fx[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                simplex_fx[i] = safe_call(simplex[i])

                # Update best
                idx_best = np.argmin(simplex_fx)
                if simplex_fx[idx_best] < best_fx:
                    best_fx = simplex_fx[idx_best]
                    best_x = simplex[idx_best].copy()

        # Phase 3: Adaptive (1+1)-ES polish from the best point
        x = best_x.copy()
        fx = best_fx
        sigma = 0.2
        while time_ok():
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5.0, 5.0)
            fy = safe_call(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.87
            if sigma < 1e-8:
                break

        return x, fx
# EVOLVE-BLOCK-END
