"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Affine-model optimizer for linear slope functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Determine the affine function f(x) = c + g^T x
        # Use d+1 evaluations to solve for g and c.
        # Evaluate at origin and at d unit-scaled points.
        x_origin = np.zeros(d)
        f_origin = func(x_origin)
        calls = 1
        best_f = f_origin
        best_x = x_origin.copy()

        # Use a moderate step size for gradient estimation
        # Since the function is linear, any step size works, but we use
        # a large one to be robust to any clipping effects.
        step = 4.0

        grad = np.zeros(d)
        for i in range(d):
            if time.time() > deadline:
                break

            x_i = x_origin.copy()
            x_i[i] += step
            f_i = func(x_i)
            calls += 1

            if f_i < best_f:
                best_f = f_i
                best_x = x_i.copy()

            # For a linear function, f(x_i) - f(origin) = g[i] * step
            grad[i] = (f_i - f_origin) / step

        # Phase 2: Compute the optimal vertex
        # For f(x) = c + g^T x on [-5,5]^d, minimum is at vertex where
        # x_i = -5 if g[i] > 0, x_i = 5 if g[i] < 0
        x_vertex = np.zeros(d)
        for i in range(d):
            if grad[i] > 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 5.0

        f_vertex = func(x_vertex)
        calls += 1

        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()

        # Phase 3: Verify and refine with additional evaluations
        # If the function is truly linear, we've found the exact minimum.
        # Use remaining budget to confirm and potentially refine.
        
        # Check a few other vertices to be safe
        # The second-best vertex differs in one coordinate
        for i in range(d):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            x_alt = x_vertex.copy()
            # Flip the i-th coordinate
            if x_vertex[i] == -5.0:
                x_alt[i] = 5.0
            else:
                x_alt[i] = -5.0

            f_alt = func(x_alt)
            calls += 1

            if f_alt < best_f:
                best_f = f_alt
                best_x = x_alt.copy()

        # Phase 4: Use remaining budget for local refinement around best point
        # In case the function isn't perfectly linear or there's numerical noise
        if calls < budget and time.time() < deadline:
            x_curr = best_x.copy()
            sigma = 0.5
            no_improve = 0
            max_no_improve = 20

            while calls < budget and time.time() < deadline:
                y = x_curr + sigma * rng.standard_normal(d)
                y = np.clip(y, -5.0, 5.0)

                fy = func(y)
                calls += 1

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    x_curr = y.copy()
                    no_improve = 0
                    sigma = min(sigma * 1.1, 2.0)
                else:
                    no_improve += 1
                    sigma *= 0.9
                    if sigma < 1e-10:
                        break

                    if no_improve >= max_no_improve:
                        # Try a random restart
                        x_restart = rng.uniform(-5.0, 5.0, size=d)
                        fr = func(x_restart)
                        calls += 1
                        if fr < best_f:
                            best_f = fr
                            best_x = x_restart.copy()
                            x_curr = x_restart.copy()
                            sigma = 1.0
                            no_improve = 0
                        else:
                            no_improve = 0
                            sigma = max(sigma * 0.5, 1e-10)

        return best_x, best_f
# EVOLVE-BLOCK-END
