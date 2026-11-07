import numpy as np
import time

class Optimizer:
    """
    Optimizer using Nelder-Mead with adaptive restarts for BBOB f13 (Sharp Ridge).
    Nelder-Mead is robust for low-dimensional problems and handles ridges well
    when combined with restarts to escape local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin for 330s limit

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration
        # Sample points to find the basin of attraction.
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            # Fallback
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Nelder-Mead implementation
        def nelder_mead(x0, f0, max_iter=100):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [f0]

            # Create other vertices by perturbing each coordinate
            for i in range(dim):
                x_new = x0.copy()
                x_new[i] += 0.1
                simplex.append(x_new)
                f_val = safe_call(x_new)
                if f_val is None:
                    return
                f_values.append(f_val)

            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            for iteration in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return

                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check for convergence
                if abs(f_values[0] - f_values[-1]) < 1e-10:
                    break

                # Compute centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = safe_call(x_r)
                if f_r is None:
                    return

                if f_values[0] <= f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                elif f_r < f_values[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        return
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                else:
                    # Try contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (simplex[-1] - centroid)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        return

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_val = safe_call(simplex[i])
                            if f_val is None:
                                return
                            f_values[i] = f_val

                # Update global best
                if f_values[0] < best_f:
                    best_f = f_values[0]
                    best_x = simplex[0].copy()

        # Main loop: Run Nelder-Mead with restarts
        while calls < budget and time.time() < deadline:
            if calls >= budget:
                break

            # Run Nelder-Mead from current best
            f_before = best_f
            nelder_mead(best_x, best_f, max_iter=50)

            # If we didn't improve much, try a new random restart
            if best_f > f_before - 1e-6:
                # Restart from a new random point
                if calls < budget - 20:  # Need at least 20 calls for a new simplex
                    x_new = rng.uniform(-5.0, 5.0, size=dim)
                    f_new = safe_call(x_new)
                    if f_new is None:
                        break
                    if f_new < best_f:
                        best_x = x_new
                        best_f = f_new
                    else:
                        # Even if not better, start NM from there?
                        # Or just break if we are running out of budget?
                        # Let's try starting NM from the new point to see if it finds a better basin
                        # But if f_new is worse, it might waste time.
                        # Let's just break if we haven't found a better global best in a restart.
                        # Actually, NM from a worse point might still find the global min if the basin is wide.
                        # But to save budget, let's only restart if we are far from budget limit.
                        pass
            else:
                # We improved, continue with NM from current best
                pass

            # If we are close to budget, do a final NM run
            if budget - calls < 30:
                nelder_mead(best_x, best_f, max_iter=5)
                break

        return best_x, best_f
