"""Direct linear solver for BBOB f5 (Linear slope) with progressive corner evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Solve the linear function exactly, evaluating progressive corners for early progress."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        d = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        def try_call(x):
            nonlocal calls
            if calls >= budget or not time_ok():
                return None
            fx = func(x)
            calls += 1
            return fx

        # Phase 1: Progressive linear solve with early corner evaluations
        # f(x) = a·x + b
        # Call 1: f(0) = b
        x0 = np.zeros(d)
        f0 = try_call(x0)
        if f0 is None:
            return x0, 0.0
        b = f0
        best_x = x0.copy()
        best_f = f0

        a = np.zeros(d)
        h = 5.0

        for i in range(d):
            if not time_ok() or calls >= budget:
                break
            # Evaluate f(e_i) to get a_i
            xi = np.zeros(d)
            xi[i] = h
            fi = try_call(xi)
            if fi is None:
                break
            a[i] = (fi - b) / h

            # Compute progressive corner: use known a[0..i], unknown a[j> i] = 0
            x_corner = np.zeros(d)
            for j in range(i + 1):
                if a[j] > 0:
                    x_corner[j] = -5.0
                elif a[j] < 0:
                    x_corner[j] = 5.0
                else:
                    x_corner[j] = 0.0
            # For j > i, x_corner[j] = 0 (unknown gradient, assume neutral)

            f_corner = try_call(x_corner)
            if f_corner is not None and f_corner < best_f:
                best_x = x_corner.copy()
                best_f = f_corner

        # At this point, if all a's are known, best_x is the exact optimum.
        # Verify/evaluate the final exact corner if not already done
        if time_ok() and calls < budget:
            x_exact = np.zeros(d)
            for i in range(d):
                if a[i] > 0:
                    x_exact[i] = -5.0
                elif a[i] < 0:
                    x_exact[i] = 5.0
                else:
                    x_exact[i] = 0.0
            # Only evaluate if different from current best
            if not np.allclose(x_exact, best_x):
                f_exact = try_call(x_exact)
                if f_exact is not None and f_exact < best_f:
                    best_x = x_exact.copy()
                    best_f = f_exact

        # Phase 2: Light polish with remaining budget
        # For a truly linear function, the corner is exact. But do a few
        # exploratory steps in case of any nonlinearity or numerical issues.
        if calls < budget and time_ok():
            rng = np.random.default_rng(42)
            sigma = 0.5
            for _ in range(min(50, budget - calls)):
                if calls >= budget or not time_ok():
                    break
                # Try a few candidates
                for _ in range(min(5, budget - calls)):
                    if calls >= budget or not time_ok():
                        break
                    y = np.clip(best_x + sigma * rng.standard_normal(d), -5, 5)
                    fy = try_call(y)
                    if fy is None:
                        break
                    if fy < best_f:
                        best_x = y.copy()
                        best_f = fy
                        sigma *= 0.7
                    else:
                        sigma *= 0.9
                        sigma = max(sigma, 0.01)

        return best_x, best_f
# EVOLVE-BLOCK-END
