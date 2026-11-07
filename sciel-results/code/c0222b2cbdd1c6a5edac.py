"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exact affine model fitting for BBOB f5 (Linear slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        dim = self.dim
        budget = self.budget
        calls_used = 0

        # Phase 1: Fit exact affine model f(x) = w^T x + b
        # Evaluate at origin and at unit basis vectors scaled to reach boundary
        # Use step = 5.0 so that x_center + step*e_i hits the boundary
        # This gives maximum signal for gradient estimation

        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_used += 1
        update_best(x0, f0)

        if time.time() > deadline:
            return best_x, best_f

        step = 5.0
        A = np.zeros((dim, dim))
        b_vec = np.zeros(dim)

        for i in range(dim):
            if time.time() > deadline:
                break
            xi = x0.copy()
            xi[i] = step
            fi = func(xi)
            calls_used += 1
            update_best(xi, fi)
            # f(xi) - f0 = step * w_i  =>  w_i = (fi - f0) / step
            b_vec[i] = fi - f0
            A[i, i] = step

        # Solve for gradient: w_i = (fi - f0) / step
        w = b_vec / step
        b = f0

        # Compute the optimal vertex of [-5, 5]^d for the linear model
        # f(x) = w^T x + b
        # To minimize, for each dimension i:
        #   if w[i] > 0, set x[i] = -5
        #   if w[i] < 0, set x[i] = +5
        #   if w[i] == 0, any value works; use 0
        x_opt = np.zeros(dim)
        for i in range(dim):
            if w[i] > 0:
                x_opt[i] = -5.0
            elif w[i] < 0:
                x_opt[i] = 5.0
            else:
                x_opt[i] = 0.0

        # Evaluate the computed optimum
        f_opt_val = func(x_opt)
        calls_used += 1
        update_best(x_opt, f_opt_val)

        if time.time() > deadline:
            return best_x, best_f

        # Phase 2: Verify with a second point to check linearity
        # If the function is truly linear, the model prediction should match
        # Evaluate at a random interior point to verify
        rng = np.random.default_rng(123)
        x_check = rng.uniform(-4.0, 4.0, size=dim)
        f_check = func(x_check)
        calls_used += 1
        update_best(x_check, f_check)

        # Check if model is accurate
        f_predicted = w @ x_check + b
        model_error = abs(f_check - f_predicted)

        if time.time() > deadline:
            return best_x, best_f

        # If model error is small, the linear model is accurate and x_opt is optimal
        # If model error is large, fall back to local search
        if model_error < 1e-6:
            # Model is accurate; x_opt should be the global optimum
            # Do a few more evaluations to confirm we haven't missed anything
            # Evaluate at the opposite vertex as a sanity check
            x_opp = -x_opt.copy()
            f_opp = func(x_opp)
            calls_used += 1
            update_best(x_opp, f_opp)

            if time.time() > deadline:
                return best_x, best_f

            # Use remaining budget for a very local (1+1)-ES around x_opt
            # with extremely small step to catch any tiny non-linearities
            x_cur = x_opt.copy()
            f_cur = f_opt_val
            sigma = 1e-4

            remaining = budget - calls_used
            for _ in range(remaining):
                if time.time() > deadline:
                    break

                y = x_cur + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_used += 1
                update_best(y, fy)

                if fy < f_cur:
                    x_cur = y
                    f_cur = fy
                    sigma *= 1.05
                else:
                    sigma *= 0.95

                if sigma < 1e-10:
                    break

        else:
            # Model is not perfectly linear; do a more thorough search
            # Start a (1+1)-ES from x_opt with moderate step
            x_cur = x_opt.copy()
            f_cur = f_opt_val
            sigma = 0.5

            remaining = budget - calls_used
            for _ in range(remaining):
                if time.time() > deadline:
                    break

                y = x_cur + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_used += 1
                update_best(y, fy)

                if fy < f_cur:
                    x_cur = y
                    f_cur = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9

                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
