"""Rewritten program: gradient-based optimizer for linear slope BBOB f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer exploiting linearity of f5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)

        # Initial point: center of box
        x = np.zeros(d)
        fx = safe_call(x)
        if fx is None:
            return x, fx

        best_x = x.copy()
        best_f = fx

        # Estimate gradient using central finite differences
        # For a linear function, one gradient estimate is sufficient
        eps = 1e-6
        grad = np.zeros(d)
        for i in range(d):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            # Clip to box
            xp = np.clip(xp, -5, 5)
            xm = np.clip(xm, -5, 5)
            fxp = safe_call(xp)
            fxm = safe_call(xm)
            if fxp is None or fxm is None:
                break
            grad[i] = (fxp - fxm) / (2 * eps)
            if time.time() > deadline:
                return best_x, best_f

        # If gradient is essentially zero, we're at a flat region or center
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            # Try a few random points to see if there's any slope
            for _ in range(min(20, budget - calls)):
                xr = rng.uniform(-5, 5, d)
                fxr = safe_call(xr)
                if fxr is None:
                    break
                if fxr < best_f:
                    best_x = xr.copy()
                    best_f = fxr
            return best_x, best_f

        # For linear function: f(x) = grad . x + b
        # The minimum over [-5,5]^d is at x_i = -5 * sign(grad_i) if grad_i != 0
        # But the actual minimizer is in [-4,4]^5, so the linear function
        # may be shifted. Let's use the gradient to determine direction.
        # 
        # Actually, for a linear function, the minimum over the box is at the
        # vertex opposite to the gradient direction. But the problem says the
        # minimizer is in [-4,4]^5, which suggests the function might be
        # clipped or the linear slope has a specific structure.
        #
        # Let's try: take a step in the negative gradient direction.
        # For linear f, if we move by alpha * (-grad), f decreases by alpha * ||grad||^2
        # We want to reach the boundary or the true minimum.
        
        # Strategy: try several step sizes and pick the best
        step_sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
        
        for alpha in step_sizes:
            if calls >= budget or time.time() > deadline:
                break
            x_new = x - alpha * grad / grad_norm * grad_norm  # = x - alpha * grad
            # Actually, for steepest descent on linear, we want to go far.
            # Let's try x - alpha * grad
            x_try = x - alpha * grad
            x_try = np.clip(x_try, -5, 5)
            fx_try = safe_call(x_try)
            if fx_try is None:
                break
            if fx_try < best_f:
                best_f = fx_try
                best_x = x_try.copy()
        
        # Now try to refine: if we're at a boundary, the true minimum might be
        # at a different vertex. Let's evaluate all 2^d = 32 vertices
        if calls + 32 <= budget and time.time() < deadline:
            for mask in range(2 ** d):
                if time.time() > deadline:
                    break
                xv = np.array([(-5 if (mask >> i) & 1 else 5) for i in range(d)])
                fxv = safe_call(xv)
                if fxv is None:
                    break
                if fxv < best_f:
                    best_f = fxv
                    best_x = xv.copy()

        # If we still have budget, do a local refinement around best_x
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            # Nelder-Mead or simple coordinate descent
            for i in range(d):
                if time.time() > deadline:
                    break
                # Try moving in both directions along dimension i
                for delta in [1.0, 0.5, 0.1, 0.01]:
                    if calls >= budget:
                        break
                    for sign in [1, -1]:
                        if calls >= budget:
                            break
                        x_try = best_x.copy()
                        x_try[i] += sign * delta
                        x_try = np.clip(x_try, -5, 5)
                        fx_try = safe_call(x_try)
                        if fx_try is None:
                            break
                        if fx_try < best_f:
                            best_f = fx_try
                            best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
