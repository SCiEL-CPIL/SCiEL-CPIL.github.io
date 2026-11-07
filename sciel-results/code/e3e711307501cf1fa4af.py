"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer: estimate gradient sign, jump to best vertex early."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            try:
                return func(x)
            except Exception:
                return None

        def time_ok():
            return time.time() < deadline

        # Step 1: Evaluate at origin to get baseline
        origin = np.zeros(dim)
        f_origin = safe_call(origin)
        if f_origin is None:
            # Fallback: random search
            rng = np.random.default_rng(42)
            for _ in range(budget - calls_made):
                if not time_ok():
                    break
                x = rng.uniform(-5, 5, dim)
                fx = safe_call(x)
                if fx is not None and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        best_f = f_origin
        best_x = origin.copy()

        # Step 2: Determine gradient sign using forward differences from origin.
        # For linear f: f(origin + step*e_i) - f(origin) = step * g_i
        # If positive, g_i > 0, so we want x_i = -5 for minimization.
        # If negative, g_i < 0, so we want x_i = +5.
        step = 1.0  # Larger step for better numerical robustness
        grad_signs = np.zeros(dim)
        for i in range(dim):
            if not time_ok():
                break
            x_plus = origin.copy()
            x_plus[i] += step
            x_plus = np.clip(x_plus, -5, 5)
            fp = safe_call(x_plus)
            if fp is None:
                break
            if fp < best_f:
                best_f = fp
                best_x = x_plus.copy()
            # For linear function, fp - f_origin = step * g_i
            grad_signs[i] = fp - f_origin

        # Step 3: Jump to the vertex determined by gradient signs
        if np.any(np.abs(grad_signs) > 1e-12):
            # x_i = -5 if grad_signs[i] > 0 (g_i > 0), else +5
            vertex = np.where(grad_signs > 0, -5.0, 5.0)
            vertex = np.clip(vertex, -5, 5)
            f_vertex = safe_call(vertex)
            if f_vertex is not None and f_vertex < best_f:
                best_f = f_vertex
                best_x = vertex.copy()
        else:
            # Gradient is ~0; try a few random vertices
            rng = np.random.default_rng(123)
            for _ in range(min(5, budget - calls_made)):
                if not time_ok():
                    break
                v = rng.choice([-5, 5], dim)
                fv = safe_call(v)
                if fv is not None and fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        # Step 4: Verification pass - check a few neighboring vertices to ensure
        # we found the true minimum (guards against near-zero gradient components)
        if calls_made < budget and time_ok():
            rng = np.random.default_rng(77)
            # Generate vertices by flipping a few coordinates of our best vertex
            for _ in range(min(20, budget - calls_made)):
                if not time_ok():
                    break
                # Flip 1-2 random coordinates
                v = best_x.copy()
                n_flips = rng.integers(1, 3)
                for _ in range(n_flips):
                    idx = rng.integers(0, dim)
                    v[idx] = 5.0 if v[idx] < 0 else -5.0
                fv = safe_call(v)
                if fv is not None and fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        # Step 5: If budget remains, do a final sweep of all 32 vertices
        # (only if we haven't already found a very good solution)
        if calls_made < budget and time_ok() and best_f > f_origin - 1e-10:
            # Check remaining vertices we haven't evaluated
            for mask in range(32):
                if calls_made >= budget or not time_ok():
                    break
                v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
                fv = safe_call(v)
                if fv is not None and fv < best_f:
                    best_f = fv
                    best_x = v.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
