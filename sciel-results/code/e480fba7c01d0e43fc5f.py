"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit linearity: estimate gradient signs, jump to optimal corner."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        # Step 1: Evaluate at origin to get offset
        x0 = np.zeros(d)
        f0 = func(x0)
        calls = 1

        # Step 2: Estimate gradient signs using forward differences
        # Use a step large enough to be above noise but small enough to stay in box
        h = 1.0
        grad_signs = np.zeros(d)
        for i in range(d):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            xi = np.zeros(d)
            xi[i] = h
            fi = func(xi)
            calls += 1
            # f(xi) - f0 = a_i * h  =>  sign(a_i) = sign(f(xi) - f0)
            diff = fi - f0
            if abs(diff) < 1e-12:
                # Try larger step
                xi[i] = 3.0
                fi = func(xi)
                calls += 1
                diff = fi - f0
            grad_signs[i] = 1.0 if diff > 0 else -1.0

        # Step 3: Construct optimal corner
        # For linear f(x) = a·x + b, min on [-5,5]^d is at x_i = -5 if a_i > 0, +5 if a_i < 0
        x_corner = np.where(grad_signs > 0, -5.0, 5.0)
        f_corner = func(x_corner)
        calls += 1

        best_x = x_corner.copy()
        best_f = f_corner

        # Step 4: Verify and potentially refine
        # Check a few other corners to make sure we have the right one
        # Also check the opposite corner
        x_opposite = np.where(grad_signs > 0, 5.0, -5.0)
        if calls < budget and time.time() < deadline:
            f_opp = func(x_opposite)
            calls += 1
            if f_opp < best_f:
                best_x = x_opposite.copy()
                best_f = f_opp

        # For a truly linear function, the corner is optimal.
        # But let's do a few random checks to be safe, and use remaining budget
        # for a local search in case the function isn't perfectly linear or
        # our gradient estimate was wrong.
        
        rng = np.random.default_rng(42)
        remaining = budget - calls
        sigma = 0.5
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            y = np.clip(best_x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < best_f:
                best_x = y.copy()
                best_f = fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                sigma = max(sigma, 0.01)

        return best_x, best_f
# EVOLVE-BLOCK-END
