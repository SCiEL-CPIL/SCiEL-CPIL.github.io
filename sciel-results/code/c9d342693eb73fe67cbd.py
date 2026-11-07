"""Rewritten optimizer for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Nelder-Mead / CMA-like hybrid for linear slope function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill

        d = self.dim
        budget = self.budget

        # Initial points: spread out to estimate gradient
        rng = np.random.default_rng(42)

        # Use a set of initial points to estimate the gradient
        # For a linear function f(x) = a^T x + b, we need d+1 points
        # to estimate a and b.

        # Start with a reasonable initial guess
        x0 = np.zeros(d)
        f0 = func(x0)

        # Estimate gradient using coordinate differences
        # f(x + h*e_i) - f(x) = a_i * h
        h = 0.1
        grad_est = np.zeros(d)
        for i in range(d):
            x_plus = x0.copy()
            x_plus[i] += h
            x_plus = np.clip(x_plus, -5, 5)
            x_minus = x0.copy()
            x_minus[i] -= h
            x_minus = np.clip(x_minus, -5, 5)
            fp = func(x_plus)
            fm = func(x_minus)
            grad_est[i] = (fp - fm) / (2 * h)

        calls_used = 1 + 2 * d  # 1 for x0, 2*d for gradient estimation

        # Now we have an estimate of the gradient. For a linear function,
        # the minimum on the box [-5,5]^d is at the corner where each
        # coordinate is -5 if grad_i > 0, +5 if grad_i < 0, or anywhere
        # if grad_i == 0.

        # But the function might not be purely linear in the original
        # coordinates due to the hidden rotation/shift. Let's use a
        # more robust approach: use the estimated gradient to do
        # gradient descent, but also keep a CMA-ES-like population.

        # Actually, for BBOB f5 (linear slope), the function is:
        # f(x) = f_opt + sum_{i=1}^{d} (x_i - x*_i) * s_i
        # where s_i are the slopes (after rotation/shift).
        # The minimum is at x* (the hidden minimizer).

        # A better approach: use the first d+1 evaluations to fit a
        # linear model, then compute the exact minimizer.

        # Let's redo this more carefully.
        # We'll use d+1 points: origin and d unit-step points.
        # Actually, let's use a more standard approach.

        # Reset and use a cleaner approach
        # Use Nelder-Mead with adaptive step size, plus gradient info

        # Let's use a population-based approach with gradient estimation
        # For efficiency, we'll use a (1+1)-ES with adaptive sigma
        # but initialize with gradient-based step.

        # Better: use the gradient estimate to jump close to the optimum
        # then refine with (1+1)-ES.

        # For linear function, if we know the gradient g, the minimum
        # on [-5,5]^d is at x_i = -5 if g_i > 0, +5 if g_i < 0.
        # But this assumes the gradient is in the original coordinates.
        # Due to rotation, the gradient in original coords points toward
        # the minimizer.

        # Let's try: compute candidate corner from gradient sign
        x_corner = np.where(grad_est > 0, -5.0, 5.0)
        x_corner = np.where(np.abs(grad_est) < 1e-10, 0.0, x_corner)
        f_corner = func(x_corner)
        calls_used += 1

        # Also try the center and a few other points
        candidates = [
            x0,
            x_corner,
            np.clip(x0 - 10 * grad_est, -5, 5),  # step in negative gradient direction
            np.clip(x0 - 5 * grad_est, -5, 5),
            np.clip(x0 - 2 * grad_est, -5, 5),
        ]

        best_x = x0
        best_f = f0
        for cx in candidates[1:]:
            cf = func(cx)
            calls_used += 1
            if cf < best_f:
                best_f = cf
                best_x = cx

        # Now refine with (1+1)-ES using remaining budget
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        sigma = 1.0
        x = best_x.copy()
        fx = best_f

        # Check time
        if time.time() > deadline:
            return best_x, best_f

        for i in range(remaining):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
