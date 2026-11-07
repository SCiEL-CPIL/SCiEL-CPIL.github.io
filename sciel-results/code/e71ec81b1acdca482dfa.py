"""Optimizer for BBOB f5 (Linear slope) on [-5, 5]^5."""
import numpy as np
import time


class Optimizer:
    """
    Exploits the linearity of the BBOB f5 function.
    f(x) = b0 + b1 * (x - shift)^T * R * c  (linear in x)
    The minimum over a box is at a vertex. We estimate the gradient
    direction and evaluate the optimal vertex early, then refine.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 290  # safety margin before 300s limit
        calls = 0

        def evaluate(x):
            nonlocal calls
            calls += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Phase 1: Estimate gradient via finite differences from center
        # Evaluate center
        x0 = np.zeros(dim)
        f0 = evaluate(x0)

        # Evaluate along each axis (positive direction)
        h = 1.0
        grad = np.zeros(dim)
        for i in range(dim):
            xi = x0.copy()
            xi[i] += h
            fi = evaluate(xi)
            grad[i] = (fi - f0) / h

        # For a linear function, the minimum is at the vertex:
        # x*_i = -5 if grad_i > 0, else 5
        # (since we want to minimize, go opposite to gradient)
        x_vertex = np.where(grad > 0, -5.0, 5.0)
        f_vertex = evaluate(x_vertex)

        # Phase 2: Refine with a local search around the vertex
        # Since the function is linear, the vertex IS the optimum.
        # But due to numerical issues or if our gradient estimate is slightly off,
        # do a few more evaluations to confirm.
        # Actually, for a perfectly linear function, the vertex is exact.
        # Let's use remaining budget to verify and potentially fine-tune
        # in case the function isn't perfectly linear (but f5 is linear slope).

        # Use remaining budget for a simple local refinement
        # Try small perturbations around the vertex
        best_x = x_vertex.copy()
        best_f = f_vertex

        # A few random checks near the vertex to be safe
        rng = np.random.default_rng(42)
        remaining = self.budget - calls
        for _ in range(min(remaining, 50)):
            if time.time() > deadline:
                break
            # Small perturbation
            dx = rng.standard_normal(dim) * 0.1
            x_trial = np.clip(best_x + dx, -5, 5)
            f_trial = evaluate(x_trial)
            if f_trial < best_f:
                best_x = x_trial
                best_f = f_trial

        # If we still have budget, do a more systematic check
        # For linear function, checking a few more vertices or midpoints
        remaining = self.budget - calls
        if remaining > 0:
            # Check the other nearby vertices
            # Actually, let's just do a simple coordinate descent
            for _ in range(min(remaining, 200)):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    # Try moving coordinate i toward the boundary in the minimizing direction
                    if best_x[i] < 4.5:
                        x_trial = best_x.copy()
                        x_trial[i] = min(5.0, best_x[i] + 1.0)
                        f_trial = evaluate(x_trial)
                        if f_trial < best_f:
                            best_x = x_trial
                            best_f = f_trial
                            improved = True
                    elif best_x[i] > -4.5:
                        x_trial = best_x.copy()
                        x_trial[i] = max(-5.0, best_x[i] - 1.0)
                        f_trial = evaluate(x_trial)
                        if f_trial < best_f:
                            best_x = x_trial
                            best_f = f_trial
                            improved = True
                if not improved:
                    break

        return best_x, best_f
