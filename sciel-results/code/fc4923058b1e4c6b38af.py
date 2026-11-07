"""Simulated Annealing for Schwefel f20 with geometric cooling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with geometric cooling for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        total_calls = 1

        # Simulated Annealing parameters
        # Start with a large temperature to explore globally
        # The scale should be related to the expected function value range
        # For Schwefel, values can be large, so start high
        T = 50.0  # initial temperature
        T_min = 1e-4  # minimum temperature
        cooling_rate = 0.9995  # geometric cooling per iteration
        # Total iterations we plan to do (budget - 1)
        # We want T to reach T_min by the end
        # T * cooling_rate^(budget-1) = T_min
        # cooling_rate = (T_min/T)^(1/(budget-1))
        # Let's use the specified cooling_rate and see how it goes
        # Or compute dynamically:
        n_iterations = budget - 1
        if n_iterations > 0:
            cooling_rate = (T_min / T) ** (1.0 / n_iterations)
        else:
            cooling_rate = 0.999

        # Step size for perturbations
        # Start large, maybe decrease slightly or keep constant
        # With SA, the step size can be fixed or adaptive
        # Let's use a step size that's proportional to the range
        step_scale = 2.0

        while total_calls < budget:
            if time.time() >= deadline:
                break

            # Generate a candidate by perturbing the current solution
            # Use Gaussian perturbation
            y = x + step_scale * np.random.randn(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            total_calls += 1

            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

            # Acceptance criterion
            delta_f = fy - fx
            if delta_f < 0:
                # Improvement, always accept
                x = y
                fx = fy
            else:
                # Worse solution, accept with probability exp(-delta_f / T)
                if T > T_min:
                    # Avoid overflow in exp for very large negative delta_f/T
                    # if delta_f is very large positive, delta_f/T is large, exp is 0
                    if delta_f / T < 700:  # exp(-700) is very small
                        prob = np.exp(-delta_f / T)
                    else:
                        prob = 0.0
                    if np.random.random() < prob:
                        x = y
                        fx = fy

            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_x, best_fx
# EVOLVE-BLOCK-END
