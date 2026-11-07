"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function minimizer: solve for gradient, then go to optimal vertex."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Step 1: Determine gradient w using d+1 evaluations
        # Evaluate at origin and at 5*e_i for each axis
        x0 = np.zeros(dim)
        f0 = func(x0)

        w = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            xi = np.zeros(dim)
            xi[i] = 5.0
            fi = func(xi)
            w[i] = (fi - f0) / 5.0

        # Step 2: The optimal vertex for a linear function over [-5,5]^d
        # is at x_i = -5*sign(w_i)
        x_opt = -5.0 * np.sign(w)
        # For zero components, 0 is fine (sign(0)=0, so x_i=0)
        f_opt = func(x_opt)

        # Step 3: If we have budget left, we can try to refine
        # but for a truly linear function, x_opt is already optimal.
        # However, let's use remaining budget to potentially improve
        # in case of any numerical issues or if the function isn't perfectly linear.
        
        # Actually, for a linear function, the vertex IS the optimum.
        # Let's just verify by checking a few nearby points if budget allows.
        
        # But the simplest and most robust approach: since we've identified
        # the exact gradient, the optimal point is the vertex.
        # Use remaining budget for a local search around x_opt just in case.
        
        remaining = budget - (dim + 2)  # used dim+1 for gradient, 1 for x_opt
        if remaining > 0 and time.time() < deadline:
            # Simple local refinement: try small perturbations
            best_x = x_opt.copy()
            best_f = f_opt
            rng = np.random.default_rng(42)
            sigma = 0.1
            for _ in range(min(remaining, 100)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x, best_f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
            x_opt, f_opt = best_x, best_f

        return x_opt, f_opt
# EVOLVE-BLOCK-END
