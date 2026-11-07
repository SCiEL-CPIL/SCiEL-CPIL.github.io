"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical multi-scale stochastic search with adaptive per-dim steps and gradient polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Adaptive per-dimension step sizes
        sigma = np.ones(dim) * 2.0

        calls_used = 1

        # Hierarchical multi-scale phases
        # Phase 1: Coarse exploration (large steps)
        # Phase 2: Medium refinement
        # Phase 3: Fine refinement
        # Phase 4: Very fine polish

        # Allocate budget across phases
        phase_budgets = [
            int(budget * 0.30),  # coarse
            int(budget * 0.25),  # medium
            int(budget * 0.25),  # fine
            int(budget * 0.20),  # polish
        ]

        # Scale factors for each phase (order of magnitude reduction)
        phase_scales = [1.0, 0.1, 0.01, 0.001]

        for phase_idx, (phase_budget, scale) in enumerate(zip(phase_budgets, phase_scales)):
            # Reset sigma for this phase based on scale
            sigma = np.ones(dim) * 2.0 * scale

            for _ in range(phase_budget):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    # Update per-dimension sigma adaptively
                    for d in range(dim):
                        if abs(y[d] - x[d]) > 1e-10:
                            sigma[d] *= 1.2  # increase step in this direction
                    x, fx = y, fy
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                else:
                    # Decrease sigma for dimensions that didn't help
                    for d in range(dim):
                        if abs(y[d] - x[d]) > 1e-10:
                            sigma[d] *= 0.85

                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 5.0)

        # Final gradient-based polish using finite differences
        if calls_used < budget and time.time() < deadline:
            # Use central differences for gradient estimation
            eps = 1e-5
            polish_budget = budget - calls_used
            max_polish_iters = min(polish_budget, 200)

            for _ in range(max_polish_iters):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Estimate gradient
                grad = np.zeros(dim)
                for d in range(dim):
                    xp = x.copy()
                    xm = x.copy()
                    xp[d] = min(5.0, xp[d] + eps)
                    xm[d] = max(-5.0, xm[d] - eps)
                    if calls_used + 2 <= budget:
                        fp = func(xp)
                        fm = func(xm)
                        calls_used += 2
                        grad[d] = (fp - fm) / (xp[d] - xm[d])
                    else:
                        break

                if calls_used >= budget:
                    break

                # Check if gradient is small enough
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-8:
                    break

                # Take a step in negative gradient direction
                step_size = 0.1 / (grad_norm + 1e-10)
                y = x - step_size * grad
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                else:
                    # Try smaller step
                    y = x - 0.01 * step_size * grad
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_x, best_f = x.copy(), fx

        return best_x, best_f
# EVOLVE-BLOCK-END
