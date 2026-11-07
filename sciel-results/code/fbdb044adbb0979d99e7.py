"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with adaptive step size and coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Adaptive step size per dimension (Bent cigar has one easy dim, rest hard)
        # Start with a moderate step
        sigma = np.ones(dim) * 1.0

        # For Bent cigar: dimension 0 has curvature 1, others have curvature 100
        # So we should have smaller steps in dimensions 1..dim-1
        # But we don't know the rotation, so use adaptive approach

        success_rate = 0.5
        no_improve = 0
        max_no_improve = 50

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Generate perturbation
            perturbation = rng.standard_normal(dim) * sigma
            y = np.clip(x + perturbation, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success: move to y
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                # Adapt step size: increase slightly on success
                sigma *= 1.1
                # Cap sigma
                sigma = np.minimum(sigma, 5.0)
            else:
                # Failure: decrease step size
                sigma *= 0.9
                sigma = np.maximum(sigma, 1e-6)
                no_improve += 1

            # If stuck for too long, try a restart from best point with different sigma
            if no_improve > max_no_improve:
                x = best_x.copy()
                fx = best_fx
                sigma = np.ones(dim) * 0.5 * np.random.uniform(0.5, 2.0)
                no_improve = 0

        # Coordinate descent polish in remaining budget
        remaining = self.budget - calls
        if remaining > 10 and time.time() < deadline:
            # Use coordinate descent to refine
            for _ in range(min(remaining - 5, 200)):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline:
                        break
                    # Try small steps in each coordinate
                    for step in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]:
                        y = best_x.copy()
                        y[d] = np.clip(best_x[d] + step, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < best_fx:
                            best_x = y
                            best_fx = fy
                            improved = True
                            break
                        y[d] = np.clip(best_x[d] - step, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < best_fx:
                            best_x = y
                            best_fx = fy
                            improved = True
                            break
                    if calls >= self.budget:
                        break
                if not improved:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
