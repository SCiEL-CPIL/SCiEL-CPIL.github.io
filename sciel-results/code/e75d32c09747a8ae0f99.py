"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with lattice exploration for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        # Phase 1: Deterministic 3^5 lattice for global coverage (243 points)
        # This ensures we sample every region of the multimodal landscape
        best_x = None
        best_f = np.inf
        calls = 0

        # Create a 3^5 grid over [-5, 5]^5
        # But 3^5 = 243, which fits in our budget
        # Actually, let's use a smarter approach: sample key points first
        # The minimizer is in [-4, 4]^5, so we can focus there

        # Phase 1: Coarse exploration with ~100 points
        # Use a structured grid + random points
        explore_budget = 100
        rng = np.random.default_rng(42)

        # Generate a set of exploration points
        # Mix of structured and random
        points = []

        # Center point
        points.append(np.zeros(n))

        # Axis-aligned points at various distances
        for dist in [1.0, 2.0, 3.0, 4.0]:
            for i in range(n):
                for sign in [-1, 1]:
                    p = np.zeros(n)
                    p[i] = sign * dist
                    points.append(p)

        # Random points in [-4, 4]^5
        while len(points) < explore_budget:
            points.append(rng.uniform(-4, 4, n))

        points = np.array(points[:explore_budget])

        for p in points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            p = np.clip(p, -5, 5)
            f = func(p)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # Phase 2: (1+1)-ES refinement from the best point
        if best_x is not None and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0  # Start with moderate step size

            # Adaptive (1+1)-ES
            while calls < budget:
                if time.time() > deadline:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    best_f = fx
                    best_x = x.copy()
                    # Increase step size on success
                    sigma *= 1.2
                else:
                    # Decrease step size on failure
                    sigma *= 0.8

                # If sigma becomes too small, try a random restart
                if sigma < 0.01:
                    # Try a few random jumps to escape local minima
                    for _ in range(3):
                        if calls >= budget or time.time() > deadline:
                            break
                        jump = x + rng.uniform(-2, 2, n)
                        jump = np.clip(jump, -5, 5)
                        fj = func(jump)
                        calls += 1
                        if fj < fx:
                            x = jump
                            fx = fj
                            best_f = fx
                            best_x = x.copy()
                            sigma = 1.0
                            break
                    else:
                        # No improvement from jumps, try smaller sigma again
                        sigma = 0.01

        return best_x, best_f
# EVOLVE-BLOCK-END
