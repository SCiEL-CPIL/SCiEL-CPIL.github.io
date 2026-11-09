"""Adaptive coordinate descent for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive coordinate descent exploiting separable ridge structure."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        # Track calls
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return func(x)

        # Initialize at center
        x = np.zeros(d)
        fx = wrapped(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step sizes per dimension
        steps = np.full(d, 3.0)  # Start large to explore quickly
        # Track recent success rate per dimension for step adaptation
        success_count = np.zeros(d, dtype=int)
        total_count = np.zeros(d, dtype=int)

        # Coordinate descent: cycle through dimensions
        # Each visit to a dimension: try +step and -step (2 evals)
        # If improvement, keep and grow step slightly
        # If no improvement, shrink step

        dim = d
        call_count = 1  # initial evaluation

        # We'll do multiple phases with different step scales
        # Phase 1: Large steps (3.0) - explore
        # Phase 2: Medium steps (1.0) - converge
        # Phase 3: Small steps (0.1) - refine
        # Phase 4: Tiny steps (0.01) - polish

        # But adaptive: let the step size adapt naturally

        # Main loop: cycle through dimensions until budget exhausted
        dim_idx = 0
        consecutive_no_improve = 0
        max_consecutive_no_improve = 50  # reset if stuck

        while calls[0] < budget and time.time() < deadline:
            # Check if we're stuck
            if consecutive_no_improve > max_consecutive_no_improve:
                # Perturb: try a random direction
                if calls[0] + 1 < budget:
                    random_dir = np.random.randn(d)
                    random_dir /= np.linalg.norm(random_dir)
                    for scale in [1.0, 0.5, 0.25]:
                        if calls[0] + 1 >= budget or time.time() >= deadline:
                            break
                        test_x = x + scale * random_dir
                        test_x = np.clip(test_x, -5, 5)
                        test_fx = wrapped(test_x)
                        if test_fx < best_fx:
                            best_fx = test_fx
                            best_x = test_x.copy()
                            x = test_x.copy()
                            consecutive_no_improve = 0
                            break
                    else:
                        # No improvement from perturbation, continue
                        pass

            # Pick next dimension
            dim_i = dim_idx % dim
            dim_idx += 1

            # Try +step in this dimension
            step_i = steps[dim_i]
            improved = False

            # Try positive direction
            if calls[0] + 1 < budget and time.time() < deadline:
                test_x = x.copy()
                test_x[dim_i] += step_i
                test_x = np.clip(test_x, -5, 5)
                test_fx = wrapped(test_x)

                if test_fx < fx:
                    # Improvement: take it and grow step
                    x = test_x
                    fx = test_fx
                    steps[dim_i] *= 1.1  # grow step
                    steps[dim_i] = min(steps[dim_i], 4.0)  # cap
                    improved = True
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        consecutive_no_improve = 0
                    else:
                        consecutive_no_improve += 1
                else:
                    consecutive_no_improve += 1

            # Try negative direction
            if calls[0] + 1 < budget and time.time() < deadline:
                test_x = x.copy()
                test_x[dim_i] -= step_i
                test_x = np.clip(test_x, -5, 5)
                test_fx = wrapped(test_x)

                if test_fx < fx:
                    # Improvement: take it and grow step
                    x = test_x
                    fx = test_fx
                    steps[dim_i] *= 1.1  # grow step
                    steps[dim_i] = min(steps[dim_i], 4.0)  # cap
                    improved = True
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        consecutive_no_improve = 0
                    else:
                        consecutive_no_improve += 1
                else:
                    consecutive_no_improve += 1

            # If no improvement in either direction, shrink step
            if not improved:
                steps[dim_i] *= 0.8  # shrink step
                steps[dim_i] = max(steps[dim_i], 1e-6)  # floor

        # Final polish: if we have budget left, do a few more very fine steps
        remaining = budget - calls[0]
        if remaining > 2 and time.time() < deadline:
            # Do a few more cycles with tiny steps
            fine_steps = steps * 0.1
            for _ in range(min(remaining // 2, 20)):
                if calls[0] + 2 >= budget or time.time() >= deadline:
                    break
                for dim_i in range(dim):
                    if calls[0] + 2 >= budget or time.time() >= deadline:
                        break
                    step_i = fine_steps[dim_i]
                    # Try +
                    test_x = x.copy()
                    test_x[dim_i] += step_i
                    test_x = np.clip(test_x, -5, 5)
                    test_fx = wrapped(test_x)
                    if test_fx < fx:
                        x = test_x
                        fx = test_fx
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    # Try -
                    if calls[0] + 1 >= budget or time.time() >= deadline:
                        break
                    test_x = x.copy()
                    test_x[dim_i] -= step_i
                    test_x = np.clip(test_x, -5, 5)
                    test_fx = wrapped(test_x)
                    if test_fx < fx:
                        x = test_x
                        fx = test_fx
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
