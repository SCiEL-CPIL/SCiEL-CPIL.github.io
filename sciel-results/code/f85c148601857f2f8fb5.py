"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Bilevel (1+1)-ES with adaptive step-size and coordinate descent refinement.
    Strategy: 
    1. Global exploration with (1+1)-ES using a robust step-size adaptation.
    2. Periodic local refinement with coordinate descent to exploit local smoothness.
    3. Single trajectory to maximize budget efficiency.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            return float(func(x))

        def check_deadline():
            return (time.time() - start_time) > deadline

        # --- Phase 1: Initialize with a few random points to find a good start ---
        # Weierstrass f16 is unimodal, but the global minimum is shifted.
        # Start with a few diverse points to ensure we are in the basin of attraction.
        starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.random.default_rng(42).uniform(-3, 3, dim),
            np.random.default_rng(43).uniform(-4, 4, dim),
        ]

        for x0 in starts:
            if check_deadline() or calls_made >= budget:
                break
            x0 = np.clip(x0, -5, 5)
            fx0 = call_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        # --- Phase 2: (1+1)-ES with Adaptive Step-Size and Coordinate Descent ---
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step-size
        rng = np.random.default_rng(123)

        # (1+1)-ES with 1/5-rule step-size adaptation
        # Also interleave coordinate descent every 20 steps to refine locally.
        step_count = 0
        cd_interval = 20  # Do coordinate descent every 20 steps

        while calls_made < budget and not check_deadline():
            step_count += 1

            # (1+1)-ES step
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = call_func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.05  # Increase step-size on success
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.9  # Decrease step-size on failure

            # Periodic coordinate descent refinement
            if step_count % cd_interval == 0:
                # Coordinate descent: try small steps in each coordinate
                for d in range(dim):
                    if check_deadline() or calls_made >= budget:
                        break
                    # Try a few small steps in positive and negative direction
                    for step_size in [0.1, 0.01, 0.001]:
                        y_pos = x.copy()
                        y_pos[d] += step_size
                        y_pos = np.clip(y_pos, -5, 5)
                        fy_pos = call_func(y_pos)
                        if fy_pos < fx:
                            x, fx = y_pos, fy_pos
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break  # Found improvement, move to next coordinate

                        y_neg = x.copy()
                        y_neg[d] -= step_size
                        y_neg = np.clip(y_neg, -5, 5)
                        fy_neg = call_func(y_neg)
                        if fy_neg < fx:
                            x, fx = y_neg, fy_neg
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            break  # Found improvement, move to next coordinate

            # Early stopping if sigma is too small
            if sigma < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
