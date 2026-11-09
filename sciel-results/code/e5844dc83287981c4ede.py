"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start CMA-ES style optimizer for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        rng = np.random.default_rng(12345)

        # Track best found
        best_x = np.zeros(dim)
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Coarse random sampling to find a good basin ---
        n_scan = max(20, budget // 5)
        # Include origin and some structured points
        candidates = [np.zeros(dim)]
        for i in range(dim):
            for val in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
                c = np.zeros(dim)
                c[i] = val
                candidates.append(c)
        # Random points
        for _ in range(n_scan - len(candidates)):
            if time.time() > deadline:
                break
            candidates.append(rng.uniform(-4, 4, dim))

        for c in candidates:
            if time.time() > deadline:
                break
            eval_point(c)

        # --- Phase 2: Multi-restart local optimization with CMA-ES-like adaptation ---
        # We'll do multiple restarts, each with a local search
        n_restarts = 8
        calls_remaining = budget - len(candidates)
        if calls_remaining <= 0:
            return best_x, best_f

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Choose starting point: best found + random perturbation
            if restart == 0:
                x0 = best_x.copy()
            else:
                # Mix of best and random
                if rng.random() < 0.5:
                    x0 = best_x + rng.normal(0, 1.0, dim)
                else:
                    x0 = rng.uniform(-4, 4, dim)
            x0 = np.clip(x0, -5, 5)

            # Local search: adaptive step size hill climbing with momentum
            x = x0.copy()
            fx = eval_point(x)
            step_size = 0.5
            momentum = np.zeros(dim)
            learning_rate = 0.7
            calls_used = 1

            # Budget for this restart
            max_local_calls = max(10, calls_remaining // n_restarts)

            for it in range(max_local_calls):
                if time.time() > deadline:
                    break
                if calls_used >= budget - len(candidates):
                    break

                # Generate candidate directions
                n_proposals = 3
                best_prop_f = np.inf
                best_prop_x = None

                for p in range(n_proposals):
                    # Mix of random and momentum-based directions
                    if p == 0:
                        # Momentum direction
                        direction = momentum + rng.normal(0, 1, dim)
                    elif p == 1:
                        # Pure random
                        direction = rng.normal(0, 1, dim)
                    else:
                        # Coordinate-wise random
                        direction = rng.normal(0, 1, dim)
                        direction *= rng.choice([0, 1], dim)

                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction = direction / norm

                    y = x + step_size * direction
                    y = np.clip(y, -5, 5)
                    fy = eval_point(y)
                    calls_used += 1

                    if fy < best_prop_f:
                        best_prop_f = fy
                        best_prop_x = y

                if best_prop_x is None:
                    break

                if best_prop_f < fx:
                    # Accept: update momentum
                    momentum = learning_rate * momentum + (1 - learning_rate) * (best_prop_x - x)
                    x = best_prop_x
                    fx = best_prop_f
                    step_size = min(step_size * 1.2, 5.0)
                else:
                    # Reject: shrink step size
                    step_size *= 0.7
                    momentum *= 0.9

                if step_size < 1e-6:
                    break

        # --- Phase 3: Fine-tuning around best point ---
        if time.time() < deadline - 5:
            # Coordinate-wise local search
            x = best_x.copy()
            fx = best_f
            step = 0.1
            for _ in range(50):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    for sign in [1, -1]:
                        y = x.copy()
                        y[i] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_point(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-7:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
