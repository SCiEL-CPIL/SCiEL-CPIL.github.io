"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: Space-filling exploration + Multi-restart (1+1)-ES.
    
    Strategy:
    1. Phase 1: Use Halton sequence for deterministic space-filling exploration.
       This ensures broad coverage of the search space [-5, 5]^dim.
    2. Phase 2: Multi-restart (1+1) Evolution Strategy.
       - Start from the best point found in Phase 1.
       - If stagnation occurs (no improvement for a while), restart from a
         perturbation of the global best or a new random point.
       - Adaptive step size (sigma) that expands on success and shrinks on failure.
    3. Deadline guard to ensure termination before the 300s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        # Safety margin: 280s out of 300s limit
        deadline = t0 + 280.0

        # Helper for Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        # --- Phase 1: Deterministic Space-Filling Exploration ---
        # Use Halton sequence for low-discrepancy sampling
        n_explore = min(100, budget // 5)  # Use ~20% of budget for exploration
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # Ensure we evaluate the origin and center as well
        candidates = [np.zeros(dim), np.full(dim, 0.0)]
        
        # Generate Halton points
        for i in range(1, n_explore + 1):
            point = np.array([halton(i, bases[d % len(bases)]) * 10.0 - 5.0 for d in range(dim)])
            candidates.append(point)

        # Evaluate exploration candidates
        for pt in candidates:
            if calls_made >= budget:
                break
            if time.time() > deadline - 5.0:  # Leave 5s for Phase 2
                break
            pt = np.clip(pt, -5.0, 5.0)
            try:
                f_val = float(func(pt))
            except Exception:
                continue
            calls_made += 1
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = pt.copy()

        # Fallback if no valid point was found
        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            try:
                best_f = float(func(best_x))
            except Exception:
                best_f = np.inf
            calls_made += 1
            if not np.isfinite(best_f):
                return best_x, best_f

        # --- Phase 2: Multi-Restart (1+1)-ES ---
        # We will run (1+1)-ES with periodic restarts if stagnation is detected.
        
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Initialize local search from current best
            x = best_x.copy()
            fx = best_f
            sigma = 1.0  # Initial step size
            no_improve = 0
            max_no_improve = 25  # Restart after 25 steps without improvement

            # Local (1+1)-ES loop
            while calls_made < budget and no_improve < max_no_improve:
                if time.time() > deadline:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                
                try:
                    fy = float(func(y))
                except Exception:
                    fy = np.inf
                calls_made += 1

                if not np.isfinite(fy):
                    no_improve += 1
                    sigma *= 0.7
                    continue

                if fy < fx:
                    # Improvement
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    # Expand step size on success
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    # No improvement
                    no_improve += 1
                    # Shrink step size on failure
                    sigma = max(sigma * 0.8, 1e-4)

            # If we exited due to stagnation (no_improve >= max_no_improve),
            # and we still have budget, we restart from a new location.
            if calls_made >= budget or time.time() > deadline:
                break
            
            if no_improve >= max_no_improve:
                # Restart strategy:
                # 30% chance: completely random restart
                # 70% chance: perturbation of the global best
                if rng.random() < 0.3:
                    best_x = rng.uniform(-5, 5, dim)
                else:
                    # Perturb global best with a larger step to escape local minima
                    step = 2.0 * rng.standard_normal(dim)
                    best_x = np.clip(best_x + step, -5.0, 5.0)
                
                # Evaluate the new restart point
                try:
                    f_restart = float(func(best_x))
                except Exception:
                    f_restart = np.inf
                calls_made += 1
                
                if np.isfinite(f_restart) and f_restart < best_f:
                    best_f = f_restart
                    # best_x is already updated

        return best_x, best_f
# EVOLVE-BLOCK-END
