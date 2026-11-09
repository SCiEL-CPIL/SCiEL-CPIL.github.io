"""Optimizer for BBOB-style f23 (Katsuura) using LHS initialization and (1+1)-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + (1+1)-ES exploitation with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        # Safety margin: The evaluator kills at 330s. We should return well before.
        # However, the function evaluations are fast. The main constraint is the budget (500 calls).
        # We use a generous deadline to avoid any risk, but effectively we are budget-limited.
        deadline = t0 + 290.0 

        # ---------- Phase 1: Latin Hypercube exploration ----------
        # Use a significant portion of the budget for exploration to find good basins early.
        # Katsuura has many local minima. LHS ensures we sample the box evenly.
        n_lhs = max(20, min(60, budget // 10))  # ~10-20% of budget
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility in tests, though evaluator might vary
        
        # Generate LHS points
        # For each dimension, we have n_lhs intervals. We pick one random point in each interval.
        # This ensures one point per "slice" in each dimension.
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create a permutation of indices 0..n_lhs-1
            perm = rng.permutation(n_lhs)
            # Add a uniform random offset within the interval [0, 1)
            offsets = rng.random(n_lhs)
            # Map to [0, 1) range: (perm + offset) / n_lhs
            normalized = (perm + offsets) / n_lhs
            # Map to [-5, 5]
            lhs_points[:, d] = normalized * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        for i in range(n_lhs):
            if calls >= budget or time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback if something went wrong (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # ---------- Phase 2: (1+1)-ES with periodic restarts ----------
        # We continue from the best point found in Phase 1.
        # We use a (1+1)-ES with adaptive step size and periodic restarts to escape local minima.
        
        x = best_x.copy()
        fx = best_f
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        sigma = 1.0  # Initial step size
        sigma_min = 1e-4
        sigma_max = 5.0
        
        # Restart parameters
        # Katsuura is highly multimodal. We need to restart frequently if we stagnate.
        # We track 'no_improve' steps. If we don't improve for a while, we restart.
        # We also do periodic restarts every N steps to ensure exploration.
        restart_interval = 50  # Check for restart every 50 steps
        no_improve = 0
        max_no_improve = 30    # If no improvement for 30 steps, force a restart
        
        step = 0
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            calls += 1
            step += 1
            
            if fy < fx:
                # Success
                x, fx = y, fy
                sigma = min(sigma * 1.2, sigma_max)
                no_improve = 0
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
            else:
                # Failure
                sigma = max(sigma * 0.8, sigma_min)
                no_improve += 1
            
            # Restart logic
            # Condition 1: Stagnation (no improvement for max_no_improve steps)
            # Condition 2: Periodic restart (every restart_interval steps)
            if no_improve >= max_no_improve or (step > 0 and step % restart_interval == 0):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Decide restart strategy
                # 70% chance: Restart near global best (exploitation)
                # 30% chance: Random restart (exploration)
                if rng.random() < 0.7:
                    # Jump to a point near the global best
                    # Use a larger sigma for the jump to ensure we leave the local basin
                    jump_sigma = max(sigma * 5.0, 1.0)
                    x = np.clip(global_best_x + jump_sigma * rng.standard_normal(dim), -5.0, 5.0)
                else:
                    # Full random restart
                    x = rng.uniform(-5.0, 5.0, dim)
                
                # Evaluate the new starting point
                fx = func(x)
                calls += 1
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
                
                # Reset ES parameters for the new basin
                sigma = 1.0
                no_improve = 0
                # Note: We do not reset 'step' or 'calls'

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
