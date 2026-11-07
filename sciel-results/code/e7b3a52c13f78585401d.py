"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, we want to finish well before.
        # 300s is the stated limit, so 290s is a safe deadline.
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        # Phase 1: Global Exploration via Multi-Restart (1+1)-ES
        # Katsuura is multimodal. A single CMA-ES might get stuck.
        # We will run several short (1+1)-ES runs from different random starts.
        
        # Reserve some budget for final local polishing
        # Let's allocate 80% to global search, 20% to local polish
        global_budget = int(budget * 0.8)
        local_budget = budget - global_budget

        # Number of restarts
        # Each restart uses a small fraction of the global budget
        # Let's say 10 restarts, each with ~80 calls
        num_restarts = 10
        calls_per_restart = global_budget // num_restarts
        if calls_per_restart < 10:
            num_restarts = max(1, global_budget // 10)
            calls_per_restart = global_budget // num_restarts

        for r in range(num_restarts):
            if used_calls >= global_budget:
                break
            if time.time() > deadline:
                break

            # Random start for this restart
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            # (1+1)-ES loop for this restart
            for _ in range(calls_per_restart):
                if used_calls >= global_budget:
                    break
                if time.time() > deadline:
                    break

                # Generate offspring
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1

                if fy < fx:
                    # Success: move to offspring, increase sigma slightly
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Adaptive sigma: increase on success
                    sigma *= 1.05
                else:
                    # Failure: decrease sigma
                    sigma *= 0.85

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 1e-4, 10.0)

                # Early termination if sigma is very small (converged locally)
                if sigma < 1e-3:
                    break

        # Phase 2: Local Polish
        # Use a finer-grained search around the best point found
        if best_x is not None and used_calls < budget:
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls
            
            # Simple coordinate descent or small random steps
            # Let's use a small random walk with decreasing step size
            step_sizes = [0.5, 0.1, 0.01, 0.001]
            calls_in_phase = 0
            max_calls_in_phase = remaining
            
            for ss in step_sizes:
                if calls_in_phase >= max_calls_in_phase:
                    break
                if time.time() > deadline:
                    break
                
                improved = False
                # Try a few steps with this size
                for _ in range(max(1, max_calls_in_phase // 4)):
                    if calls_in_phase >= max_calls_in_phase:
                        break
                    if time.time() > deadline:
                        break
                    
                    z = rng.standard_normal(d)
                    y = x + ss * z
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_in_phase += 1
                    used_calls += 1
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        improved = True
                    else:
                        # If no improvement after a few tries, break to next size
                        # But we don't track "few tries" easily here, so just continue
                        pass

                # If no improvement in this step size, we could break early,
                # but the loop structure above handles it by just moving to next size.
                # Actually, the inner loop runs a fixed number of times per size.
                # This is a simple heuristic.

        return best_x, best_f
# EVOLVE-BLOCK-END
