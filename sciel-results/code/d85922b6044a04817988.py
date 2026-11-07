"""Optimizer for BBOB-style f12 (Bent cigar) with quadratic surrogate + adaptive step."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: quadratic surrogate for large steps + fine-tuning ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        # Evaluation counter
        calls = [0]

        def eval(x):
            calls[0] += 1
            return float(func(np.asarray(x, dtype=float)))

        # --- Phase 1: Global exploration with random restarts + local quadratic model ---
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Initial diverse sampling
        n_init = min(20, budget)
        for i in range(n_init):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = eval(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 2: Hooke-Jeeves Pattern Search ---
        # Hooke-Jeeves is effective for anisotropic functions like Bent Cigar
        # because it can take large steps along the valley direction.
        
        def hooke_jeesves(x_start, f_start, delta_init, max_iter):
            x = x_start.copy()
            f = f_start
            delta = delta_init
            min_delta = 1e-5
            stagnation = 0
            max_stagnation = 10
            
            for _ in range(max_iter):
                if calls[0] >= budget or time.time() > deadline:
                    break
                
                # Exploratory phase
                x_old = x.copy()
                f_old = f
                improved = False
                
                for i in range(dim):
                    if calls[0] >= budget or time.time() > deadline:
                        return x, f
                    
                    # Try +delta
                    x_plus = x.copy()
                    x_plus[i] += delta
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = eval(x_plus)
                    
                    if f_plus < f:
                        x = x_plus
                        f = f_plus
                        improved = True
                        # Continue in same direction (pattern move logic inside exploratory)
                        while calls[0] < budget and time.time() < deadline:
                            x_next = x.copy()
                            x_next[i] += delta
                            x_next = np.clip(x_next, -5, 5)
                            f_next = eval(x_next)
                            if f_next < f:
                                x = x_next
                                f = f_next
                            else:
                                break
                    else:
                        # Try -delta
                        x_minus = x.copy()
                        x_minus[i] -= delta
                        x_minus = np.clip(x_minus, -5, 5)
                        f_minus = eval(x_minus)
                        
                        if f_minus < f:
                            x = x_minus
                            f = f_minus
                            improved = True
                            # Continue in same direction
                            while calls[0] < budget and time.time() < deadline:
                                x_next = x.copy()
                                x_next[i] -= delta
                                x_next = np.clip(x_next, -5, 5)
                                f_next = eval(x_next)
                                if f_next < f:
                                    x = x_next
                                    f = f_next
                                else:
                                    break
                
                # Pattern phase
                if improved:
                    # If we moved, take a pattern step in the direction of the net move
                    # The net move is x - x_old
                    # We try to jump to x + (x - x_old) = 2x - x_old
                    x_pattern = 2 * x - x_old
                    x_pattern = np.clip(x_pattern, -5, 5)
                    f_pattern = eval(x_pattern)
                    
                    if f_pattern < f:
                        x = x_pattern
                        f = f_pattern
                        stagnation = 0
                    else:
                        # Pattern step failed, shrink delta
                        delta *= 0.5
                        stagnation += 1
                else:
                    # No improvement in exploratory phase
                    delta *= 0.5
                    stagnation += 1
                
                if stagnation > max_stagnation or delta < min_delta:
                    # Restart or break if stuck
                    if delta < min_delta:
                        break
                    # Random restart within local neighborhood
                    jump = rng.uniform(-delta, delta, dim)
                    x_restart = np.clip(x + jump, -5, 5)
                    f_restart = eval(x_restart)
                    if f_restart < f:
                        x = x_restart
                        f = f_restart
                    delta = min(delta * 2, 1.0) # Reset delta somewhat
                    stagnation = 0
            
            return x, f

        # Run Hooke-Jeeves
        if best_x is not None:
            x_hj, f_hj = hooke_jeesves(best_x, best_f, 1.0, 100)
            if f_hj < best_f:
                best_f = f_hj
                best_x = x_hj.copy()

        # --- Phase 3: Fine-tuning with (1+1)-ES on remaining budget ---
        if best_x is not None and calls[0] < budget and time.time() < deadline:
            sigma_es = 0.05
            x = best_x.copy()
            fx = best_f
            for _ in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = eval(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma_es *= 1.3
                else:
                    sigma_es *= 0.85
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
