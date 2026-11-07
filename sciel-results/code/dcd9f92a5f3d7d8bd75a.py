"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        
        # Track success rate for adaptive step size
        successes = 0
        window_size = 10
        recent_outcomes = []
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Adaptive sigma based on success rate (1/5 rule approximation)
            if len(recent_outcomes) >= window_size:
                success_rate = sum(recent_outcomes) / window_size
                if success_rate > 0.3:
                    sigma *= 1.2
                elif success_rate < 0.1:
                    sigma *= 0.8
                else:
                    sigma *= 1.0
                recent_outcomes = []
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                recent_outcomes.append(1)
                successes += 1
            else:
                recent_outcomes.append(0)
            
            # Coordinate descent polish every 50 calls
            if calls % 50 == 0 and calls < self.budget - 50:
                for d in range(self.dim):
                    step = 0.5
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        trial = x.copy()
                        trial[d] += step
                        trial = np.clip(trial, -5, 5)
                        ft = func(trial)
                        calls += 1
                        if ft < fx:
                            x, fx = trial, ft
                            step *= 1.5
                        else:
                            trial = x.copy()
                            trial[d] -= step
                            trial = np.clip(trial, -5, 5)
                            ft = func(trial)
                            calls += 1
                            if ft < fx:
                                x, fx = trial, ft
                                step *= 1.5
                            else:
                                step *= 0.5
                        if step < 0.01:
                            break
                    if calls >= self.budget - 1:
                        break
            if calls >= self.budget - 1:
                break
        
        return x, fx
# EVOLVE-BLOCK-END
