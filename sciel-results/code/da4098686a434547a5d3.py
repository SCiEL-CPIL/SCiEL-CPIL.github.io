"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with coordinate descent polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim

        # Phase 1: Initial exploration with multiple random points
        best_x = None
        best_fx = np.inf
        n_initial = min(50, self.budget // 10)
        
        for i in range(n_initial):
            if time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: (1+1)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0
        sigma_min = 1e-10
        sigma_max = 10.0
        success_count = 0
        window_size = 10
        
        calls_made = n_initial
        remaining = self.budget - calls_made
        
        for _ in range(remaining):
            if time.time() >= deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.3, sigma_max)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                success_count += 1
            else:
                sigma = max(sigma * 0.85, sigma_min)
            
            # Periodic coordinate descent refinement
            if calls_made % 20 == 0 and time.time() < deadline - 5:
                # Coordinate descent on current best
                for _cd in range(5):
                    for d in range(dim):
                        # Try small steps in each coordinate
                        for step in [sigma * 0.5, -sigma * 0.5, sigma * 0.1, -sigma * 0.1]:
                            if time.time() >= deadline:
                                break
                            trial = x.copy()
                            trial[d] = np.clip(trial[d] + step, -5, 5)
                            ft = func(trial)
                            calls_made += 1
                            if ft < fx:
                                x, fx = trial, ft
                                if fx < best_fx:
                                    best_fx = fx
                                    best_x = x.copy()
                                break
                        if time.time() >= deadline:
                            break
                    if time.time() >= deadline:
                        break
                
                # Reduce sigma after coordinate descent
                sigma = max(sigma * 0.5, sigma_min)
        
        # Phase 3: Final polish with very small steps
        if time.time() < deadline - 2 and calls_made < self.budget:
            polish_sigma = max(sigma * 0.1, 1e-8)
            for _ in range(min(50, self.budget - calls_made)):
                if time.time() >= deadline:
                    break
                for d in range(dim):
                    if time.time() >= deadline:
                        break
                    for sign in [1, -1]:
                        if time.time() >= deadline:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * polish_sigma, -5, 5)
                        ft = func(trial)
                        calls_made += 1
                        if ft < fx:
                            x, fx = trial, ft
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                            polish_sigma *= 0.5
                            break
                    polish_sigma = max(polish_sigma * 0.7, 1e-12)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
