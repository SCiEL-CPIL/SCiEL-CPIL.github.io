"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start CMA-ES with structured exploration for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        rng = np.random.default_rng(12345)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def check_deadline():
            return time.time() > deadline

        # Phase 1: Structured exploration using Sobol sequence for good coverage
        # Use ~250 calls for global exploration
        n_explore = 250
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate 256 points (power of 2 for Sobol)
        sobol_points = sobol.random(n_explore)  # in [0,1]^dim
        sobol_points = sobol_points * 10.0 - 5.0  # map to [-5, 5]^dim
        
        # Also add some random points for diversity
        random_points = rng.uniform(-5, 5, size=(50, dim))
        
        # Combine and shuffle
        all_points = np.vstack([sobol_points, random_points])
        rng.shuffle(all_points)
        
        # Evaluate exploration points
        for x in all_points:
            if calls_made >= n_explore + 50 or check_deadline():
                break
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
        
        # Phase 2: Multi-start local optimization around top candidates
        # Keep track of top 5 points from exploration
        top_points = []
        # Re-evaluate to get top points (we don't have stored values, so re-sample)
        # Actually, let's just use the best point and do a few local refinements
        
        if calls_made < budget and best_x is not None:
            # Try multiple local optimizations from different starting points
            n_local_starts = min(5, (budget - calls_made) // 20)
            
            # Generate diverse starting points near the best and some random
            start_points = [best_x.copy()]
            for _ in range(n_local_starts - 1):
                # Perturb best point
                perturbed = best_x + rng.standard_normal(dim) * rng.uniform(0.5, 2.0)
                start_points.append(np.clip(perturbed, -5, 5))
            
            for start_idx, x0 in enumerate(start_points):
                if calls_made >= budget or check_deadline():
                    break
                
                # Local CMA-ES-like optimization
                x = x0.copy()
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                
                sigma = rng.uniform(0.3, 1.0)
                stall_count = 0
                max_stall = 15
                local_calls = 0
                max_local_calls = min(40, budget - calls_made)
                
                while local_calls < max_local_calls and calls_made < budget:
                    if check_deadline():
                        break
                    
                    # Generate candidate
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    local_calls += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.03
                        stall_count = 0
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                    else:
                        sigma *= 0.93
                        stall_count += 1
                    
                    if stall_count > max_stall:
                        break

        # Phase 3: Final fine-tuning if budget remains
        if calls_made < budget and best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            stall_count = 0
            
            while calls_made < budget and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.9
                    stall_count += 1
                
                if stall_count > 30:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
