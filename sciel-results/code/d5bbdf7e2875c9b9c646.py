"""Optimizer for BBOB-style f15 (Rastrigin rotated) using Multi-Start (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer using Multi-Start (1+1)-ES with adaptive step size.
    This approach is robust for multimodal functions like Rastrigin by
    exploring multiple basins and refining the best one found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # 1. Initialization: Evaluate a diverse set of points
        # We use the origin, random points, and points near the boundaries of the likely search area [-4, 4]
        init_candidates = []
        init_candidates.append(np.zeros(dim))
        
        # Add random points in [-4, 4]
        for _ in range(9):
            init_candidates.append(rng.uniform(-4, 4, dim))
            
        # Add some points on the "integer lattice" within [-4, 4] to help with Rastrigin periodicity
        # We sample a subset of the lattice
        lattice_points = []
        # For dim=5, we can't check all 9^5 points. We pick random integer points.
        for _ in range(10):
            p = rng.integers(-4, 5, size=dim).astype(float)
            lattice_points.append(p)
        init_candidates.extend(lattice_points)
        
        # Evaluate initial candidates
        for p in init_candidates:
            if calls_made >= budget or time.time() > deadline:
                break
            evaluate(p)

        if best_x is None:
            # Fallback
            evaluate(np.zeros(dim))
            return best_x, best_f

        # 2. Multi-Start (1+1)-ES
        # We will run ES from the best point found. If it stagnates, we try a new start.
        
        # Parameters
        window_size = 15
        min_sigma = 1e-4
        max_sigma = 5.0
        
        # Run ES from the current best
        current_x = best_x.copy()
        current_f = best_f
        sigma = 1.0
        
        successes = []
        stagnation_count = 0
        max_stagnation = 50
        
        # We keep a list of "restart points" - initially just the best.
        # If we get stuck, we might try a random restart.
        
        while calls_made < budget and time.time() < deadline:
            # Generate offspring
            y = current_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            
            if fy < current_f:
                # Success
                current_x = y
                current_f = fy
                successes.append(1)
                stagnation_count = 0
            else:
                # Failure
                successes.append(0)
                stagnation_count += 1
            
            # Adapt sigma based on success rate in the last window
            if len(successes) >= window_size:
                successes = successes[-window_size:]
                success_rate = np.mean(successes)
                
                # 1/5 rule: if success rate is > 0.2, increase sigma; if < 0.2, decrease
                if success_rate > 0.25:
                    sigma *= 1.1
                elif success_rate < 0.15:
                    sigma *= 0.9
                
                sigma = np.clip(sigma, min_sigma, max_sigma)
                
                # Check for stagnation
                if stagnation_count > max_stagnation:
                    # Restart logic
                    # Try a new random point or perturb the best
                    if calls_made < budget - 10:
                        # 50% chance to try a completely new random point
                        if rng.random() < 0.5:
                            new_start = rng.uniform(-4, 4, dim)
                        else:
                            # Perturb the best point with a larger step
                            new_start = best_x + 2.0 * rng.standard_normal(dim)
                        
                        new_start = np.clip(new_start, -5, 5)
                        new_f = evaluate(new_start)
                        
                        if new_f < current_f:
                            current_x = new_start
                            current_f = new_f
                        else:
                            # Keep current_x but reset stagnation counter and maybe reduce sigma
                            current_x = best_x.copy()
                            current_f = best_f
                            sigma = max(sigma * 0.5, min_sigma)
                        
                        successes = []
                        stagnation_count = 0
                    else:
                        # No budget for restart, just keep going with smaller sigma
                        sigma *= 0.5
                        successes = []
                        stagnation_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
