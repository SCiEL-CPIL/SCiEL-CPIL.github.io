import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 280.0

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Initial Global Search using Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, j + 2)
            return points

        global_calls = 25
        halton_points = halton_sequence(global_calls, dim)
        halton_points = halton_points * 10 - 5

        for x in halton_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        remaining = total_budget - calls_made

        # Phase 2: Hierarchical Multi-Scale Stochastic Tunneling
        # Coarse steps to locate basins, then reduce step magnitude by order of magnitude
        # to exploit self-similar structure of Weierstrass function.
        # Use (1+1)-ES with random restarts for robustness on non-smooth function.
        
        # Define hierarchical step sizes: start large, reduce by 10x at each level
        # Weierstrass has features at multiple scales, so this multi-scale approach
        # should converge rapidly to global minimum.
        
        # Step sizes for each hierarchical level
        # Level 0: coarse (step ~ 2.0)
        # Level 1: medium (step ~ 0.2)
        # Level 2: fine (step ~ 0.02)
        # Level 3: very fine (step ~ 0.002)
        # Level 4: ultra fine (step ~ 0.0002)
        
        step_levels = [2.0, 0.2, 0.02, 0.002, 0.0002]
        
        # For each level, we will do a certain number of (1+1)-ES iterations
        # with random restarts if stagnation is detected.
        
        # Track best per level to allow backtracking if a finer level fails
        level_best = {}
        
        for level_idx, base_step in enumerate(step_levels):
            if remaining < 10 or time.time() > deadline:
                break
            
            # Adaptive step size within this level
            step = base_step
            
            # Number of iterations for this level
            # Use more iterations for coarser levels, fewer for finer
            max_iters = max(10, min(100, remaining // 5))
            
            stagnation_count = 0
            stagnation_limit = 15
            restart_count = 0
            max_restarts = 5
            
            for it in range(max_iters):
                if remaining < 5 or time.time() > deadline:
                    break
                
                # Generate random direction
                direction = rng.normal(0, 1, dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                
                # Propose candidate
                candidate = best_x + direction * step
                candidate = np.clip(candidate, -5, 5)
                
                try:
                    fc = wrapper(candidate)
                except TimeoutError:
                    break
                
                remaining = total_budget - calls_made
                
                if fc < best_f:
                    best_f = fc
                    best_x = candidate.copy()
                    stagnation_count = 0
                    # Slightly increase step if we're making progress (up to base_step)
                    step = min(base_step, step * 1.05)
                else:
                    stagnation_count += 1
                    # Decrease step if stagnating
                    step *= 0.95
                    step = max(step, base_step * 0.1)
                
                # Check for stagnation and restart
                if stagnation_count >= stagnation_limit:
                    restart_count += 1
                    if restart_count > max_restarts:
                        break
                    
                    # Random restart: perturb best_x significantly
                    restart_step = base_step * 0.5
                    restart_dir = rng.normal(0, 1, dim)
                    restart_dir /= (np.linalg.norm(restart_dir) + 1e-10)
                    restart_x = np.clip(best_x + restart_step * restart_dir, -5, 5)
                    
                    try:
                        fr = wrapper(restart_x)
                    except TimeoutError:
                        break
                    
                    remaining = total_budget - calls_made
                    
                    if fr < best_f:
                        best_f = fr
                        best_x = restart_x.copy()
                    
                    # Reset stagnation counter and step
                    stagnation_count = 0
                    step = base_step
            
            # Save best for this level
            level_best[level_idx] = (best_f, best_x.copy())
            
            # If we made significant progress, continue to next finer level
            # If not, we might want to stay at this level or go back
            # For now, just proceed to next level
        
        # Final polish: use the finest step size for a few more iterations
        if remaining > 5 and time.time() < deadline:
            step = 0.0001
            for _ in range(min(20, remaining)):
                if time.time() > deadline:
                    break
                direction = rng.normal(0, 1, dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                candidate = np.clip(best_x + direction * step, -5, 5)
                try:
                    fc = wrapper(candidate)
                except TimeoutError:
                    break
                if fc < best_f:
                    best_f = fc
                    best_x = candidate.copy()
                remaining = total_budget - calls_made

        return best_x, best_f
