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
        best_x = np.zeros(dim)
        
        start_time = time.time()
        deadline = start_time + 325.0

        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Sobol-based initialization for uniform coverage
        # Use scipy's Sobol sequence if available, otherwise fallback to stratified random
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
            init_count = 20
            init_points = sobol.random(init_count) * 10.0 - 5.0
        except ImportError:
            init_count = 20
            # Stratified sampling fallback
            init_points = np.zeros((init_count, dim))
            for i in range(init_count):
                for d in range(dim):
                    # Divide into 5 segments, pick random point in one
                    seg = rng.integers(0, 5)
                    init_points[i, d] = -5.0 + seg * 2.0 + rng.uniform(0, 2.0)

        init_fvals = []
        for p in init_points:
            if calls_made >= total_budget:
                break
            f_val = wrapper(p)
            init_fvals.append(f_val)
        
        if calls_made >= total_budget:
            return best_x, best_f

        # Phase 2: Multi-scale (1+1)-ES with restarts
        # Weierstrass is multi-modal and self-similar. Use adaptive sigma with restarts.
        
        # Start from the best found in initialization
        current_best_idx = np.argmin(init_fvals)
        x = init_points[current_best_idx].copy()
        fx = init_fvals[current_best_idx]
        
        # (1+1)-ES with 1/5 success rule and multi-scale sigma
        # Sigma starts large for exploration, decays with success
        sigma = 2.0  # Large initial step for global exploration
        sigma_min = 1e-4
        sigma_max = 5.0
        successes = []
        window_size = 15
        
        # Budget allocation: 350 calls for main search, 150 for final polish
        main_budget = 350
        polish_budget = 150
        
        steps = 0
        stagnation_counter = 0
        best_fx = fx
        
        while steps < main_budget and calls_made < total_budget - polish_budget and time.time() < deadline:
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5.0, 5.0)
            
            fx_trial = wrapper(x_trial)
            
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                successes.append(1)
                stagnation_counter = 0
                
                if fx < best_fx:
                    best_fx = fx
            else:
                successes.append(0)
                stagnation_counter += 1
                
            # Update sigma using 1/5 success rule
            if len(successes) >= window_size:
                success_rate = np.mean(successes[-window_size:])
                if success_rate > 0.2:
                    sigma *= 1.2  # Increase step size
                elif success_rate < 0.2:
                    sigma /= 1.2  # Decrease step size
                sigma = np.clip(sigma, sigma_min, sigma_max)
                successes = successes[-window_size//2:]  # Keep recent history
                
            # Restart if stagnating for too long
            if stagnation_counter > 30:
                # Random restart near current best or global best
                if rng.random() < 0.5:
                    x = best_x + rng.standard_normal(dim) * 0.5
                else:
                    x = best_x.copy()
                fx = wrapper(x)
                sigma = 1.0  # Reset sigma for restart
                successes = []
                stagnation_counter = 0
                
            steps += 1

        # Phase 3: Final polish with small sigma
        remaining_budget = total_budget - calls_made
        if remaining_budget > 0 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            successes = []
            
            while calls_made < total_budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)
                
                fx_trial = wrapper(x_trial)
                
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    successes.append(1)
                else:
                    successes.append(0)
                    
                if len(successes) >= 10:
                    success_rate = np.mean(successes[-10:])
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma /= 1.1
                    sigma = np.clip(sigma, 1e-5, 0.5)
                    successes = successes[-5:]

        return best_x, best_f
