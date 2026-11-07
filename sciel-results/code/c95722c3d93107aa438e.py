import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        rng = np.random.default_rng(0)
        
        # Start near origin since minimizer is in [-4,4]^5
        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1
        
        # Phase 1: Exploration with population and correlated directions
        # Use ~40% of budget for exploration
        explore_budget = int(budget * 0.4)
        pop_size = 8
        
        # Diagonal covariance for exploration
        sigma = np.ones(dim) * 2.0  # Larger initial step for exploration
        
        # Track recent directions for correlation
        recent_steps = []
        recent_successes = []
        
        # Phase 1: Coarse exploration
        for i in range(explore_budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Generate a small population
            for j in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate candidate with potential correlation
                if len(recent_steps) >= 5:
                    if rng.random() < 0.4:
                        # Correlated step based on recent successful directions
                        recent_steps_arr = np.array(recent_steps[-5:])
                        recent_successes_arr = np.array(recent_successes[-5:])
                        weighted_steps = recent_steps_arr * recent_successes_arr[:, np.newaxis]
                        avg_step = np.sum(weighted_steps, axis=0) / (np.sum(recent_successes_arr) + 1e-8)
                        step = sigma * (0.5 * rng.standard_normal(dim) + 0.5 * avg_step)
                    else:
                        step = sigma * rng.standard_normal(dim)
                else:
                    step = sigma * rng.standard_normal(dim)
                
                y = np.clip(x_best + step, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx_best:
                    x_best = y
                    fx_best = fy
                    recent_steps.append(step)
                    recent_successes.append(1)
                else:
                    recent_steps.append(step)
                    recent_successes.append(0)
                
                if len(recent_steps) > 20:
                    recent_steps = recent_steps[-20:]
                    recent_successes = recent_successes[-20:]
            
            # Adapt sigma based on success rate in this population
            if recent_successes:
                success_rate = np.mean(recent_successes[-pop_size:])
                if success_rate > 0.5:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.7
            
            sigma = np.clip(sigma, 0.5, 5.0)
        
        # Phase 2: Fine exploitation
        # Reset sigma for fine-grained search
        sigma = np.clip(sigma, 0.1, 2.0) * 0.5
        sigma = np.clip(sigma, 1e-4, 2.0)
        
        # Reset direction tracking for exploitation phase
        recent_steps = []
        recent_successes = []
        
        remaining_budget = budget - calls
        
        for i in range(remaining_budget):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Generate candidate with fine-grained steps and correlation
            if len(recent_steps) >= 5:
                if rng.random() < 0.5:
                    recent_steps_arr = np.array(recent_steps[-5:])
                    recent_successes_arr = np.array(recent_successes[-5:])
                    weighted_steps = recent_steps_arr * recent_successes_arr[:, np.newaxis]
                    avg_step = np.sum(weighted_steps, axis=0) / (np.sum(recent_successes_arr) + 1e-8)
                    step = sigma * (0.6 * rng.standard_normal(dim) + 0.4 * avg_step)
                else:
                    step = sigma * rng.standard_normal(dim)
            else:
                step = sigma * rng.standard_normal(dim)
            
            y = np.clip(x_best + step, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx_best:
                x_best = y
                fx_best = fy
                recent_steps.append(step)
                recent_successes.append(1)
                # Increase sigma slightly for improved dimensions
                improved_mask = np.abs(step) > 1e-6
                sigma[improved_mask] *= 1.1
            else:
                recent_steps.append(step)
                recent_successes.append(0)
                # Decrease sigma
                sigma *= 0.95
            
            if len(recent_steps) > 20:
                recent_steps = recent_steps[-20:]
                recent_successes = recent_successes[-20:]
            
            sigma = np.clip(sigma, 1e-4, 2.0)
            
            # Periodic adaptation in exploitation phase
            if i % 10 == 0 and recent_successes:
                success_rate = np.mean(recent_successes[-10:])
                if success_rate > 0.5:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.85
        
        return x_best, fx_best
