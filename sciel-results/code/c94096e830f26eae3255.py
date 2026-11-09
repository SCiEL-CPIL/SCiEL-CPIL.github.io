import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        # 1. Global Scan - Reduced to free budget for refinement
        n_scan = 30
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            if time.time() > deadline - 5.0: break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            
        if best_x is None:
             x0 = rng.uniform(-5, 5, self.dim)
             return x0, func(x0)

        # 2. Local Refinement
        remaining = self.budget - len(candidates)
        if remaining <= 0:
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])
        top_candidates = candidates[:2]
        
        # Allocate 75% budget to best, 25% to second best
        budget_best = int(remaining * 0.75)
        budget_second = remaining - budget_best
        
        def run_es(start_x, start_f, n_evals):
            x = start_x.copy()
            fx = start_f
            sigma = 1.5  # Moderate initial sigma
            successes = 0
            total = 0
            stagnant = 0
            
            for _ in range(n_evals):
                if time.time() > deadline - 5.0:
                    break
                # Generate neighbor
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    stagnant = 0
                else:
                    stagnant += 1
                    
                total += 1
                if total % 10 == 0:
                    success_rate = successes / total
                    if success_rate > 0.35:
                        sigma *= 1.15  # Conservative growth
                    elif success_rate < 0.15:
                        sigma *= 0.8   # Conservative decay
                    successes = 0
                    total = 0
                    sigma = max(sigma, 0.25)  # Floor to prevent over-shrinkage
                    
                # If stuck, try a moderate jump
                if stagnant > 30:
                    x = np.clip(x + rng.standard_normal(self.dim) * 1.0, -5, 5)
                    fx = func(x)
                    sigma = 1.0
                    stagnant = 0

            return x, fx

        # Refine best
        x_best, f_best = run_es(best_x, best_f, budget_best)
        if f_best < best_f:
             best_x, best_f = x_best, f_best
             
        # Refine second best if exists and budget remains
        if len(top_candidates) > 1:
             x_second, f_second = top_candidates[1]
             if budget_second > 0:
                 x_s, f_s = run_es(x_second, f_second, budget_second)
                 if f_s < best_f:
                     best_x, best_f = x_s, f_s

        # 3. Final Polish Phase - very small steps to refine on narrow peaks
        polish_budget = 50
        if time.time() < deadline - 10.0:
            polish_sigma = 0.05
            for _ in range(polish_budget):
                if time.time() > deadline - 5.0:
                    break
                y = np.clip(best_x + polish_sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x = y
                    best_f = fy
                # Gradually reduce sigma during polish
                polish_sigma *= 0.95

        return best_x, best_f
