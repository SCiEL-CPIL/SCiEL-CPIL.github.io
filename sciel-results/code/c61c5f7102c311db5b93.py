"""Rewritten optimizer for BBOB f21 (Gallagher 101 peaks)."""
import numpy as np
import time
import math


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global scan with multiple random restarts
        # f21 has 101 peaks, so we need good global coverage
        # Use a moderate number of candidates for the scan phase
        
        # Strategy:
        # - Phase 1 (first ~40% of budget): scan with (1+1)-ES from multiple restarts
        #   but keep the best few basins
        # - Phase 2 (remaining 60%): refine the best basin with adaptive sigma
        
        # Actually, let's do a simpler but effective approach:
        # 1. Quick global scan: evaluate ~50 random points to find promising regions
        # 2. From the best few, run (1+1)-ES with sigma adaptation
        # 3. Use remaining budget for final polish
        
        # Phase 1: Global scan
        n_scan = 50
        candidates = []
        for i in range(n_scan):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x.copy()))
        
        # Sort by fitness, keep top 3
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Refine each of the top 3 candidates with (1+1)-ES
        # Allocate remaining budget among them
        remaining = budget - len(candidates)
        
        # We'll do sequential refinement: start with best, then second, etc.
        # But actually, for AOCC we want early progress, so let's interleave or
        # just refine the best one thoroughly.
        
        # Let's refine the best candidate with most of the budget,
        # and give some to the second best as backup.
        
        # Split: 70% to best, 30% to second best (if available)
        if len(candidates) >= 2:
            budget_best = int(remaining * 0.7)
            budget_second = remaining - budget_best
        else:
            budget_best = remaining
            budget_second = 0
        
        # Refine best candidate
        x_best, fx_best = candidates[0]
        sigma = 1.5
        sigma_min = 0.01
        sigma_max = 3.0
        success = 0
        target = 5  # 1/5 rule: target success rate 1/5
        
        for i in range(budget_best):
            if time.time() > deadline:
                break
            
            # Adaptive sigma: use 1/5 rule
            y = x_best + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            
            if fy < fx_best:
                x_best, fx_best = y, fy
                success += 1
            else:
                success -= 1
            
            # Adapt sigma every 5 steps
            if (i + 1) % target == 0:
                if success > 0:
                    sigma *= 1.2
                elif success < 0:
                    sigma *= 0.85
                success = 0
            
            sigma = np.clip(sigma, sigma_min, sigma_max)
        
        # Refine second best candidate
        if len(candidates) >= 2 and budget_second > 0:
            x_second, fx_second = candidates[1]
            sigma = 1.0
            success = 0
            
            for i in range(budget_second):
                if time.time() > deadline:
                    break
                
                y = x_second + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx_second:
                    x_second, fx_second = y, fy
                    success += 1
                else:
                    success -= 1
                
                if (i + 1) % target == 0:
                    if success > 0:
                        sigma *= 1.2
                    elif success < 0:
                        sigma *= 0.85
                    success = 0
                
                sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Update best if second became better
            if fx_second < fx_best:
                x_best, fx_best = x_second, fx_second
        
        # Phase 3: Final polish with very small sigma
        if time.time() < deadline - 5:
            sigma_polish = 0.05
            n_polish = min(50, int((deadline - time.time()) * 100))  # rough estimate
            for i in range(n_polish):
                if time.time() > deadline:
                    break
                y = x_best + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < fx_best:
                    x_best, fx_best = y, fy
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.9
                sigma_polish = max(sigma_polish, 0.001)
        
        return x_best, fx_best
