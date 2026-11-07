"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        total_calls = 0
        budget = self.budget

        # Phase 1: Multi-start with (1+1)-ES
        # We allocate most of the budget to the main search and keep some for polishing.
        # Let's use 90% for search and 10% for polish to be safe.
        search_budget = int(budget * 0.9)
        polish_budget = budget - search_budget
        
        n_starts = 5
        # Distribute search_budget among starts
        budget_per_start = max(1, search_budget // n_starts)
        
        for start_idx in range(n_starts):
            if time.time() > deadline or total_calls >= budget:
                break
            
            # Initialize: first at center, then random
            if start_idx == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)
            
            if total_calls >= budget:
                break
                
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            # (1+1)-ES with 1/5 success rule
            sigma = 1.5
            successes = 0
            window = 10
            count = 0
            stall_count = 0
            
            # Run until we run out of local budget or global budget
            while count < budget_per_start and total_calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    stall_count = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    stall_count += 1
                
                # 1/5 success rule: adjust sigma every 'window' steps
                if count % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.5:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    else:
                        sigma *= 1.0
                    successes = 0
                
                # Restart if stalled too much
                if stall_count > 30:
                    break
            
            # Check if we should restart
            if stall_count > 30:
                continue
        
        # Phase 2: Final polish with small steps
        if best_x is not None and total_calls < budget and time.time() < deadline - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            # Use remaining budget for polishing
            remaining_calls = budget - total_calls
            polish_steps = min(polish_budget, remaining_calls)
            
            for _ in range(polish_steps):
                if time.time() > deadline or total_calls >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                # Stop if sigma is very small
                if sigma < 1e-6:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
