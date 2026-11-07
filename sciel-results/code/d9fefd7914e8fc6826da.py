"""Optimized program for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-Start (1+1)-ES with 1/5th success rule for Büche-Rastrigin.
    
    Strategy:
    1. Phase 1: Sample 10 random points to find a good starting region.
    2. Phase 2: Run a single (1+1)-ES with adaptive sigma (1/5th success rule) 
       until budget or time limit is reached.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We stop at 290s.
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Global best tracking
        best_x = None
        best_f = float('inf')
        evals_used = 0

        # --- Phase 1: Multi-Start Exploration ---
        # Evaluate a small number of random points to find a good starting region.
        # This captures the "population diversity" aspect to handle multimodality.
        n_starts = 10
        if budget < n_starts + 10:
            n_starts = max(1, budget // 10)
        
        # Generate candidates
        candidates = rng.uniform(-5, 5, (n_starts, dim))
        
        # Evaluate candidates
        candidate_f = []
        for c in candidates:
            if time.time() > deadline or evals_used >= budget:
                break
            fc = func(c)
            candidate_f.append(fc)
            evals_used += 1
            
        if candidate_f:
            best_idx = np.argmin(candidate_f)
            x = candidates[best_idx].copy()
            fx = candidate_f[best_idx]
            best_x = x.copy()
            best_f = fx
        else:
            # Fallback if no evaluations could be made (should not happen)
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            evals_used += 1
            best_x = x.copy()
            best_f = fx

        # --- Phase 2: Local Refinement with (1+1)-ES ---
        # Use the remaining budget for adaptive local search.
        # Inspired by the high-scoring (1+1)-ES implementations.
        
        sigma = 1.5
        success_window = []
        window_size = 10
        min_sigma = 0.01
        max_sigma = 10.0

        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            evals_used += 1
            
            # Update best and track success
            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_window.append(0)
            
            # Adapt sigma based on 1/5th success rule
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window[-window_size:])
                if success_rate > 0.2:
                    sigma *= 1.05  # Increase step size if too many successes
                elif success_rate < 0.1:
                    sigma *= 0.95  # Decrease step size if too few successes
                # Keep window manageable
                if len(success_window) > window_size:
                    success_window.pop(0)
                
                sigma = max(min_sigma, min(sigma, max_sigma))

        return best_x, best_f
# EVOLVE-BLOCK-END
