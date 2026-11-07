"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Powell with final fine-tuning for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
            return f_val

        dim = self.dim
        
        # Phase 1: Multi-start Powell
        # We generate a set of diverse starting points.
        # Since the domain is small, a coarse grid or random points work well.
        # Let's use a mix of random and structured points.
        
        rng = np.random.default_rng(42)
        
        # Generate initial candidates
        # 1. Center
        starts = [np.zeros(dim)]
        # 2. Random points
        n_random_starts = 15
        for _ in range(n_random_starts):
            starts.append(rng.uniform(-5, 5, dim))
        
        # We will run Powell on the best few starting points.
        # First, evaluate all starts to pick the best ones.
        start_fits = []
        for s in starts:
            if call_count >= max_calls:
                break
            if time.time() > deadline:
                break
            try:
                f_s = counted_func(s)
                start_fits.append((f_s, s))
            except Exception:
                break
        
        # Sort starts by fitness
        start_fits.sort(key=lambda x: x[0])
        
        # Run Powell on the top 3 starts
        # Powell is efficient, so we give it a decent maxiter but rely on budget.
        top_k = 3
        for i in range(min(top_k, len(start_fits))):
            if call_count >= max_calls:
                break
            if time.time() > deadline - 5.0: # Leave time for final polish
                break
                
            f_init, x_init = start_fits[i]
            
            # If this start is already worse than the current best by a lot, skip?
            # No, it might be in a different basin. But if it's very bad, maybe skip to save budget.
            # However, Powell converges fast. Let's try it.
            
            try:
                res = minimize(counted_func, x_init, method='Powell', 
                               options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-10})
                # minimize updates best_x/best_f via counted_func
            except Exception:
                # Budget exceeded or other error, break out
                break
        
        # Phase 2: Fine-tuning with small-step random walk / coordinate descent
        # This helps refine the solution to high precision, which is good for AOCC.
        if call_count < max_calls and time.time() < deadline - 2.0:
            remaining = max_calls - call_count
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            steps = min(remaining, 50)
            
            # Use a simple (1+1)-ES with very small steps
            for _ in range(steps):
                if call_count >= max_calls or time.time() > deadline - 1.0:
                    break
                x_new = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    f_new = counted_func(x_new)
                    if f_new < f - 1e-12:
                        x = x_new
                        f = f_new
                        # Optionally adapt sigma down if we are making progress
                        # sigma *= 0.9 
                except Exception:
                    break
            
            # Update global best if the fine-tuning found something better
            # (counted_func already updates best_x/best_f, so this is redundant but safe)
            if f < best_f:
                best_f = f
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
