"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with per-dimension step sizes and budget reallocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_1plus1(self, func, rng, x0, sigma0, max_calls, deadline):
        """
        (1+1)-ES with per-dimension step sizes using 1/5 success rule.
        Returns best_x, best_f, calls_used, best_progress (for budget reallocation).
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        sigma = np.full(dim, sigma0, dtype=float)
        calls = 1
        
        # 1/5 success rule: optimal step size adaptation
        c_inc = (1.0 + 0.2) ** 0.2  # ~1.037, actually let me use the standard values
        # Standard 1/5 rule: if success rate > 1/5, increase sigma; else decrease
        # Simpler: c_inc = 1.1, c_dec = 0.9 is common, but let's use more aggressive
        c_inc = 1.15
        c_dec = 0.85

        best_x = x.copy()
        best_f = fx
        initial_f = fx
        
        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break
                
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= c_inc
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= c_dec
                
            sigma = np.clip(sigma, 1e-10, 5.0)

        # Compute progress relative to initial
        progress = initial_f - best_f if initial_f > best_f else 0.0
        return best_x, best_f, calls, progress

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Create a diverse set of starting points
        # Center
        starts = [np.zeros(dim)]
        
        # Axis-aligned points (along each dimension)
        for d in range(dim):
            x0 = np.zeros(dim)
            x0[d] = 2.0
            starts.append(x0)
            x0[d] = -2.0
            starts.append(x0)
        
        # Random points
        num_random = 4
        for _ in range(num_random):
            x0 = rng.uniform(-3, 3, size=dim)
            starts.append(x0)
        
        # Sort starts: center first, then others
        # We'll process them in order, with budget reallocation
        
        # Initial budget allocation: give more to center
        # Total starts: 1 + 2*5 + 4 = 15
        num_starts = len(starts)
        
        # Adaptive budget allocation
        # Start with equal allocation, then reallocate based on progress
        base_alloc = budget // num_starts
        base_alloc = max(base_alloc, 10)  # Minimum 10 calls per restart
        
        # Process each restart
        for i, x0 in enumerate(starts):
            if total_calls >= budget:
                break
            
            remaining = budget - total_calls
            if remaining < 5:
                break
            
            # Allocate budget: center gets more, others get base
            if i == 0:
                alloc = min(base_alloc * 3, remaining)
            else:
                alloc = min(base_alloc, remaining)
            
            if alloc < 2:
                break
            
            # Initial sigma: smaller for center (exploit), larger for others (explore)
            if i == 0:
                sigma0 = 1.5
            else:
                sigma0 = 2.0
            
            bx, bf, used, progress = self._es_1plus1(func, rng, x0, sigma0, alloc, deadline)
            total_calls += used
            
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            
            # If we're close to deadline, stop
            if time.time() > deadline - 10.0:
                break
        
        # Final polish: use best point found with very small sigma
        if total_calls < budget and time.time() < deadline - 5.0:
            remaining = budget - total_calls
            if remaining >= 10:
                sigma0 = 0.1
                bx, bf, used, _ = self._es_1plus1(func, rng, best_x, sigma0, remaining, deadline)
                total_calls += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
