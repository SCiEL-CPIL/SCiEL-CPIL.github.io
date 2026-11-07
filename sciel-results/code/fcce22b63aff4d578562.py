"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based grid search exploiting Rastrigin periodicity, then (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used
            fx = func(x)
            calls_used += 1
            return fx

        # Phase 1: Lattice-based deterministic grid search
        # Rastrigin has periodic structure with period 2 in each dimension.
        # Evaluate at integer and half-integer coordinates to find the global basin.
        
        # Create grid points: integers and half-integers in [-5, 5]
        # That's 11 points per dimension (from -5 to 5 in steps of 1)
        # But we also want half-integers: -5, -4.5, -4, ..., 4.5, 5 => 21 points per dim
        # 21^5 = 4,084,101 points - too many for 500 budget
        
        # Instead, use a coarser lattice: integers only first (11^5 = 161,051 - still too many)
        # Use a smarter approach: evaluate at key points based on Rastrigin structure
        
        # For Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # The global minimum is at x=0 in each dimension (before shift/rotation)
        # With hidden shift/rotation, we need to search more broadly
        
        # Strategy: Use a multi-resolution lattice
        # Level 1: Coarse grid (integers) - 11^5 = 161,051 points, but we only have 500 calls
        # So we need to be selective
        
        # Better approach: Use Latin Hypercube Sampling + local refinement
        # Or: Evaluate at strategic points based on the periodic structure
        
        # Since we have limited budget, let's use a hybrid:
        # 1. Evaluate at 200 well-distributed points (LHS-like)
        # 2. Take the best few and do local (1+1)-ES refinement
        
        # Phase 1: Stratified random sampling with periodic awareness
        n_phase1 = 200
        best_x = None
        best_fx = float('inf')
        
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            if calls_used >= budget - 50:  # reserve budget for phase 2
                break
            
            # Sample points with periodic structure in mind
            # Use stratified sampling: divide [-5,5] into 5 intervals per dim
            x = np.zeros(dim)
            for d in range(dim):
                # Choose stratum
                stratum = (i + d) % 5
                base = -5 + stratum * 2.0
                # Sample within stratum with some randomness
                x[d] = base + rng.uniform(0, 2.0)
            
            x = np.clip(x, -5, 5)
            fx = evaluate(x)
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # Phase 2: Local refinement using (1+1)-ES from the best point
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 1.0  # Start with moderate step size
            
            remaining = budget - calls_used
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls_used >= budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, 5.0)
                else:
                    sigma *= 0.85
                    if sigma < 0.01:
                        sigma = 0.01
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
