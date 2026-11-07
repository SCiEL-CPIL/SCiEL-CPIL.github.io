"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


class Optimizer:
    """Lattice-based grid search exploiting Rastrigin periodicity, then local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        calls = 0

        # Phase 1: Lattice-based deterministic grid search
        # Rastrigin has local minima at integer coordinates.
        # We systematically evaluate points at integer and half-integer offsets
        # to rapidly identify the global basin.
        
        # Generate lattice points: integers and half-integers in [-5, 5]
        # That's 21 values per dimension: -5, -4.5, -4, ..., 4.5, 5
        # But 21^5 = 4,084,101 points - too many.
        # Use coarser grid first: just integers (-5 to 5), that's 11^5 = 161,051
        # Still too many. Use a smarter approach.
        
        # Strategy: Evaluate at key lattice points using a space-filling or
        # systematic approach with limited budget for Phase 1.
        
        # Phase 1a: Evaluate at integer lattice points (11^5 = 161,051 max)
        # But we only have 500 total calls, so we need to be selective.
        
        # Better strategy: Use a two-phase approach
        # Phase 1: Coarse grid with ~200 calls to find promising region
        # Phase 2: Fine local search around best point with remaining budget
        
        # Coarse grid: sample 11 points per dimension at integers
        # But evaluate only a subset using stratified sampling
        
        # Actually, let's use a more efficient approach:
        # 1. Evaluate at origin and unit vectors
        # 2. Use the results to guide a local search
        
        # Phase 1: Systematic exploration using lattice structure
        # Evaluate at all integer points in a reduced box first
        # Then refine around the best point
        
        # Use budget allocation: 300 for exploration, 200 for exploitation
        
        # Phase 1: Evaluate on a structured grid
        # For Rastrigin, minima are near integers. Let's evaluate at:
        # - All points where coordinates are in {-2, -1, 0, 1, 2} (5^5 = 3125 points)
        # That's too many. Let's use a smarter subset.
        
        # Use a space-filling curve or random sampling with lattice bias
        
        # Simplest effective approach:
        # 1. Start with a random point
        # 2. Do a local search using the lattice structure
        
        # Let me use a different strategy:
        # Phase 1: Evaluate at 100 random points + key lattice points
        # Phase 2: Local refinement from best point
        
        # Key lattice points: origin, ±1 in each dimension, ±2 in each dimension
        # That's 1 + 10 + 10 = 21 points
        
        # Actually, let's just do a proper grid search with limited resolution
        
        # Phase 1: Coarse grid search
        # Use 5 points per dimension: -4, -2, 0, 2, 4
        # That's 5^5 = 3125 points - still too many for 300 budget
        
        # Use 3 points per dimension: -3, 0, 3
        # That's 3^5 = 243 points - fits in 300 budget
        
        # Phase 1: Evaluate at 3^5 = 243 points on a coarse grid
        grid_vals = [-3.0, 0.0, 3.0]
        
        best_x = None
        best_f = float('inf')
        
        # Generate all grid points
        for combo in product(grid_vals, repeat=dim):
            if calls >= 300 or time.time() > deadline:
                break
            x = np.array(combo, dtype=np.float64)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local refinement around best point
        if best_x is None:
            # Fallback: random start
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
            calls += 1
        
        # Local search using (1+1)-ES style
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        
        while calls < budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * np.random.default_rng(calls).standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85
        
        return x, fx
# EVOLVE-BLOCK-END
