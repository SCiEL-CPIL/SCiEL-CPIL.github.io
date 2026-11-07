"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based grid search exploiting Rastrigin periodicity, then local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0

        def timed_call(x):
            nonlocal calls_used
            calls_used += 1
            return func(x)

        # Phase 1: Lattice-based deterministic grid search
        # Rastrigin has period 2, so evaluate at integer and half-integer offsets
        # within [-5, 5]^dim. Use a coarse-to-fine approach.
        
        # Coarse lattice: step size 2.0 (one period), evaluate at integers
        # Points: -5, -3, -1, 1, 3, 5 -> 6 points per dim -> 6^5 = 7776 points
        # That's too many. Use step 2.5 or just key points.
        
        # Better: Use step 2.0 but only evaluate a subset. 
        # Actually for dim=5, 6^5=7776 is fine within 500 budget? No, 7776 > 500.
        # So we need to be selective. Let's use a smarter approach.
        
        # Strategy: Evaluate at 0 (origin), then at ±2, ±4 in each dimension
        # Key lattice points: combinations of {-4, -2, 0, 2, 4} in each dim
        # That's 5^5 = 3125 points. Still too many.
        
        # Let's do a two-phase lattice:
        # Phase 1a: Evaluate at origin and single-axis deviations
        # Phase 1b: Refine around best point found
        
        # Actually, let's be more strategic. With only 500 calls:
        # - Spend ~100 calls on a structured lattice
        # - Spend ~400 calls on local refinement
        
        # Lattice: Evaluate at points where each coord is in {-4, -2, 0, 2, 4}
        # But 5^5 = 3125 is too many. Instead, use a space-filling approach.
        
        # Alternative: Use Latin Hypercube or Sobol for initial spread,
        # but the directive says lattice-based. Let's do a partial lattice.
        
        # Evaluate at:
        # 1. Origin: (0,0,0,0,0)
        # 2. Single-axis: ±2 and ±4 in each dimension (10 per dim * 5 dims = 50 points)
        # 3. Two-axis combinations: ±2 in pairs (C(5,2)*4 = 40 points)
        # Total: 1 + 50 + 40 = 91 points for coarse lattice
        
        # Let's generate these points
        lattice_points = []
        
        # Origin
        lattice_points.append(np.zeros(dim))
        
        # Single-axis deviations at ±2, ±4
        for i in range(dim):
            for offset in [-4, -2, 2, 4]:
                p = np.zeros(dim)
                p[i] = offset
                lattice_points.append(p)
        
        # Two-axis combinations at ±2
        for i in range(dim):
            for j in range(i+1, dim):
                for oi in [-2, 2]:
                    for oj in [-2, 2]:
                        p = np.zeros(dim)
                        p[i] = oi
                        p[j] = oj
                        lattice_points.append(p)
        
        # Evaluate all lattice points, keeping track of best
        best_x = None
        best_f = float('inf')
        
        for p in lattice_points:
            if time.time() > deadline:
                break
            if calls_used >= budget - 100:  # reserve calls for refinement
                break
            f_val = timed_call(p)
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        # If we didn't find a good point, try random initialization
        if best_x is None or best_f == float('inf'):
            rng = np.random.default_rng(42)
            best_x = rng.uniform(-5, 5, dim)
            best_f = timed_call(best_x)
        
        # Phase 2: Local refinement using (1+1)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # start with smaller sigma for refinement
        
        while calls_used < budget - 1 and time.time() < deadline:
            y = np.clip(x + sigma * np.random.default_rng(calls_used).standard_normal(dim), -5, 5)
            fy = timed_call(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                if sigma < 1e-6:
                    # Restart with larger sigma from current best
                    sigma = 0.3
        
        return x, fx
# EVOLVE-BLOCK-END
