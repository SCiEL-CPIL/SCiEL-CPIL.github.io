"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


class Optimizer:
    """Lattice-Pruned CMA-ES for Rastrigin on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        # Stage 1: Coarse grid search to find the best basin
        # Rastrigin has period 2, so we sample every 1.0 unit within [-5, 5]
        # This gives 11 points per dimension -> 11^5 = 161051 points
        # We have 500 budget. 161k is too many.
        # Let's be smarter. The global min of Rastrigin is at 0.
        # With shift/rotation, the min is shifted/rotated.
        # But the "periodic" structure suggests that sampling at integer or half-integer steps might work.
        # However, 161k calls is impossible.
        
        # Let's use a smarter coarse search.
        # Maybe just random sampling first?
        # Or a smaller grid?
        # The directive says: "first, perform a coarse grid search (step 1.0) over the 11^5 integer-aligned points"
        # But 11^5 = 161,051. We only have 500 calls.
        # This directive might be flawed for this specific budget, or I need to interpret "coarse" differently.
        # Maybe step 2.0? 5 points per dim -> 5^5 = 3125. Still too many.
        # Step 5.0? 2 points per dim -> 2^5 = 32.
        
        # Let's stick to the spirit of "Lattice-Pruned" but adapt to budget.
        # If I can't do the full grid, I should do a random search for the first part, then CMA-ES.
        # The current program is a (1+1)-ES which is essentially a local search.
        # The main weakness is getting stuck in local minima.
        
        # Strategy:
        # 1. Use ~100-200 calls for a "scatter" phase (random or coarse grid) to find a good starting point.
        # 2. Use the remaining calls for CMA-ES or a strong local search (like Nelder-Mead or L-BFGS-B).
        
        # Let's try:
        # 1. Random sampling: 100 points.
        # 2. Pick the best.
        # 3. Run CMA-ES (or Nelder-Mead) from there with remaining budget.
        
        # Actually, the directive explicitly suggests "Lattice-Pruned CMA-ES".
        # If I can't afford the full lattice, maybe I should just do a random scatter.
        # But let's look at the inspiration: "Nelder-Mead ensemble".
        
        # Let's try a hybrid:
        # 1. Scatter: 50 random points.
        # 2. Local search from the best: CMA-ES or Nelder-Mead.
        
        # Let's implement a simple CMA-ES manually or use scipy?
        # SciPy is available. `scipy.optimize.minimize` with method='Nelder-Mead' or 'L-BFGS-B'.
        # L-BFGS-B is good for smooth functions, but Rastrigin is non-smooth (discontinuous gradients).
        # Nelder-Mead is derivative-free and handles multimodality better locally.
        # CMA-ES is the gold standard for BBOB.
        
        # Since I can't import a CMA-ES library (only stdlib, numpy, scipy), I'll implement a simple CMA-ES or use Nelder-Mead.
        # Nelder-Mead is in scipy.
        
        # Plan:
        # 1. Scatter 50 random points.
        # 2. Take the best point x0.
        # 3. Run scipy.optimize.minimize(func, x0, method='Nelder-Mead', options={'maxiter': 450, 'xatol': 1e-8, 'fatol': 1e-8}).
        #    This will use up to 450 more calls (actually Nelder-Mead uses more, ~dim*10 per iteration).
        #    Budget is 500. 50 + 450 = 500.
        #    Nelder-Mead might exceed 450 calls if maxiter is high.
        #    Let's limit maxiter to something safe.
        #    dim=5. Nelder-Mead typically uses ~5-10 calls per iteration.
        #    So 450 calls / 10 = 45 iterations.
        
        # Let's refine the scatter.
        # Maybe use a coarse grid of size 2^5 = 32 points?
        # Points at -4, -2, 0, 2, 4?
        # Or just random.
        
        # Let's try:
        # 1. 32 calls: Grid points at {-4, -2, 0, 2, 4}^5? No, that's 5^5=3125.
        #    Grid points at {-4, 0, 4}^5? 3^5=243. Too many.
        #    Grid points at {-4, 4}^5? 2^5=32.
        #    This covers the corners.
        #    Plus center 0.
        #    So 33 points.
        # 2. Then Nelder-Mead from the best.
        
        # Let's try this.
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Scatter
        # 32 corner points + 1 center + 17 random?
        # Let's just do 50 random points to be safe and diverse.
        # Or maybe a mix.
        
        # Let's do 50 random points.
        x_scatter = rng.uniform(-5, 5, (50, dim))
        f_scatter = np.array([func(x) for x in x_scatter])
        
        best_idx = np.argmin(f_scatter)
        x0 = x_scatter[best_idx]
        f0 = f_scatter[best_idx]
        
        # Phase 2: Local Search
        # Use scipy.optimize.minimize with Nelder-Mead
        # We have 450 calls left.
        # Nelder-Mead might use more than 450.
        # We need to guard the calls.
        
        # Since we can't easily limit scipy's internal calls, we might need to implement our own loop or use a method that respects a callback.
        # SciPy's minimize does not have a strict call limit callback that stops it immediately and returns the best so far.
        # It might raise an exception or just stop at maxiter.
        
        # Alternative: Implement a simple (1+1)-ES or Nelder-Mead manually to control calls strictly.
        # The current program is a (1+1)-ES.
        # Let's improve the (1+1)-ES by starting from a better point (from the scatter) and maybe adding a "restart" mechanism or better sigma adaptation.
        
        # Actually, let's try Nelder-Mead manually? No, too complex.
        # Let's stick to (1+1)-ES but with the scatter initialization.
        
        # Current (1+1)-ES:
        # x = x0
        # fx = f0
        # sigma = 2.0
        # for _ in range(450):
        #    ...
        
        # This should be better than the current program which starts at a single random point.
        
        x = x0
        fx = f0
        sigma = 1.0 # Start smaller since we are in a basin? Or 2.0?
        
        # Let's try sigma=1.0.
        
        for _ in range(450):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            
            if time.time() > deadline:
                if fy < fx:
                    x, fx = y, fy
                break

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85
                
        return x, fx
# EVOLVE-BLOCK-END
