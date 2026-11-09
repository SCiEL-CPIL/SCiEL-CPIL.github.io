"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State variables
        best_x = np.zeros(dim)
        best_f = float('inf')
        call_count = 0
        
        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return best_f
            call_count += 1
            try:
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except:
                return 1e100

        bounds = [(-5, 5)] * dim
        
        # Strategy:
        # 1. "Valley Hunting": The Rosenbrock function has a narrow, curved valley 
        #    leading to the minimum. Since the function is rotated and shifted, 
        #    the valley's location is unknown. 
        #    We perform a deterministic coarse search using a radial grid around the origin 
        #    (and potentially other centers if time permits, but origin is likely close 
        #    due to the [-4,4] constraint on the minimizer) to find points with low function 
        #    values. Points in the valley have significantly lower values than points 
        #    outside the valley.
        # 2. "Gradient Polish": Once a point in the valley is found, we use L-BFGS-B 
        #    with strict tolerances to slide down the valley to the minimum. 
        #    L-BFGS-B is extremely efficient for smooth functions like Rosenbrock 
        #    once in the basin of attraction.

        # Phase 1: Coarse Search (Valley Hunting)
        # We allocate ~30% of the budget to finding good starting points.
        # We use a spherical grid around the origin.
        # The minimizer is in [-4, 4]^5. The origin (0,0,0,0,0) is a reasonable center.
        # We'll sample points on spheres of increasing radii.
        
        # Define radii to check. The valley might be near the origin or shifted.
        # Since the shift is in [-4,4], the valley could be anywhere in that box.
        # A radial search from 0 might miss it if the shift is large.
        # Alternative: A grid search over the [-4, 4]^5 box is too expensive (8^5=32768).
        # Let's use a Latin Hypercube or Sobol sequence for the initial global search 
        # to ensure coverage of the [-5, 5]^5 box.
        
        # However, the directive says "Replace the multi-start L-BFGS-B heuristic with a 
        # deterministic global search using Differential Evolution (DE)...". 
        # Wait, the directive in the prompt text says: 
        # "Strategy to try in this change: Replace the multi-start L-BFGS-B heuristic with 
        # a deterministic global search using Differential Evolution (DE) to robustly locate 
        # the rotated valley, followed by a single high-precision L-BFGS-B polish..."
        
        # BUT, the current program ALREADY uses DE. 
        # The "CURRENT PROGRAM" provided in the prompt uses DE + L-BFGS-B.
        # The "INSPIRATIONS" show high scores for "Direct solver", "Multi-restart SciPy", "Multi-start L-BFGS-B".
        # The "STAGNATION" note says: "Replace the multi-start L-BFGS-B heuristic with a 
        # deterministic global search using Differential Evolution (DE)...".
        # This is confusing. The current program IS the DE one. 
        # Let's re-read carefully.
        # "CURRENT PROGRAM (score 0.0028) and its strategy: ... Phase 1: Differential Evolution ... Phase 2: L-BFGS-B Polish"
        # So the current program is the DE one.
        # The directive says: "Strategy to try in this change: Replace the multi-start L-BFGS-B heuristic with a deterministic global search using Differential Evolution (DE)..."
        # This implies the *parent* of the current program might have been the multi-start L-BFGS-B one? 
        # Or the directive is generic?
        # "Step 1: in one sentence, state a strategy MATERIALLY DIFFERENT from the current program's"
        
        # If the current program is DE + L-BFGS-B, and it scores 0.0028 (very bad), 
        # and the inspirations (Direct Solver, Multi-restart L-BFGS-B) score 5-7 (very good),
        # then DE is likely failing or too slow/wasteful for this specific budget (500 calls).
        # DE with popsize=12 and maxiter=29 uses 348 calls just for the initial population and iterations?
        # Actually, DE evaluates the population each iteration. 
        # Popsize 12 * 29 iterations = 348 calls. 
        # Then L-BFGS-B uses the rest.
        # Why is it so bad? Maybe DE gets stuck or doesn't converge well in 29 iterations for dim=5?
        # Or maybe the "counted_func" wrapper is interfering with DE's internal logic?
        # SciPy's DE expects the function to be called. If we return `best_f` when budget is exceeded,
        # it might confuse the convergence criteria.
        
        # Let's try a different approach: **Nelder-Mead** or **Powell** for the global part?
        # Or, exploit the specific structure of Rosenbrock more directly.
        
        # The inspirations suggest "Direct solver" and "Multi-start L-BFGS-B" are better.
        # "Direct solver" (score 6.5) likely uses the known properties of Rosenbrock.
        # "Multi-start L-BFGS-B" (score 7.0) likely uses many random restarts of L-BFGS-B.
        
        # Let's try **Multi-start L-BFGS-B**.
        # Why? L-BFGS-B is very good at finding the minimum if you start in the basin of attraction.
        # The basin of attraction for Rosenbrock is the valley.
        # If we can find several points in the valley, L-BFGS-B will converge fast.
        # DE is slow to converge to the valley.
        # A simpler global search might be better: **Grid Search** or **Random Sampling** followed by L-BFGS-B.
        
        # Given the budget of 500, we can afford ~10-20 restarts of L-BFGS-B if each takes ~20-40 calls.
        # But we need to find the valley first.
        
        # Let's use a **Coarse Grid Search** on a reduced resolution to find the valley, then L-BFGS-B.
        # Or, **Sobol Sampling** for better space-filling than random.
        
        # Strategy:
        # 1. Generate ~100 Sobol points in [-5, 5]^5.
        # 2. Evaluate them.
        # 3. Keep the top 5 best points.
        # 4. Run L-BFGS-B from each of these 5 points, sharing the remaining budget.
        
        # This is "Multi-start L-BFGS-B" but with a structured initial sampling.
        
        # Let's implement this.
        
        # Generate Sobol points
        from scipy.stats import qmc
        
        # Number of initial samples
        n_samples = 100
        # Ensure we have enough budget for the rest
        if budget < 150:
            n_samples = int(budget * 0.3)
        
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        samples = sampler.random(n=n_samples)
        # Scale from [0,1] to [-5,5]
        samples = samples * 10 - 5
        
        # Evaluate initial samples
        candidates = []
        for x in samples:
            if call_count >= budget:
                break
            f_val = counted_func(x)
            candidates.append((f_val, x.copy()))
        
        # Sort by function value
        candidates.sort(key=lambda item: item[0])
        
        # Take top K candidates for polishing
        K = 5
        top_candidates = [x for f, x in candidates[:K]]
        
        # Phase 2: Multi-start L-BFGS-B Polish
        # Allocate remaining budget to K starts.
        remaining_budget = budget - call_count
        if remaining_budget > 0 and top_candidates:
            # We'll try to run L-BFGS-B on each top candidate.
            # We need to be careful with the call count.
            # L-BFGS-B in SciPy doesn't have a strict "max calls" option, only maxiter.
            # We can use a wrapper that stops if we exceed the allocated budget per start?
            # Or just let it run and hope it converges before the global budget runs out.
            # Since L-BFGS-B is efficient, it should converge quickly if in the basin.
            
            # To be safe, we can limit the number of iterations for L-BFGS-B.
            # For dim=5, 50-100 iterations should be enough if close.
            
            for x0 in top_candidates:
                if call_count >= budget - 5: # Leave a small buffer
                    break
                if time.time() > deadline:
                    break
                
                try:
                    res = minimize(
                        counted_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=bounds, 
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception as e:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
