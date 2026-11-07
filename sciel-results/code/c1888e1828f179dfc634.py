import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start with diverse initial points
        # Use stratified sampling for better coverage in 5D
        initial_points = []
        
        # 1. Center
        initial_points.append(np.zeros(dim))
        
        # 2. Stratified random sampling: divide each dimension into 3 bins
        # This ensures we don't cluster too much
        n_strat = 15
        for _ in range(n_strat):
            # Pick a random bin for each dimension
            bins = rng.integers(0, 3, dim)
            x = np.zeros(dim)
            for d in range(dim):
                # Uniform within the bin
                bin_start = -5.0 + bins[d] * (10.0 / 3.0)
                bin_end = bin_start + (10.0 / 3.0)
                x[d] = rng.uniform(bin_start, bin_end)
            initial_points.append(x)
            
        # 3. A few pure random points
        for _ in range(5):
            initial_points.append(rng.uniform(-5, 5, dim))

        # Evaluate all initial points
        evaluated = []
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 10.0:
                break
            x0 = clip(x0)
            fx = evaluate(x0)
            evaluated.append((fx, x0.copy()))
        
        if not evaluated:
            # Safety
            x0 = np.zeros(dim)
            evaluate(x0)
            evaluated.append((best_f, best_x.copy()))

        # Sort by fitness
        evaluated.sort(key=lambda t: t[0])

        # Phase 2: Local refinement using Adaptive (1+1)-ES on top candidates
        # We will refine the top 5 candidates
        top_k = min(5, len(evaluated))
        
        # Budget allocation for Phase 2
        # Reserve 20 calls for final polish
        reserve_budget = 20
        available_for_refinement = budget - calls_made - reserve_budget
        
        if available_for_refinement > 0:
            calls_per_candidate = max(10, available_for_refinement // top_k)
            
            for k in range(top_k):
                if calls_made >= budget or remaining_time() < 5.0:
                    break
                
                fx_init, x_init = evaluated[k]
                
                # Adaptive (1+1)-ES
                x = x_init.copy()
                fx = fx_init
                sigma = 1.0  # Initial step size, roughly 10% of box width
                
                # Success rate tracking
                window_size = 10
                successes = []
                
                max_steps = calls_per_candidate
                
                for _ in range(max_steps):
                    if calls_made >= budget or remaining_time() < 2.0:
                        break
                    
                    # Generate trial
                    y = x + sigma * rng.standard_normal(dim)
                    y = clip(y)
                    fy = evaluate(y)
                    
                    # Update success window
                    if fy < fx:
                        successes.append(1)
                        x = y
                        fx = fy
                        # Increase sigma on success
                        sigma *= 1.1
                    else:
                        successes.append(0)
                        # Decrease sigma on failure
                        sigma *= 0.9
                        
                    # Keep window size constant
                    if len(successes) > window_size:
                        successes.pop(0)
                        
                    # Adapt sigma based on success rate
                    if len(successes) == window_size:
                        sr = sum(successes) / window_size
                        # Target ~0.25 success rate
                        if sr > 0.4:
                            sigma *= 1.05
                        elif sr < 0.1:
                            sigma *= 0.95
                            
                    # Clamp sigma
                    sigma = np.clip(sigma, 1e-4, 5.0)

                # Update global best if this candidate improved
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Final fine-tuning with very small steps on the global best
        if calls_made < budget and remaining_time() > 5.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05 # Start small
            
            # Use remaining calls
            while calls_made < budget and remaining_time() > 1.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                else:
                    sigma *= 0.95
                    if sigma < 1e-6:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
