import numpy as np
import scipy.optimize
import scipy.stats
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. 
        # We aim to finish well before. 295s gives a 35s buffer.
        deadline = start_time + 295.0 
        call_count = 0
        best_f = float('inf')
        best_x = None

        def wrapped_func(x):
            nonlocal call_count, best_f, best_x
            # Check budget and time
            if call_count >= self.budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            
            call_count += 1
            try:
                # Ensure x is a numpy array and clip to bounds
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (self.dim,):
                    x_arr = x_arr.flatten()[:self.dim]
                x_clipped = np.clip(x_arr, -5, 5)
                
                fx = float(func(x_clipped))
                
                if np.isnan(fx) or np.isinf(fx):
                    return 1e10
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                
                return fx
            except Exception:
                return 1e10

        # Phase 1: Global Initialization
        # Reduced from 100 to 50 to allow more local search iterations.
        # 50 points is sufficient for 5D to find basins of attraction for most BBOB functions.
        n_initial = 50
        if self.budget < n_initial:
            n_initial = max(10, self.budget - 10)
        
        # Generate Sobol points (low discrepancy)
        try:
            sobol = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
            n_sobol = n_initial // 2
            points_sobol = sobol.random(n_sobol) * 10.0 - 5.0
        except Exception:
            points_sobol = np.empty((0, self.dim))
            n_sobol = 0

        # Generate Random points for the rest
        n_random = n_initial - n_sobol
        rng = np.random.default_rng(42)
        points_random = rng.uniform(-5, 5, (n_random, self.dim))
        
        # Combine
        all_initial_points = np.vstack([points_sobol, points_random])
        
        # Shuffle to avoid bias
        rng.shuffle(all_initial_points)

        candidates = []
        
        # Evaluate initial points
        for x in all_initial_points:
            if call_count >= self.budget or time.time() > deadline:
                break
            fx = wrapped_func(x)
            candidates.append((fx, x))
        
        if not candidates:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization on Top Candidates
        # We have ~250 calls left. 
        # L-BFGS-B is very efficient in 5D. 
        # We can afford to run it on more candidates.
        top_k = min(12, len(candidates))
        
        remaining_budget = self.budget - call_count
        # Allocate budget per local search. 
        # Reserve 20 calls for final polish/exploration.
        local_budget_total = max(0, remaining_budget - 20)
        local_budget_per = local_budget_total // top_k if top_k > 0 else 0
        
        # Ensure each local search has at least a minimum number of iterations
        # L-BFGS-B usually converges in < 50 evals for 5D smooth functions.
        local_budget_per = max(20, min(local_budget_per, 80))

        local_results = []
        
        for i in range(top_k):
            if call_count >= self.budget or time.time() > deadline:
                break
            
            start_x = candidates[i][1]
            
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100, 
                        'maxfun': local_budget_per, 
                        'ftol': 1e-10, 
                        'gtol': 1e-8, 
                        'disp': False
                    }
                )
                if res.fun < 1e9: # Valid result
                    local_results.append((res.fun, res.x))
            except Exception:
                # If L-BFGS-B fails, keep the initial point
                local_results.append((candidates[i][0], start_x))
        
        # Update best
        if local_results:
            local_results.sort(key=lambda t: t[0])
            best_local_f, best_local_x = local_results[0]
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x

        # Phase 3: Exploration & Polish
        # If we have significant budget left, try to escape potential local minima 
        # (relevant for Rastrigin/Gallagher) or polish the best solution.
        remaining = self.budget - call_count
        if remaining > 10 and best_x is not None and time.time() < deadline - 2:
            
            # Strategy: Try a few random perturbations around the best point.
            # If a perturbation leads to a lower value, run a quick local search on it.
            n_perturbations = min(5, remaining // 10)
            
            for _ in range(n_perturbations):
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                # Gaussian perturbation
                # Scale depends on dimension, 0.5 is a reasonable start for [-5,5]
                perturbation = rng.standard_normal(self.dim) * 0.5
                x_try = np.clip(best_x + perturbation, -5, 5)
                
                fx_try = wrapped_func(x_try)
                
                if fx_try < best_f:
                    best_f = fx_try
                    best_x = x_try.copy()
                    
                    # If we found a better point, polish it immediately
                    if call_count < self.budget - 5 and time.time() < deadline - 2:
                        try:
                            polish_budget = min(20, self.budget - call_count)
                            res = scipy.optimize.minimize(
                                wrapped_func,
                                best_x,
                                method='L-BFGS-B',
                                bounds=[(-5, 5)] * self.dim,
                                options={
                                    'maxiter': 50,
                                    'maxfun': polish_budget,
                                    'ftol': 1e-10,
                                    'gtol': 1e-8,
                                    'disp': False
                                }
                            )
                            if res.fun < best_f:
                                best_f = res.fun
                                best_x = np.clip(res.x, -5, 5)
                        except Exception:
                            pass
                else:
                    # If perturbation didn't help, we still used a call.
                    # Continue to next perturbation.
                    pass
        
        # Final fallback if no valid result
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < self.budget:
                try:
                    best_f = wrapped_func(best_x)
                except Exception:
                    best_f = 1e10

        return best_x, best_f
