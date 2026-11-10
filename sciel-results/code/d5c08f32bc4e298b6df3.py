import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 280  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization with diverse sampling
        # Use 10% of budget for initial population
        init_pop_size = max(10, self.budget // 10)
        
        # Simple Sobol-like or just random with stratification
        # For 4 dims, random is okay, but let's try to cover corners
        rng = np.random.default_rng(42)
        
        # Create initial points
        # Mix of random and boundary points to help find extremes
        X_init = []
        # Random
        for _ in range(init_pop_size // 2):
            X_init.append(rng.random(self.dim) * span + lo)
        # Boundary focused (helps find Pareto extremes)
        for _ in range(init_pop_size - len(X_init)):
            x = rng.random(self.dim)
            # Force 20% of dimensions to be near bounds
            mask = rng.random(self.dim) < 0.2
            if np.any(mask):
                x[mask] = rng.random(np.sum(mask)) # 0 or 1
                x[mask] = x[mask] * span[mask] + lo[mask]
            else:
                x = x * span + lo
            X_init.append(x)
            
        X_init = np.array(X_init)
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        
        # Build initial Pareto Front
        pareto_front = []
        
        for x, f in zip(X_init, F_init):
            dominated = False
            for _, f_old in pareto_front:
                if np.all(f_old <= f) and np.any(f_old < f):
                    dominated = True
                    break
            if not dominated:
                # Add new point
                pareto_front.append((x, f))
                # Remove dominated points
                pareto_front = [
                    (x_p, f_p) for x_p, f_p in pareto_front 
                    if not (np.all(f <= f_p) and np.any(f < f_p))
                ]
        
        used_evals = len(X_init)
        
        # 2. Main Loop
        sigma = 0.1 # Initial step size relative to span
        stagnation_counter = 0
        
        while used_evals < self.budget:
            if time.time() - start_time > timeout:
                break
                
            if not pareto_front:
                # Fallback: random restart
                x = rng.random(self.dim) * span + lo
                f = problem(x)
                pareto_front.append((x, f))
                used_evals += 1
                continue

            # Select parent based on Hypervolume contribution (approximation)
            # To keep it fast, we can just pick randomly or use a simple weight
            # Let's try to pick the point with the largest "gap" to neighbors
            # Or simply random is often good enough. Let's stick to random for speed.
            parent_x, parent_f = pareto_front[rng.integers(len(pareto_front))]
            
            # Mutate
            # Use Gaussian noise
            noise = rng.standard_normal(self.dim) * sigma * span
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate
            child_f = problem(child_x)
            used_evals += 1
            
            # Update Pareto Front
            dominated = False
            for _, f_old in pareto_front:
                if np.all(f_old <= child_f) and np.any(f_old < child_f):
                    dominated = True
                    break
            
            if not dominated:
                # Add child
                pareto_front.append((child_x, child_f))
                # Remove dominated
                pareto_front = [
                    (x_p, f_p) for x_p, f_p in pareto_front 
                    if not (np.all(child_f <= f_p) and np.any(child_f < f_p))
                ]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                # If stagnating, increase sigma to escape local optima
                if stagnation_counter > 30:
                    sigma *= 1.5
                    if sigma > 0.5:
                        sigma = 0.5
                    stagnation_counter = 0 # Reset to avoid immediate re-trigger
            
            # Decay sigma slowly when making progress
            if stagnation_counter == 0:
                sigma *= 0.99 
                if sigma < 0.001:
                    sigma = 0.001

        # 3. Polish phase: Tchebycheff decomposition with L-BFGS-B
        # This is a refinement step to squeeze out more performance
        # We have remaining budget. Let's use it to polish the best points.
        
        if used_evals < self.budget and time.time() - start_time < timeout:
            # Sort front by HV contribution (approx) or just take top N
            # Since HV contribution is hard to compute exactly fast, let's just take a few random ones or the ones with best individual objectives
            # Better: Use the hypervolume of the set to guide?
            # Let's just pick the 3 best points in terms of sum of objectives (normalized)
            
            # Normalize objectives to [0,1] for ranking
            min_f = np.min([f for _, f in pareto_front], axis=0)
            max_f = np.max([f for _, f in pareto_front], axis=0)
            span_f = max_f - min_f
            span_f[span_f == 0] = 1.0
            
            ranked_front = []
            for x, f in pareto_front:
                norm_f = (f - min_f) / span_f
                score = np.sum(norm_f)
                ranked_front.append((score, x, f))
            
            ranked_front.sort(key=lambda x: x[0])
            
            # Pick top 3 candidates for polishing
            candidates = [item for item in ranked_front[:3]]
            
            # Tchebycheff decomposition: use a set of weights
            # For 2 objectives, weights are (w, 1-w)
            # Use a grid of weights to cover the front
            n_weights = 10
            weights = np.linspace(0.05, 0.95, n_weights)
            
            # We need to track evaluations carefully
            # L-BFGS-B will call the objective function multiple times
            # We need to ensure we don't exceed the budget
            
            # Let's use a wrapper to count evaluations
            eval_count = [0]
            
            def counting_problem(x):
                if used_evals + eval_count[0] >= self.budget:
                    raise Exception("Budget exceeded")
                eval_count[0] += 1
                return problem(x)
            
            bounds_scipy = list(zip(lo, hi))
            
            for x_start, f_start in candidates:
                if used_evals + eval_count[0] >= self.budget:
                    break
                if time.time() - start_time > timeout:
                    break
                
                for w in weights:
                    if used_evals + eval_count[0] >= self.budget:
                        break
                    if time.time() - start_time > timeout:
                        break
                    
                    # Tchebycheff aggregation: min max w_i * (f_i - z_i)
                    # z_i is the ideal point (best value for each objective)
                    z = np.min([f for _, f in pareto_front], axis=0)
                    
                    def tchebycheff_obj(x):
                        fx = counting_problem(x)
                        return np.max(w * (fx - z))
                    
                    try:
                        res = minimize(tchebycheff_obj, x_start, method='L-BFGS-B', 
                                       bounds=bounds_scipy, options={'gtol': 1e-08, 'maxiter': 50})
                        if res.success:
                            x_opt = res.x
                            f_opt = problem(x_opt)
                            used_evals += 1 # Final evaluation
                            
                            # Update Pareto Front
                            dominated = False
                            for _, f_old in pareto_front:
                                if np.all(f_old <= f_opt) and np.any(f_old < f_opt):
                                    dominated = True
                                    break
                            
                            if not dominated:
                                # Add child
                                pareto_front.append((x_opt, f_opt))
                                # Remove dominated
                                pareto_front = [
                                    (x_p, f_p) for x_p, f_p in pareto_front 
                                    if not (np.all(f_opt <= f_p) and np.any(f_opt < f_p))
                                ]
                    
                    except Exception:
                        pass

        # Return value is ignored, but must not crash
        return pareto_front
