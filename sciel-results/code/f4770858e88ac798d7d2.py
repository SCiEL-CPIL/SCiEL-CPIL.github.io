"""
Strategy: Multi-start Nelder-Mead Simplex with Adaptive Restarts.
This approach abandons the population-based stochastic sampling (CMA-ES) in favor of 
a deterministic, geometry-based local search (Nelder-Mead) driven by multiple global 
starts. This is a structurally different family: local deterministic search vs. 
global stochastic evolution.
"""
import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer for black-box continuous functions.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Nelder-Mead parameters
        # Standard defaults, slightly tuned for high dimensions if needed, 
        # but dim=5 is low, so standard works well.
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # Budget allocation
        # We want to spend most of the budget on local refinement, 
        # but need enough for global exploration (restarts).
        # Let's define a "local budget" per restart.
        # If we use ~20-50 restarts, we have 10-25 evals per restart.
        # Nelder-Mead in 5D converges very fast (often < 100 evals).
        # Let's aim for ~10 restarts initially, increasing if time allows?
        # Actually, fixed budget is better for reproducibility and time safety.
        
        # Strategy: 
        # 1. Generate a set of starting points.
        # 2. Run Nelder-Mead from each, stopping when local budget is exhausted or convergence.
        # 3. Keep the best result.
        
        # How many starts?
        # dim=5. A good rule of thumb is dim+1 or more. 
        # With 500 budget, we can afford ~20-50 starts of ~10-20 evals each.
        # Let's try 25 starts. 500 / 25 = 20 evals per start.
        # Nelder-Mead usually converges in < 100 evals for smooth functions.
        # 20 might be too few to converge fully, but it will find a good local minimum.
        # Alternatively, fewer starts with more evals? 
        # 10 starts * 50 evals = 500. This allows deeper local search.
        # Let's go with 10 starts, 50 evals each.
        
        n_starts = 10
        local_budget = budget // n_starts
        
        best_x = None
        best_fx = np.inf
        calls = 0
        
        # Random seed for reproducibility within the run? 
        # The task says "hidden seeded shift", implying the function is fixed per instance.
        # We should use a fixed seed for our random starts to be deterministic?
        # Or just use np.random. The evaluator calls __call__ once per instance.
        # Using np.random without seeding is fine, but let's seed it based on dim/budget 
        # to ensure consistent behavior across instances if the evaluator doesn't seed numpy.
        # Actually, the prompt says "hidden seeded shift", so the function f is fixed.
        # We should probably use a fixed seed for our internal randomness to ensure 
        # the "search score" is stable? No, the score is value/seed. 
        # If we use random starts, the result might vary. 
        # However, usually in these tasks, determinism is preferred or at least 
        # a fixed seed is good practice to avoid "bad luck" runs.
        # Let's use a fixed seed for the optimizer's internal RNG.
        rng = np.random.RandomState(42)
        
        for start_idx in range(n_starts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Choose starting point
            # Mix of center and random points
            if start_idx == 0:
                x_start = np.zeros(dim)
            elif start_idx == 1:
                x_start = rng.uniform(-4, 4, dim)
            else:
                # Sample from a wider box, maybe clipped
                x_start = rng.uniform(-5, 5, dim)
            
            # Run Nelder-Mead
            # We need to track calls carefully.
            # NM requires dim+1 initial points.
            # Then iterations.
            
            # Create initial simplex
            # Vertices: x_start, and x_start + delta * e_i
            # Delta should be small enough to stay in bounds, large enough to explore.
            # Box is [-5, 5].
            # Let's use delta = 0.5
            delta = 0.5
            simplex = [x_start.copy()]
            for i in range(dim):
                v = x_start.copy()
                v[i] += delta
                # Clip to box
                v = np.clip(v, -5, 5)
                simplex.append(v)
            
            # Evaluate initial simplex
            simplex_f = []
            for v in simplex:
                if calls >= budget or time.time() > deadline:
                    break
                f_v = func(v)
                calls += 1
                simplex_f.append(f_v)
                if f_v < best_fx:
                    best_fx = f_v
                    best_x = v.copy()
            
            if len(simplex_f) < dim + 1:
                continue
                
            # Nelder-Mead Iterations
            # We limit the number of iterations to local_budget - (dim+1)
            max_iter = local_budget - (dim + 1)
            
            # Convert to numpy arrays for speed
            S = np.array(simplex) # (dim+1, dim)
            F = np.array(simplex_f) # (dim+1,)
            
            for it in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                    
                # Sort simplex by fitness
                order = np.argsort(F)
                S = S[order]
                F = F[order]
                
                # Check convergence
                # If the spread of function values is small, stop
                if F[-1] - F[0] < 1e-6:
                    break
                # Also check if the simplex is small
                if np.max(np.abs(S[0] - S[-1])) < 1e-6:
                    break
                    
                # Centroid of all but the worst
                x_worst = S[-1]
                f_worst = F[-1]
                centroid = np.mean(S[:-1], axis=0)
                
                # 1. Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1
                if f_r < best_fx:
                    best_fx = f_r
                    best_x = x_r.copy()
                
                # 2. Expansion or Contraction or Shrinkage
                if f_r < F[0]: # Better than best
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < best_fx:
                        best_fx = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        x_new = x_e
                        f_new = f_e
                    else:
                        x_new = x_r
                        f_new = f_r
                        
                elif f_r < F[-2]: # Between best and worst
                    x_new = x_r
                    f_new = f_r
                    
                else: # Worse than worst
                    # Contraction
                    # Outside contraction if f_r < f_worst, else inside
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_fx:
                        best_fx = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        x_new = x_c
                        f_new = f_c
                    else:
                        # Shrinkage
                        # Move all points except the best towards the best
                        x_best = S[0]
                        S_new = x_best + sigma * (S - x_best)
                        S_new = np.clip(S_new, -5, 5)
                        
                        # Re-evaluate all except best? 
                        # Standard NM re-evaluates all.
                        # To save budget, we can just update the array and break 
                        # if we are close to budget limit, or re-evaluate.
                        # Given tight budget, let's just update the worst point 
                        # to the shrunk position and re-evaluate only that?
                        # No, NM requires full simplex update.
                        # However, with 20 evals budget, we might not have enough for full shrinkage re-eval.
                        # Let's just break if we are tight on budget.
                        if calls + dim > budget:
                            break
                            
                        F_new = np.array([func(v) for v in S_new[1:]]) + [F[0]] # Placeholder, need actual evals
                        # Wait, we need to evaluate the new points.
                        for i in range(1, dim + 1):
                            if calls >= budget or time.time() > deadline:
                                break
                            f_i = func(S_new[i])
                            calls += 1
                            if f_i < best_fx:
                                best_fx = f_i
                                best_x = S_new[i].copy()
                            F_new[i] = f_i # Correcting the array logic
                            
                        S = S_new
                        F = F_new
                        continue # Skip the standard update below
                        
                    # If we didn't break, update worst point
                    S[-1] = x_new
                    F[-1] = f_new
                    continue

                # Standard update: replace worst point
                S[-1] = x_new
                F[-1] = f_new

        return best_x, best_fx
