"""Portfolio search for BBOB f7 (Step ellipsoidal) with population-based adaptive step search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based search with adaptive step sizes for plateau-aware optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Population size
        pop_size = 10
        
        # Initialize population
        # Mix of random points in [-5, 5] and [-2, 2] to bias towards center
        population = []
        for i in range(pop_size):
            if i < pop_size // 2:
                x = rng.uniform(-5.0, 5.0, dim)
            else:
                x = rng.uniform(-2.0, 2.0, dim)
            fx = func(x)
            # Initialize sigma: larger for outer points, smaller for inner
            sigma = 1.0 if i < pop_size // 2 else 0.5
            population.append({'x': x, 'f': fx, 'sigma': sigma, 'no_improve': 0})
        
        calls = pop_size

        # Global best
        best_individual = min(population, key=lambda p: p['f'])
        best_x = best_individual['x'].copy()
        best_f = best_individual['f']
        
        # Stagnation tracking
        global_no_improve = 0
        stagnation_limit = 20 # If no global improvement in 20 calls, inject diversity
        
        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Process each individual in the population
            # We might not be able to process all if budget runs out mid-iteration
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                ind = population[i]
                x = ind['x']
                fx = ind['f']
                sigma = ind['sigma']
                
                # Generate trial
                # Adaptive sigma: if stuck, increase; if improving, decrease
                # But we just adjust sigma based on the previous step's outcome (stored in no_improve)
                if ind['no_improve'] > 5:
                    # Stuck for a while, increase step to escape plateau
                    sigma *= 1.5
                elif ind['no_improve'] == 0:
                    # Just improved, decrease step to refine
                    sigma *= 0.8
                else:
                    # Neutral
                    pass
                
                sigma = min(max(sigma, 0.01), 5.0)
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                calls += 1
                
                if fy < fx - 1e-10:
                    # Improvement
                    ind['x'] = y
                    ind['f'] = fy
                    ind['sigma'] = sigma # Keep the successful sigma
                    ind['no_improve'] = 0
                    
                    # Update global best
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        global_no_improve = 0
                else:
                    # No improvement
                    ind['no_improve'] += 1
                    # Do not update position, keep x and fx
                    # Sigma will be adjusted in next iteration based on no_improve count
                    # But we can increase it now to help next step? 
                    # Actually, the logic above adjusts sigma *before* the step based on previous state.
                    # So if no_improve was 0, we reduced sigma. If no_improve > 5, we increased it.
                    # This is fine.
                    
                    global_no_improve += 1

            # Stagnation check: if global best hasn't improved, replace worst individuals
            if global_no_improve >= stagnation_limit:
                # Sort population by fitness
                population.sort(key=lambda p: p['f'])
                # Replace the worst half with new random points
                n_replace = pop_size // 2
                for i in range(n_replace):
                    # Replace worst (at end of sorted list)
                    # Actually, we want to replace the worst, which are at the end of the sorted list (highest f)
                    # Let's just create new individuals and swap
                    new_x = rng.uniform(-5.0, 5.0, dim)
                    new_f = func(new_x)
                    calls += 1
                    if calls >= budget:
                        break
                    
                    # Replace the worst individual (last in sorted list)
                    # But we modified the list order. Let's just pick the worst from original indices?
                    # Simpler: just overwrite the last n_replace entries in the sorted list
                    # And then re-sort? No, just overwrite the worst ones.
                    # Since we sorted ascending, worst are at the end.
                    population[-(n_replace-i)] = {
                        'x': new_x,
                        'f': new_f,
                        'sigma': 1.0,
                        'no_improve': 0
                    }
                
                # Update global best if any new point was better (unlikely but possible)
                current_best_in_pop = min(population, key=lambda p: p['f'])
                if current_best_in_pop['f'] < best_f:
                    best_f = current_best_in_pop['f']
                    best_x = current_best_in_pop['x'].copy()
                
                # Reset stagnation counter
                global_no_improve = 0
                # Maybe increase stagnation limit slightly?
                # stagnation_limit = min(stagnation_limit + 5, 50)

        return best_x, best_f
# EVOLVE-BLOCK-END
