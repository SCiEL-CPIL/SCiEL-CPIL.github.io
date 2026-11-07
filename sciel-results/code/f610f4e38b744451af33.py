import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        # Phase 1: Exploration
        n_init = 50
        points = np.random.uniform(-5, 5, (n_init, self.dim))
        values = np.array([func(p) for p in points])
        
        best_idx = np.argmin(values)
        best_x = points[best_idx].copy()
        best_f = values[best_idx]
        
        # Sort initial points to have a list of "promising" starts
        sorted_indices = np.argsort(values)
        start_candidates = [points[i].copy() for i in sorted_indices[:5]]
        
        calls_used = n_init
        remaining_budget = self.budget - calls_used
        
        # Phase 2: Exploitation with (mu+lambda)-ES
        # We will try to refine the best candidate. If stuck, try next.
        # To keep it simple and robust, we just run one main ES but with restarts if stagnation occurs.
        
        mu = 4
        lambda_ = 20
        sigma = 1.0
        x_mean = best_x.copy()
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        stagnation_count = 0
        stagnation_limit = 40
        
        while calls_used < self.budget and (time.time() - start_time) < deadline:
            # Generate offspring
            offspring = []
            offspring_vals = []
            
            for _ in range(lambda_):
                z = np.random.standard_normal(self.dim)
                y = x_mean + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                offspring.append(y)
                offspring_vals.append(fy)
                calls_used += 1
                
                if calls_used >= self.budget or (time.time() - start_time) >= deadline:
                    break
            
            if not offspring:
                break
                
            # Sort offspring by fitness
            indices = np.argsort(offspring_vals)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_vals = [offspring_vals[i] for i in indices]
            
            # Check improvement
            if sorted_vals[0] < best_f:
                best_x = sorted_offspring[0].copy()
                best_f = sorted_vals[0]
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            # Update mean using recombination of top mu
            if len(sorted_offspring) >= mu:
                x_mean = np.average(sorted_offspring[:mu], axis=0, weights=weights)
                
            # Adapt sigma
            # Success rate: fraction of offspring better than the previous mean (or best?)
            # Standard 1/5 rule is for (1+1). For (mu+lambda), we can use:
            # If the best offspring is better than the mean, we might increase sigma?
            # Let's use a simple heuristic:
            # If best improved, increase sigma slightly to explore more? No, usually decrease to converge.
            # If no improvement, decrease sigma.
            # Better: Use the success rate of the top mu offspring relative to the parent mean.
            # Here, let's just use:
            # If best_f improved, sigma *= 1.1 (explore more around new best? Or converge? 
            # Actually, for Rastrigin, we want to find the global min. 
            # If we are improving, we are in a good basin. We can keep sigma or reduce it to fine-tune.
            # If we are NOT improving, we are stuck. Reduce sigma to try to find a better direction? 
            # Or restart?
            
            # Let's use a standard adaptation:
            # Count how many of the top mu are better than the current mean?
            # Simpler: 
            # If best improved: sigma = max(0.1, sigma * 0.9) # Converge
            # If not improved: sigma = min(5.0, sigma * 1.1) # Expand? 
            # This is tricky.
            
            # Let's try a different adaptation:
            # If the best offspring is significantly better, reduce sigma.
            # If the best offspring is similar, keep sigma.
            # If the best offspring is worse, reduce sigma? No, that's (1+1) logic.
            
            # Let's stick to a robust strategy:
            # If improved: sigma *= 0.95
            # If not improved: 
            #   stagnation_count += 1
            #   if stagnation_count > stagnation_limit:
            #       # Restart from next candidate
            #       if start_candidates:
            #           x_mean = start_candidates.pop(0).copy()
            #           sigma = 2.0
            #           stagnation_count = 0
            #       else:
            #           # Random restart near best
            #           x_mean = best_x + np.random.uniform(-1, 1, self.dim)
            #           x_mean = np.clip(x_mean, -5, 5)
            #           sigma = 1.0
            #           stagnation_count = 0
            #       sigma *= 0.9 # Reduce exploration?
            #   else:
            #       sigma *= 0.95 # Slowly contract
            
            # Let's refine the adaptation logic for Rastrigin.
            # Rastrigin has many local minima. We need to jump between them.
            # A large sigma is good for jumping. A small sigma is good for converging.
            
            if sorted_vals[0] < best_f:
                sigma *= 0.9
                if sigma < 0.05:
                    sigma = 0.05
            else:
                stagnation_count += 1
                if stagnation_count > stagnation_limit:
                    # Try to restart
                    if start_candidates:
                        x_mean = start_candidates.pop(0).copy()
                    else:
                        x_mean = best_x + np.random.uniform(-0.5, 0.5, self.dim)
                        x_mean = np.clip(x_mean, -5, 5)
                    sigma = 1.5
                    stagnation_count = 0
                else:
                    sigma *= 0.95
                    if sigma < 0.05:
                        sigma = 0.05

        return best_x, best_f
