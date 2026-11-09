"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for BBOB-style Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        
        # Define multiple initial centers spread across the domain
        # Using 4 independent CMA-ES instances to balance exploration and exploitation
        n_starts = 4
        centers = []
        for _ in range(n_starts):
            # Uniform random centers within [-4, 4] to be safe, or [-5, 5]
            # Task says minimizer is in [-4, 4], so starting within [-5, 5] is fine.
            # Let's use a grid-like spread for better coverage initially.
            c = rng.uniform(-5, 5, dim)
            centers.append(c)
            
        # Initial sigma for each instance
        # Domain size is 10. A reasonable initial step is 1/4 of the domain or so.
        initial_sigma = 2.0
        
        # State for each instance: x (mean), sigma, C (covariance matrix), step_size
        # To keep it simple and fast, we implement a simplified CMA-ES 
        # or use scipy.optimize if available? No, standard library + numpy is preferred for robustness, 
        # but SciPy is available. However, implementing a lightweight CMA-ES is safer against version issues 
        # and allows fine-grained control over the budget.
        
        # Let's implement a simplified (1+1)-ES per start? No, CMA-ES is better for rotation.
        # But implementing full CMA-ES is complex. 
        # Alternative: Use a population-based evolution strategy with adaptive step size 
        # that approximates CMA-ES behavior without the full covariance update complexity?
        # Or just use multiple (1+1)-ES instances?
        
        # The current program is a single (1+1)-ES. 
        # A multi-start (1+1)-ES is a valid "different strategy" (diversity).
        # But CMA-ES is significantly better for rotated functions.
        # Let's try to implement a basic CMA-ES.
        
        # Actually, let's look at the inspiration 2 which used Nelder-Mead.
        # Schwefel is multimodal. Nelder-Mead can get stuck.
        # CMA-ES is the gold standard for BBOB.
        
        # Given the 300s limit and 500 calls, we have very few calls.
        # 500 calls / 4 starts = 125 calls per start.
        # CMA-ES with lambda = 4 * (4 + floor(3 * ln(5))) = 4 * (4 + 4) = 32 evaluations per generation.
        # 125 / 32 ~ 4 generations. This is very few for CMA-ES to converge.
        
        # Maybe a simpler approach: Multi-start with local search (Nelder-Mead) after a coarse search?
        # Or just a very aggressive multi-start (1+1)-ES with larger initial steps?
        
        # Let's stick to a robust Multi-Start (1+1)-ES but with better initialization and step size adaptation.
        # The key improvement over the single (1+1)-ES is diversity to avoid getting stuck in local minima 
        # or poor initial positions due to the hidden shift/rotation.
        
        # Strategy:
        # 1. Split budget among N starts.
        # 2. Each start runs a (1+1)-ES.
        # 3. Track global best.
        
        # To be "materially different", we should try a method that handles the landscape better.
        # How about a simple population-based DE (Differential Evolution)?
        # DE is robust to rotation and doesn't need covariance matrix updates.
        
        # Let's implement a simple DE (Rand/1/Bin) with adaptive scaling factor.
        # Population size: 10.
        # Generations: 500 / 10 = 50.
        
        pop_size = 10
        generations = budget // pop_size
        
        # Initialize population
        pop = rng.uniform(-5, 5, (pop_size, dim))
        fvals = np.array([func(x) for x in pop])
        
        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        
        # DE parameters
        F = 0.8  # Scaling factor
        CR = 0.9  # Crossover rate
        
        calls_made = pop_size
        
        for gen in range(generations):
            if time.time() > deadline:
                break
            
            # Sort population by fitness
            order = np.argsort(fvals)
            sorted_pop = pop[order]
            sorted_fvals = fvals[order]
            
            new_pop = np.empty_like(pop)
            new_pop[0] = sorted_pop[0] # Keep best
            new_fvals = np.empty_like(fvals)
            new_fvals[0] = sorted_fvals[0]
            
            for i in range(1, pop_size):
                if time.time() > deadline:
                    # Fill remaining with best
                    new_pop[i] = best_x
                    new_fvals[i] = best_f
                    continue
                
                # Select 3 distinct random individuals different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = rng.choice(candidates, 3, replace=False)
                
                # Create trial vector
                trial = sorted_pop[a] + F * (sorted_pop[b] - sorted_pop[c])
                trial = np.clip(trial, -5, 5)
                
                # Crossover
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = trial[j]
                    else:
                        trial[j] = sorted_pop[i][j]
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= sorted_fvals[i]:
                    new_pop[i] = trial
                    new_fvals[i] = f_trial
                else:
                    new_pop[i] = sorted_pop[i]
                    new_fvals[i] = sorted_fvals[i]
            
            # Update population
            pop = new_pop
            fvals = new_fvals
            
            # Update global best
            cur_best_idx = np.argmin(fvals)
            if fvals[cur_best_idx] < best_f:
                best_f = fvals[cur_best_idx]
                best_x = pop[cur_best_idx].copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
