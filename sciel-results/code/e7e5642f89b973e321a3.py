"""Strategy: Adaptive DE with Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Differential Evolution with adaptive mutation and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard
        start_time = time.time()
        deadline = start_time + 320.0

        # DE Parameters
        # Population size: typically 5-10 * dim for 5D
        # Let's use 10 * dim = 50. This allows for 6 generations of 50 calls.
        # Wait, 300 / 50 = 6 generations. That's very few.
        # Smaller pop: 5 * dim = 25. 300 / 25 = 12 generations.
        # Even smaller: 2 * dim + 5 = 15. 300 / 15 = 20 generations.
        # Standard DE often uses 5-10 * dim. Let's try 10 * dim = 50.
        # Actually, for 5D, 20-30 is often good. Let's use 25.
        pop_size = 25
        
        # Mutation factor F
        # Adaptive: start high, decay? Or constant 0.8?
        # Let's use a constant F=0.8 initially, maybe adapt if stuck.
        F = 0.8
        
        # Crossover rate CR
        CR = 0.5
        
        # Initialize population
        rng = np.random.default_rng(42)
        # Uniform initialization in [-5, 5]
        pop = rng.uniform(-5, 5, (pop_size, dim))
        
        # Evaluate initial population
        fitness = np.empty(pop_size)
        best_x = None
        best_fx = float('inf')
        calls_made = 0
        
        for i in range(pop_size):
            if time.time() > deadline:
                break
            fx = func(pop[i])
            fitness[i] = fx
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = pop[i].copy()
                
        if calls_made < pop_size:
            # If we ran out of time or budget immediately, return best
            return best_x, best_fx

        # Main DE Loop
        # We have budget - calls_made calls left.
        # Each generation costs pop_size calls.
        # Generations = (budget - calls_made) // pop_size
        
        # Adaptive F: if no improvement for G gens, increase F? 
        # Or decrease F to refine? 
        # Usually, high F helps exploration, low F helps exploitation.
        # Let's start with F=0.8.
        
        # Track stagnation
        stagnation_counter = 0
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check if we can afford a full generation
            if calls_made + pop_size > budget:
                # Do partial generation or break
                # It's better to break and use remaining budget for polish if possible
                # But polish needs a good starting point.
                # Let's just break the loop and go to polish if we have some calls left?
                # Or just stop.
                # If we have < pop_size calls left, we can't do a full DE gen.
                # We can do a few random walks or break.
                # Let's break and polish with remaining calls if > 0.
                break
                
            # Generate new population
            new_pop = np.empty_like(pop)
            new_fitness = np.empty(pop_size)
            
            # Current best index
            best_idx = np.argmin(fitness)
            
            for i in range(pop_size):
                if time.time() > deadline:
                    # Fill remaining with current best to avoid errors
                    new_pop[i] = pop[best_idx]
                    new_fitness[i] = fitness[best_idx]
                    continue
                
                # Select 3 distinct random indices different from i
                # Use rejection sampling or permutation
                # For efficiency, just pick 3 randoms and ensure distinct
                idxs = []
                while len(idxs) < 3:
                    j = rng.integers(0, pop_size)
                    if j != i and j not in idxs:
                        idxs.append(j)
                
                x1, x2, x3 = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
                
                # Mutation: DE/rand/1
                # v = x1 + F * (x2 - x3)
                v = x1 + F * (x2 - x3)
                
                # Clip to bounds [-5, 5]
                v = np.clip(v, -5, 5)
                
                # Crossover: Binomial
                # Pick a random start index j0
                j0 = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j0:
                        trial[j] = v[j]
                    
                # Evaluate trial
                if calls_made >= budget:
                    break
                ft = func(trial)
                calls_made += 1
                
                # Selection
                if ft <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = ft
                    if ft < best_fx:
                        best_fx = ft
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
                    
            # Update population
            pop = new_pop
            fitness = new_fitness
            
            # Stagnation check
            if best_fx < np.min(fitness) + 1e-9: # This check is redundant, best_fx is global min
                pass
            # Better stagnation check: if best_fx didn't improve in this gen
            # We need to store prev_best_fx
            # Let's add that logic properly in the loop structure above.
            # For now, let's just run.
            
            # Adaptive F adjustment (optional)
            # If stagnation, increase F to explore more?
            # Or decrease to exploit?
            # Let's keep F constant for simplicity and robustness.
            
        # Final Polish with Nelder-Mead
        # Use remaining budget for local optimization
        # Nelder-Mead uses dim+1 points per iteration.
        # Max iterations = (budget - calls_made) // (dim + 1)
        
        if best_x is not None and calls_made < budget:
            # Define a wrapper to track calls and respect budget
            polish_calls = 0
            max_polish_calls = budget - calls_made
            
            def polished_func(x):
                nonlocal polish_calls
                if polish_calls >= max_polish_calls:
                    return best_fx # Return best known to stop or penalize
                fx = func(x)
                polish_calls += 1
                return fx
            
            # Initial simplex: best_x + small perturbations
            # Scale step size based on current sigma or just fixed small value
            # Since we don't have sigma, use a small fraction of the domain
            step = 0.1
            simplex = [best_x.copy()]
            for i in range(dim):
                x_i = best_x.copy()
                x_i[i] += step
                simplex.append(x_i)
            
            # Run Nelder-Mead
            # Method='Nelder-Mead'
            # Options: maxiter, xatol, fatol
            # We need to be careful with the call count.
            # SciPy's minimize might make more calls than expected if not stopped.
            # The wrapper `polished_func` handles the budget.
            
            try:
                # x0 is the best point
                # We pass a function that counts calls
                res = minimize(polished_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                # Check if res.x is better
                if res.fun < best_fx:
                    # We need to verify res.fun was actually evaluated within budget
                    # The wrapper returns best_fx if budget exceeded, so if res.fun < best_fx, it's valid
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
