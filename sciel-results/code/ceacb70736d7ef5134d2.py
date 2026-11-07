"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Differential Evolution with Nelder-Mead polish for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        
        def time_left():
            return deadline - time.time()
            
        def de_search(func, x0, f0, max_calls, deadline, dim, rng):
            """
            Run Differential Evolution from x0 with f0.
            Returns (best_x, best_f, calls_used).
            """
            # DE Parameters
            # Population size 20 is good for dim=5, balancing exploration and exploitation
            pop_size = 20
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            
            # Initialize population around x0
            # We use a spread to ensure global coverage initially, but centered on the best known point
            # Since we call this from multiple starts, we can be a bit more local or global depending on strategy.
            # For robustness, let's initialize with a mix of local and global.
            # Actually, standard DE initializes the whole pop.
            # If x0 is a good point, we can put it in the pop and fill the rest randomly in a larger box or around x0.
            # Given the box is [-5,5] and opt is in [-4,4], random init in [-5,5] is safe.
            
            # Initialize population
            pop = np.random.uniform(-5, 5, size=(pop_size, dim))
            # Ensure x0 is in the population
            pop[0] = x0
            f_pop = np.array([func(x) for x in pop])
            calls_used = pop_size
            
            best_idx = np.argmin(f_pop)
            local_best_x = pop[best_idx].copy()
            local_best_f = f_pop[best_idx]
            
            # Update global best if improved
            if local_best_f < f0:
                f0 = local_best_f
                x0 = local_best_x.copy()
                
            # Sort population initially to maintain "best" at index 0 for simplicity?
            # Standard DE doesn't require sorted population, but it helps for selection.
            # Let's keep track of indices.
            
            while calls_used < max_calls and time_left() > 1.0:
                # Generate offspring for each individual
                new_pop = np.zeros_like(pop)
                new_f_pop = np.zeros(pop_size)
                
                for i in range(pop_size):
                    # Select three distinct random individuals r1, r2, r3
                    # Exclude i
                    candidates = [j for j in range(pop_size) if j != i]
                    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                    
                    # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    
                    # Crossover: binomial crossover
                    # For each dimension, decide whether to take from v or x_i
                    j_rand = rng.integers(0, dim)
                    trial = pop[i].copy()
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    
                    # Clip to box
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate
                    f_trial = func(trial)
                    calls_used += 1
                    
                    # Selection
                    if f_trial <= f_pop[i]:
                        new_pop[i] = trial
                        new_f_pop[i] = f_trial
                    else:
                        new_pop[i] = pop[i]
                        new_f_pop[i] = f_pop[i]
                        
                pop = new_pop
                f_pop = new_f_pop
                
                # Update best
                idx_min = np.argmin(f_pop)
                if f_pop[idx_min] < local_best_f:
                    local_best_f = f_pop[idx_min]
                    local_best_x = pop[idx_min].copy()
                    
            return local_best_x, local_best_f, calls_used

        def nelder_mead_polish(func, x0, f0, max_calls, deadline, dim):
            """
            Run Nelder-Mead from x0.
            Returns (best_x, best_f, calls_used).
            """
            if max_calls < 10 or time_left() < 1.0:
                return x0, f0, 0
            
            # Use scipy's minimize for robustness
            # We need to count function calls
            call_count = [0]
            
            def wrapped_func(x):
                call_count[0] += 1
                if call_count[0] > max_calls or time_left() < 0.5:
                    raise Exception("Budget exceeded")
                return func(x)
                
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_calls, 'xatol': 1e-6, 'fatol': 1e-6})
                calls_used = call_count[0]
                if res.success or res.fun < f0:
                    return res.x, res.fun, calls_used
                else:
                    return x0, f0, calls_used
            except Exception:
                return x0, f0, call_count[0]

        # Phase 1: Global Search with DE
        # Use 80% of budget for DE
        de_budget = int(budget * 0.8)
        
        # Start DE from a few diverse points?
        # Or just one DE run? DE is population-based, so it handles diversity internally.
        # Let's do one main DE run, but initialize it with a few random points to seed it?
        # The DE implementation above initializes the whole pop randomly.
        # This is a good global strategy.
        
        x_start = np.zeros(dim)
        f_start = func(x_start)
        calls_made += 1
        if f_start < best_f:
            best_f = f_start
            best_x = x_start.copy()
            
        # Run DE
        x_de, f_de, used_de = de_search(func, x_start, f_start, de_budget, deadline, dim, rng)
        calls_made += used_de
        
        if f_de < best_f:
            best_f = f_de
            best_x = x_de.copy()
            
        # Phase 2: Polish with Nelder-Mead
        # Use remaining budget for NM
        nm_budget = budget - calls_made
        if nm_budget > 10 and time_left() > 5.0:
            x_nm, f_nm, used_nm = nelder_mead_polish(func, best_x, best_f, nm_budget, deadline, dim)
            calls_made += used_nm
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
                
        return best_x, best_f

# EVOLVE-BLOCK-END
