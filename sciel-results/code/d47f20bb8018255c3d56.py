"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Rastrigin-aware optimizer using Differential Evolution for global search
    and Nelder-Mead for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit

        # Call counter and wrapper
        calls_made = 0
        def counted_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            try:
                val = func(x)
                return float(val)
            except Exception:
                return np.inf

        # --- Phase 1: Differential Evolution (Global Search) ---
        # DE is robust to rotation and multimodality.
        # Population size: 10 (2*dim)
        # Generations: 30
        # Total DE evals: 300
        pop_size = 10
        n_generations = 30
        
        # Initialize population uniformly in [-5, 5]
        # The problem states minimizer is in [-4, 4], but box is [-5, 5].
        # Using [-5, 5] ensures we don't miss edge cases, but [-4, 4] is a strong prior.
        # Let's use [-4.5, 4.5] to be safe but focused.
        x_min = np.full(dim, -4.5)
        x_max = np.full(dim, 4.5)
        
        # Generate initial population
        pop = np.random.uniform(x_min, x_max, size=(pop_size, dim))
        
        # Evaluate initial population
        pop_f = np.array([counted_func(x) for x in pop])
        
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE Parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        for gen in range(n_generations):
            if time.time() > deadline or calls_made >= budget:
                break

            current_best_idx = np.argmin(pop_f)
            current_best_x = pop[current_best_idx]
            current_best_f = pop_f[current_best_idx]
            
            # Update global best if improved
            if current_best_f < best_f:
                best_f = current_best_f
                best_x = current_best_x.copy()

            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Select 3 distinct individuals other than i
                r1, r2, r3 = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # Differential Evolution: v = x_r1 + F * (x_r2 - x_r3)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Clip to bounds
                v = np.clip(v, x_min, x_max)
                
                # Crossover: u_j = v_j if rand < CR or j == j_rand else x_i_j
                j_rand = np.random.randint(0, dim)
                u = pop[i].copy()
                mask = np.random.random(dim) < CR
                mask[j_rand] = True  # Ensure at least one gene from v
                u[mask] = v[mask]
                
                # Evaluate trial vector
                u_f = counted_func(u)
                
                # Selection
                if u_f < pop_f[i]:
                    pop[i] = u
                    pop_f[i] = u_f

        # --- Phase 2: Local Refinement with Nelder-Mead ---
        # Use remaining budget for Nelder-Mead starting from the best DE result.
        # Nelder-Mead is derivative-free and robust.
        # We wrap the function to stop if we exceed the total budget.
        
        # Reset a local counter for the local search to know how many are left
        # But we just use the global counted_func which handles the hard stop.
        
        try:
            # Nelder-Mead options
            # maxiter: limit iterations, but counted_func will hard stop at 500 total calls.
            # Since we used ~300 for DE, we have ~200 left.
            # Nelder-Mead uses (n+1) simplex points + reflections/expansions.
            # 200 calls is plenty for 5D NM to converge if close.
            
            res = minimize(
                counted_func, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass

        # If we have time and calls left, try a second local polish from a slightly perturbed best_x?
        # Or just return.
        
        # Ensure best_x is within bounds
        best_x = np.clip(best_x, -5.0, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
