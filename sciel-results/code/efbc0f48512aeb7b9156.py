"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Adaptive Differential Evolution (JADE) followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Allocate budget: 85% to DE, 15% to Nelder-Mead
        de_budget = int(budget * 0.85)
        nm_budget = budget - de_budget
        
        # --- Phase 1: Adaptive Differential Evolution (JADE) ---
        # Population size
        NP = 2 * dim  # Standard recommendation for DE
        if NP < 10:
            NP = 10
            
        rng = np.random.default_rng(42)
        
        # Initialize population uniformly in [-5, 5]
        # To avoid bias, we can use a wider range or just the box
        population = rng.uniform(-5, 5, (NP, dim))
        fvals = np.array([func(np.clip(population[i], -5, 5)) for i in range(NP)])
        calls = NP
        
        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]
        
        # JADE Parameters
        F = 0.5  # Initial scaling factor
        CR = 0.5  # Initial crossover rate
        
        # History for adaptation
        F_hist = []
        CR_hist = []
        
        while calls < de_budget:
            # Check if we can afford a full generation
            # Each generation costs NP evaluations
            if calls + NP > de_budget:
                # Do a partial generation or stop? 
                # For simplicity, if we can't do a full generation, we break.
                # But we can do partial updates if we want.
                # Let's just break to keep it simple and robust.
                break
                
            new_population = population.copy()
            new_fvals = fvals.copy()
            gen_F_hist = []
            gen_CR_hist = []
            
            for i in range(NP):
                # Select r1, r2, r3 distinct from i
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # JADE Mutation: x_r1 + F * (x_r2 - x_r3)
                # Or use the best individual? JADE uses x_r1, x_r2, x_r3 from current pop.
                # Sometimes using best helps convergence. Let's stick to standard DE/rand/1 for robustness.
                mutant = population[r1] + F * (population[r2] - population[r3])
                
                # JADE Crossover
                # Determine crossover rate for this individual
                # In JADE, CR is also adapted. Here we use a fixed CR per generation for simplicity 
                # or adapt it. Let's use a fixed CR=0.5 for now, or random per individual.
                # Standard DE uses random CR per individual or fixed.
                # Let's use a random CR per individual drawn from a Beta distribution?
                # For simplicity, let's use a fixed CR=0.5 for the trial vector creation, 
                # but we can adapt it globally.
                
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i][j]
                        
                trial_clipped = np.clip(trial, -5, 5)
                f_trial = func(trial_clipped)
                calls += 1
                
                # Selection
                if f_trial <= fvals[i]:
                    new_population[i] = trial_clipped
                    new_fvals[i] = f_trial
                    # Record successful parameters
                    gen_F_hist.append(F)
                    gen_CR_hist.append(CR)
                    
                    # Update global best
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial_clipped.copy()
            
            population = new_population
            fvals = new_fvals
            
            # Adapt F and CR using JADE rules
            if gen_F_hist:
                # Mean of successful F and CR
                mean_F = np.mean(gen_F_hist)
                mean_CR = np.mean(gen_CR_hist)
                
                # JADE adaptation:
                # F_{gen+1} = 0.9 * F_gen + 0.1 * mean(F_success)
                # CR_{gen+1} = 0.9 * CR_gen + 0.1 * mean(CR_success)
                # This is a simplified version. JADE actually uses Cauchy/Beta distributions.
                # Let's use a simple weighted average.
                F = 0.9 * F + 0.1 * mean_F
                CR = 0.9 * CR + 0.1 * mean_CR
                
                # Clamp parameters
                F = np.clip(F, 0.1, 0.9)
                CR = np.clip(CR, 0.1, 0.9)
            else:
                # If no success, keep current or reset?
                # Keep current to avoid drastic changes.
                pass
        
        # --- Phase 2: Nelder-Mead ---
        # Use remaining budget for local refinement
        if nm_budget > 0 and calls < budget:
            # Create a wrapper to count calls and stop
            nm_calls = [0]
            def nm_func(x):
                if nm_calls[0] >= nm_budget:
                    return float('inf')
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                nm_calls[0] += 1
                return val
            
            # Start from best found by DE
            x0 = best_x
            res = minimize(nm_func, x0, method='Nelder-Mead', 
                           options={'maxiter': nm_budget, 'xatol': 1e-8, 'fatol': 1e-12})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
