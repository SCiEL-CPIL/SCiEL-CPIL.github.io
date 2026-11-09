"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, 10s buffer
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # DE Parameters
        # Population size: 5 * dim is standard, but for dim=5, 25 is reasonable.
        # With budget 500, we can afford roughly 20 generations.
        # Let's use a slightly smaller population to allow more generations if needed, 
        # or stick to standard. 10 * dim = 50? 
        # Standard DE usually uses NP = 5*D to 10*D.
        # Let's try NP = 20. 500 / 20 = 25 generations.
        npop = 20 
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        # Initialize population uniformly in [-5, 5]
        # For Sphere, starting spread out is good.
        pop = rng.uniform(-5, 5, (npop, dim))
        fit = np.full(npop, np.inf)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        # Helper to evaluate and update
        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            # Check budget and time
            if calls_used >= budget or time.time() > deadline:
                return np.inf
            
            # Clip to box (func does it too, but good practice to keep in bounds for DE logic)
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            calls_used += 1
            
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Initial evaluation
        for i in range(npop):
            fit[i] = evaluate(pop[i])
            if calls_used >= budget or time.time() > deadline:
                break
        
        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Main loop
        while calls_used < npop: # At least one more full gen possible? 
            # Actually, loop until budget is exhausted or time runs out
            if time.time() > deadline:
                break
                
            # Find current best index
            best_idx = np.argmin(fit)
            
            # Check if we can afford a full generation
            if calls_used + npop > budget:
                # Do partial generation or just break? 
                # Breaking is safer to avoid complex partial logic, 
                # but we might waste a few calls. 
                # Let's just break if we can't finish a gen, 
                # OR we can run until budget is hit inside the gen loop.
                # The inner loop below handles the budget check per individual.
                pass

            # Generate new population
            new_pop = np.empty_like(pop)
            new_fit = np.full(npop, np.inf)
            
            for i in range(npop):
                if calls_used >= budget or time.time() > deadline:
                    new_fit[i] = fit[i] # Keep old
                    new_pop[i] = pop[i]
                    continue
                
                # Select 3 distinct random indices different from i
                # For small npop, simple rejection sampling or permutation is fast.
                # npop=20, so distinct is easy.
                while True:
                    r1, r2, r3 = rng.integers(0, npop, 3)
                    if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                        break
                
                # Differential Evolution mutation: v = x[r1] + F * (x[r2] - x[r3])
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover: binomial
                # j_rand: random integer in [0, dim-1]
                j_rand = rng.integers(0, dim)
                # u: random numbers in [0, 1]
                u = rng.random(dim)
                
                # Create trial vector
                trial = np.empty_like(v)
                for j in range(dim):
                    if u[j] <= CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip trial to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate trial
                trial_fit = evaluate(trial)
                
                # Selection
                if trial_fit <= fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                else:
                    new_pop[i] = pop[i]
                    new_fit[i] = fit[i]
            
            # Update population
            pop = new_pop
            fit = new_fit
            
            # If we used up budget, the loop condition will handle exit next iteration
            # But we should check if we stopped mid-generation due to budget
            if calls_used >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
