import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        
        n = self.dim
        budget = self.budget
        NP = 10
        F = 0.8
        CR = 1.0
        
        # Initialize population
        pop = rng.uniform(-5, 5, (NP, n))
        f_pop = np.array([func(x) for x in pop])
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]
        
        evals = NP
        gen = 0
        no_improve_gen = 0
        indices = np.arange(NP)
        
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Check stagnation and restart if necessary
            if no_improve_gen > 30:
                if evals + NP <= budget:
                    # Restart population around best_x
                    pop = best_x + rng.normal(0, 2.0, (NP, n))
                    pop = np.clip(pop, -5, 5)
                    f_pop = np.array([func(x) for x in pop])
                    evals += NP
                    best_idx = np.argmin(f_pop)
                    if f_pop[best_idx] < best_f:
                        best_f = f_pop[best_idx]
                        best_x = pop[best_idx].copy()
                    no_improve_gen = 0
                    gen = 0
                else:
                    break
                continue

            improved = False
            for i in range(NP):
                if evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                mask = np.ones(NP, dtype=bool)
                mask[i] = False
                r1, r2, r3 = rng.choice(indices[mask], 3, replace=False)
                
                # Donor vector
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)
                
                # Evaluate
                fv = func(v)
                evals += 1
                
                # Selection
                if fv <= f_pop[i]:
                    pop[i] = v
                    f_pop[i] = fv
                    if fv < best_f:
                        best_f = fv
                        best_x = v.copy()
                        improved = True
            
            if not improved:
                no_improve_gen += 1
            else:
                no_improve_gen = 0
            gen += 1

        return best_x, best_f
