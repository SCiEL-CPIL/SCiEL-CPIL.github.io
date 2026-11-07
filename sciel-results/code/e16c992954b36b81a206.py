"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution with random scalarization for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        pop_size = 10
        generations = self.budget // pop_size  # 200 // 10 = 20
        F = 0.8
        CR = 0.9

        bounds_arr = np.array(bounds)
        lows = bounds_arr[:, 0]
        highs = bounds_arr[:, 1]

        pop = np.random.uniform(lows, highs, size=(pop_size, self.dim))
        pop_obj = np.array([problem(x) for x in pop])
        
        for gen in range(generations):
            # Pick a random weight for this generation
            w = np.random.rand()
            
            # Calculate scalarized objectives for current pop
            scalar_pop = pop_obj[:, 0] * w + pop_obj[:, 1] * (1 - w)
            
            for i in range(pop_size):
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, lows, highs)
                
                u = np.random.uniform(size=self.dim) < CR
                j_rand = np.random.randint(self.dim)
                u[j_rand] = True
                
                trial = np.where(u, v, pop[i])
                trial_obj = problem(trial)
                
                # Selection based on scalarized objective
                scalar_trial = trial_obj[0] * w + trial_obj[1] * (1 - w)
                
                if scalar_trial < scalar_pop[i]:
                    pop[i] = trial
                    pop_obj[i] = trial_obj
                    scalar_pop[i] = scalar_trial

        return None
# EVOLVE-BLOCK-END
