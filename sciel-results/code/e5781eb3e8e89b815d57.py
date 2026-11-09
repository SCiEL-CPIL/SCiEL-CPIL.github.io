"""Optimizer using differential evolution with adaptive parameters for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution optimizer with adaptive strategy selection."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin from 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Smaller population to allow more generations and local search
        pop_size = 20 
        
        # Initialize population uniformly in [-5, 5]^n
        pop = rng.uniform(-5, 5, (pop_size, n))
        fitness = np.array([func(x) for x in pop])
        eval_count = pop_size

        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        # DE parameters
        F = 0.8
        CR = 0.9
        
        # Local search parameters
        local_search_freq = 5 # Every 5 generations
        local_search_budget = 20 # Max calls for local search per trigger
        
        generation = 0
        no_improve = 0

        def nelder_mead(x0, max_iter):
            """Simple Nelder-Mead implementation for local polishing."""
            # Initialize simplex
            simplex = [x0.copy() for _ in range(n + 1)]
            for i in range(n):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += 0.1 * (5 - (-5)) # Small step relative to bounds
            
            # Evaluate initial simplex
            fvals = np.array([func(x) for x in simplex])
            local_evals = n + 1
            
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                if local_evals >= local_search_budget:
                    break
                    
                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                local_evals += 1
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    local_evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    local_evals += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            local_evals += 1
                        if local_evals >= local_search_budget:
                            break

            # Return best from simplex
            best_local_idx = np.argmin(fvals)
            return simplex[best_local_idx], fvals[best_local_idx], local_evals

        while eval_count < budget:
            # Check time
            if time.time() > deadline - 2:
                break

            # Sort population by fitness
            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]

            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            generation += 1
            
            # Trigger local search if stagnation or periodic
            if (no_improve > 10 or generation % local_search_freq == 0) and eval_count + local_search_budget < budget:
                if time.time() < deadline - 10:
                    # Run local search on the current best
                    local_x, local_f, used_evals = nelder_mead(best_x.copy(), local_search_budget)
                    eval_count += used_evals
                    
                    if local_f < best_f:
                        best_f = local_f
                        best_x = local_x
                        # Update population best with local result
                        pop[0] = local_x
                        fitness[0] = local_f
                    no_improve = 0

            if eval_count >= budget:
                break

            # Generate offspring for each individual (DE step)
            for i in range(pop_size):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 2:
                    break

                # DE/rand/1/bin
                while True:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5, 5)

                # Binary crossover
                jrand = rng.integers(0, n)
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]

                # Evaluate
                trial_f = func(trial)
                eval_count += 1

                # Selection
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f

        return best_x, best_f
# EVOLVE-BLOCK-END
