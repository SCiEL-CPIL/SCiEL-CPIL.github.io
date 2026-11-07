"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid Differential Evolution and Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, rng):
        dim = self.dim
        # Initialize simplex
        simplex = [x0.copy() for _ in range(dim + 1)]
        for i in range(dim):
            simplex[i+1] = x0.copy()
            simplex[i+1][i] += 0.1 * (abs(x0[i]) + 1.0)
        
        fvals = np.array([func(np.clip(x, -5, 5)) for x in simplex])
        count = dim + 1
        best_idx = np.argmin(fvals)
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while count < budget:
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = fvals[order]
            best_idx = 0
            
            x_best = simplex[0]
            x_worst = simplex[-1]
            x_second = simplex[1]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            count += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                count += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (x_worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                count += 1
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        count += 1
                    if count >= budget:
                        break
            
            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break
        
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    def _differential_evolution(self, func, budget, rng):
        dim = self.dim
        pop_size = max(10, 2 * dim + 1)
        # Ensure we don't exceed budget in initialization
        if pop_size > budget:
            pop_size = budget
            
        # Initialize population
        population = [rng.uniform(-5, 5, dim) for _ in range(pop_size)]
        fvals = np.array([func(np.clip(x, -5, 5)) for x in population])
        count = pop_size
        
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        
        best_idx = np.argmin(fvals)
        best_x = population[best_idx].copy()
        best_f = fvals[best_idx]
        
        while count < budget:
            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = population[current_best_idx].copy()
                current_best_idx = np.argmin(fvals)

            # Generate new population
            new_population = population.copy()
            new_fvals = fvals.copy()
            
            for i in range(pop_size):
                if count + 1 > budget:
                    break
                    
                # Select three distinct individuals
                idxs = np.random.choice(pop_size, 3, replace=False)
                while len(set(idxs)) < 3:
                    idxs = np.random.choice(pop_size, 3, replace=False)
                a, b, c = idxs
                
                # Mutation
                v = population[a] + F * (population[b] - population[c])
                
                # Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                        
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                count += 1
                
                if f_trial <= fvals[i]:
                    new_population[i] = trial
                    new_fvals[i] = f_trial
            
            population = new_population
            fvals = new_fvals
            
            # Check for stagnation
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break
                
        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Split budget: 70% DE, 30% Nelder-Mead
        de_budget = int(budget * 0.7)
        nm_budget = budget - de_budget
        
        # Phase 1: Differential Evolution
        x_best, f_best = self._differential_evolution(func, de_budget, rng)
        
        # Phase 2: Nelder-Mead on the best solution
        if nm_budget > 0:
            x_final, f_final = self._nelder_mead(func, x_best, nm_budget, rng)
            if f_final < f_best:
                x_best = x_final
                f_best = f_final
        
        return x_best, f_best
# EVOLVE-BLOCK-END
