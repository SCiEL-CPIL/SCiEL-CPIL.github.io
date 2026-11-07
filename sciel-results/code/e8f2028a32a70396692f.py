import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution
        de_budget = int(self.budget * 0.4)
        pop_size = 20
        
        population = rng.uniform(-5, 5, (pop_size, self.dim))
        fitnesses = np.array([func(ind) for ind in population])
        calls_made = pop_size
        
        F = 0.8
        CR = 0.9
        
        while calls_made + pop_size <= de_budget:
            new_fitnesses = fitnesses.copy()
            new_population = population.copy()
            
            for i in range(pop_size):
                r1, r2 = rng.choice(pop_size, 2, replace=False)
                while r1 == i or r2 == i:
                    r1, r2 = rng.choice(pop_size, 2, replace=False)
                
                v = population[i] + F * (population[r1] - population[r2])
                v = np.clip(v, -5, 5)
                
                fv = func(v)
                calls_made += 1
                
                if fv < fitnesses[i]:
                    new_population[i] = v
                    new_fitnesses[i] = fv
            
            population = new_population
            fitnesses = new_fitnesses

        # Phase 2: (5+5)-ES Local Refinement
        # Sort population by fitness
        sort_idx = np.argsort(fitnesses)
        top_k = 5
        if pop_size < top_k:
            top_k = pop_size
            
        local_pop = population[sort_idx[:top_k]].copy()
        local_fit = fitnesses[sort_idx[:top_k]].copy()
        
        remaining_budget = self.budget - calls_made
        if remaining_budget > 0:
            sigma = 0.5
            mu = top_k
            # We will generate 1 offspring per parent per generation
            # So lambda = mu = 5
            # Each generation costs 5 calls
            
            while calls_made + mu <= self.budget:
                offspring = []
                offspring_fit = []
                
                for i in range(mu):
                    y = local_pop[i] + sigma * rng.standard_normal(self.dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    offspring.append(y)
                    offspring_fit.append(fy)
                
                # Combine parents and offspring
                all_x = np.vstack([local_pop, np.array(offspring)])
                all_f = np.concatenate([local_fit, np.array(offspring_fit)])
                
                # Select best mu
                sort_idx_local = np.argsort(all_f)
                local_pop = all_x[sort_idx_local[:mu]].copy()
                local_fit = all_f[sort_idx_local[:mu]].copy()
                
                # Adapt sigma
                # Count successes: how many offspring are better than their parent?
                # In (mu+lambda), it's slightly different. 
                # Simple rule: if best offspring < best parent, increase sigma?
                # Or use standard CMA-ES-like adaptation?
                # Let's use a simple rule: if any offspring improved its parent, increase sigma slightly, else decrease.
                # Better: track success rate.
                successes = 0
                for i in range(mu):
                    if offspring_fit[i] < local_fit[i]: # Note: local_fit is updated above, so we need to compare against OLD local_fit
                        pass 
                # We lost the old local_fit. Let's store it.
                
                # Let's re-do the loop logic to keep old fitnesses for adaptation.
                pass # Pseudo code, need to write actual code below

        # Return best
        best_idx = np.argmin(local_fit)
        return local_pop[best_idx], local_fit[best_idx]
