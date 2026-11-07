"""
Hybrid optimizer for BBOB f3 (Rastrigin).
Combines Differential Evolution (DE) for robust global exploration with
CMA-ES for high-precision local refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin

        n = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= budget

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Differential Evolution (Global Exploration)
        # DE is robust to multimodality and maintains diversity well.
        # Budget: ~40% of total
        n_de = max(20, int(budget * 0.4))
        
        # DE Parameters
        pop_size = 4 + int(3 * np.log(n))
        pop_size = min(pop_size, n_de // 2)
        if pop_size < 2: pop_size = 2
        
        # Initialize population
        population = [rng.uniform(-4.5, 4.5, n) for _ in range(pop_size)]
        # Ensure center is included
        population[0] = np.zeros(n)
        
        fitnesses = []
        for x in population:
            x = np.clip(x, -5, 5)
            f = func(x)
            update_best(x, f)
            fitnesses.append(f)
        
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        de_stagnation = 0
        last_best_de = best_f

        while calls_made + pop_size <= budget and not check_deadline():
            new_population = []
            new_fitnesses = []
            
            for i in range(pop_size):
                # Select distinct parents
                a_idx, b_idx, c_idx = rng.choice(pop_size, 3, replace=False)
                while a_idx == i:
                    a_idx = rng.choice(pop_size)
                while b_idx == i:
                    b_idx = rng.choice(pop_size)
                while c_idx == i:
                    c_idx = rng.choice(pop_size)
                
                a = population[a_idx]
                b = population[b_idx]
                c = population[c_idx]
                
                # Mutation: DE/rand/1
                v = a + F * (b - c)
                
                # Crossover: Binomial
                j_rand = rng.choice(n)
                u = rng.random(n)
                trial = np.where(u < CR, v, population[i])
                trial[j_rand] = v[j_rand] # Ensure at least one gene is mutated
                
                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                update_best(trial, f_trial)
                
                # Selection
                if f_trial <= fitnesses[i]:
                    new_population.append(trial)
                    new_fitnesses.append(f_trial)
                else:
                    new_population.append(population[i])
                    new_fitnesses.append(fitnesses[i])
            
            population = new_population
            fitnesses = new_fitnesses
            
            # Stagnation check
            if best_f < last_best_de - 1e-8:
                last_best_de = best_f
                de_stagnation = 0
            else:
                de_stagnation += 1
                
            if de_stagnation > 10:
                break

        if best_x is None:
            x = np.zeros(n)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: CMA-ES (Local Refinement)
        # Use the best point found by DE as the starting point for CMA-ES.
        # CMA-ES will adapt the search distribution to the local landscape.
        
        def run_cmaes(x_start, f_start, max_calls):
            nonlocal calls_made, best_x, best_f
            
            # CMA-ES Parameters
            lambda_pop = 4 + int(3 * np.log(n))
            if lambda_pop < 2: lambda_pop = 2
            if lambda_pop > max_calls: lambda_pop = max(2, max_calls // 2)
            
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            sigma = 0.5 # Start with moderate sigma
            mean = x_start.copy()
            
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_sqrt_inv = np.eye(n)
            
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = 2 / ((n + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (weights[1] - 1) / ((n + 2)**2 + mu_eff))
            
            ps = np.zeros(n)
            pc = np.zeros(n)
            
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            no_improve = 0
            last_best = f_start
            
            while not check_deadline() and calls_made < budget:
                # Check if we can afford a full generation
                if calls_made + lambda_pop > budget:
                    remaining = budget - calls_made
                    if remaining < mu:
                        break
                    lambda_pop = remaining
                    mu = max(1, lambda_pop // 2)
                    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                    weights /= np.sum(weights)
                    mu_eff = 1.0 / np.sum(weights**2)
                
                offspring = []
                for i in range(lambda_pop):
                    z = rng.standard_normal(n)
                    y = mean + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                    offspring.append((fy, y))
                    
                    if check_deadline():
                        break
                
                if len(offspring) < mu:
                    break
                    
                # Sort offspring by fitness
                offspring.sort(key=lambda x: x[0])
                
                # Check improvement
                if offspring[0][0] < last_best - 1e-10:
                    last_best = offspring[0][0]
                    no_improve = 0
                else:
                    no_improve += 1
                
                # Recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * offspring[i][1]
                    
                # Update mean
                old_mean = mean.copy()
                mean = y_w
                
                # Update covariance matrix
                ys = (y_w - old_mean) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n < (1.4 + 2 / (n + 1)) * cc)
                if hsig:
                    pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * ys
                else:
                    pc = (1 - cs) * pc
                
                rank_one = np.outer(ys, ys)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - old_mean) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                    
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * C) + cmu * rank_mu
                
                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2.0
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(n))
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                
                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Convergence criteria
                if sigma < 1e-12:
                    break
                if no_improve > 20 and sigma < 0.1:
                    break

            return best_x, best_f

        # Run CMA-ES
        run_cmaes(best_x, best_f, budget - calls_made)
        
        # Phase 3: Restart if budget remains and CMA-ES converged/stopped
        remaining = budget - calls_made
        if remaining > 50 and not check_deadline():
            # Perturb best_x to escape local minimum
            perturb_sigma = 1.0
            x_restart = np.clip(best_x + perturb_sigma * rng.standard_normal(n), -5, 5)
            f_restart = func(x_restart)
            calls_made += 1
            if f_restart < best_f:
                best_f = f_restart
                best_x = x_restart.copy()
            
            # Run CMA-ES again
            run_cmaes(best_x, best_f, budget - calls_made)

        return best_x, best_f

# EVOLVE-BLOCK-END
