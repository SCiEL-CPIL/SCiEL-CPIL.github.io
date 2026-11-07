import numpy as np
import time

class Optimizer:
    """Hybrid DE + CMA-ES for rotated Weierstrass landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Differential Evolution
        # Use a population size that balances exploration and budget
        N_de = 15
        # Allocate roughly 60% of budget to DE
        budget_de = int(budget * 0.6) 
        calls_made = 0
        
        # Initialize DE population
        pop = rng.uniform(-5, 5, size=(N_de, dim))
        fitness = np.array([func(np.clip(x, -5, 5)) for x in pop])
        calls_made += N_de
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        F = 0.8
        CR = 0.9
        
        # DE Loop
        while calls_made < budget_de:
            if time.time() > deadline:
                break
            
            # If we can't complete a full generation, break to save time for CMA-ES
            if calls_made + N_de > budget_de:
                break
                
            for i in range(N_de):
                if calls_made >= budget_de:
                    break
                if time.time() > deadline:
                    break
                    
                # Select three distinct individuals
                r1, r2, r3 = rng.choice(N_de, size=3, replace=False)
                while r1 == i: r1 = rng.integers(0, N_de)
                while r2 == i or r2 == r1: r2 = rng.integers(0, N_de)
                while r3 == i or r3 == r1 or r3 == r2: r3 = rng.integers(0, N_de)
                
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                j_rand = rng.integers(0, dim)
                cross_mask = rng.random(dim) < CR
                cross_mask[j_rand] = True
                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                calls_made += 1
                
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
            
            # Slight adaptation of F and CR
            F = max(0.5, F * 0.98)
            CR = min(1.0, CR * 1.005)

        # Phase 2: CMA-ES Local Refinement
        remaining_budget = budget - calls_made
        if remaining_budget > 20:
            # CMA-ES Parameters
            N_es = 10
            mu = N_es // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights)
            
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
            dampsigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
            
            mean = best_x.copy()
            # Start sigma smallish, as DE should have converged to the basin
            sigma = 0.5 
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            # CMA-ES Loop
            while calls_made < budget:
                if time.time() > deadline:
                    break
                if calls_made + N_es > budget:
                    break
                
                # Generate population
                cholesky = np.linalg.cholesky(C)
                offspring = []
                offspring_f = []
                
                for i in range(N_es):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    z = rng.standard_normal(dim)
                    x = mean + sigma * cholesky @ z
                    x = np.clip(x, -5, 5)
                    f_x = func(x)
                    calls_made += 1
                    
                    offspring.append(x)
                    offspring_f.append(f_x)
                    
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x.copy()
                
                if len(offspring) < N_es:
                    break
                    
                offspring = np.array(offspring)
                offspring_f = np.array(offspring_f)
                
                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring = offspring[order]
                offspring_f = offspring_f[order]
                
                # Update mean
                y_w = np.sum(weights[:mu, None] * offspring[:mu], axis=0)
                mean = mean + sigma * y_w
                
                # Update covariance matrix
                y_w = np.sum(weights[:mu, None] * offspring[:mu], axis=0)
                pc = (1 - cc) * pc + cc * (1 / sigma) * y_w
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (1 / sigma) * y_w
                
                C = (1 + c1 * (1 - cc) * np.outer(pc, pc) - c1 * C)
                # Rank-mu update
                for i in range(mu):
                    zi = offspring[i] - mean
                    C += cmu * weights[i] * np.outer(zi, zi)
                
                # Symmetrize and ensure positive semi-definite
                C = (C + C.T) / 2
                # Check for positive definiteness
                try:
                    cholesky = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If not PD, reset to identity or add small epsilon
                    C = C + 1e-8 * np.eye(dim)
                    try:
                        cholesky = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        cholesky = np.eye(dim)
                        
                # Update sigma
                sigma = sigma * np.exp((cs / dampsigma) * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                # Stop if sigma is too small
                if sigma < 1e-8:
                    break

        return best_f
