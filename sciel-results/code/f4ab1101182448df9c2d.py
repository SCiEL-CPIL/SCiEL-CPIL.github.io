import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin for 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget
        
        # Differential Evolution with population-based search
        # Structurally different: uses difference vectors from multiple parents
        rng = np.random.default_rng(123)
        
        # Phase 1: Initialize population with Latin Hypercube Sampling
        pop_size = min(20, max(10, budget // 25))
        n_init = pop_size
        
        population = rng.uniform(-5, 5, size=(n_init, dim))
        fitness = np.full(n_init, float('inf'))
        
        for i in range(n_init):
            if eval_count >= budget:
                break
            x = np.clip(population[i], -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            fitness[i] = fx
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort population by fitness
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        
        # Phase 2: Differential Evolution loop
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        stagnation_count = 0
        best_f_prev = best_f
        
        while eval_count < budget and time.time() < deadline - 10:
            # Check for stagnation
            if best_f == best_f_prev:
                stagnation_count += 1
            else:
                stagnation_count = 0
                best_f_prev = best_f
            
            # Adaptive F and CR based on stagnation
            F_adapt = F * (1.0 + 0.1 * min(stagnation_count, 5))
            F_adapt = min(F_adapt, 1.5)
            
            # For each individual, create trial vector
            for i in range(n_init):
                if eval_count >= budget:
                    break
                if time.time() > deadline - 10:
                    break
                
                # Select three distinct random individuals
                r1, r2, r3 = rng.choice(n_init, size=3, replace=False)
                
                # DE/rand/1 mutation
                v = population[r1] + F_adapt * (population[r2] - population[r3])
                
                # Binary crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = population[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate trial
                ft = func(trial)
                eval_count += 1
                
                # Selection: keep better of parent and trial
                if ft <= fitness[i]:
                    population[i] = trial
                    fitness[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        stagnation_count = 0
                        best_f_prev = best_f
            
            # Re-sort population
            sorted_idx = np.argsort(fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]
            
            # If stuck for many generations, inject diversity
            if stagnation_count > 10:
                for i in range(n_init):
                    if rng.random() < 0.3:
                        population[i] = rng.uniform(-5, 5, dim)
                        if eval_count < budget:
                            ft = func(population[i])
                            eval_count += 1
                            fitness[i] = ft
                            if ft < best_f:
                                best_f = ft
                                best_x = population[i].copy()
                sorted_idx = np.argsort(fitness)
                population = population[sorted_idx]
                fitness = fitness[sorted_idx]
                stagnation_count = 0
        
        # Phase 3: Local polish with (1+1)-ES from best point
        remaining = budget - eval_count
        if remaining > 0 and time.time() < deadline - 5:
            rng_local = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            no_improve = 0
            limit = 15
            
            while remaining > 0 and time.time() < deadline - 5:
                if no_improve > limit:
                    sigma = max(0.01, sigma * 2.0)
                    no_improve = 0
                
                y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining -= 1
                eval_count += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                    if sigma < 1e-6:
                        break

        return best_x, best_f


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
