import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.func = func
        self.deadline = self.start_time + 280.0  # Safety margin
        self.best_x = None
        self.best_f = float('inf')
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            x = np.asarray(x)
            if x.shape != (self.dim,):
                x = np.reshape(x, (self.dim,))
            
            self.call_count += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = float('inf')
            
            if np.isfinite(f_val) and f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            
            return f_val

        dim = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size: 10 * dim is a common robust choice for 5D
        pop_size = max(10 * dim, 20) 
        # Ensure we have enough budget for at least a few generations
        # If budget is small, reduce pop_size
        if budget < pop_size * 5:
            pop_size = max(10, budget // 5)
            
        # JADE Parameters
        F = 0.5  # Initial mutation factor
        CR = 0.5 # Initial crossover rate
        c1 = 0.1
        c2 = 0.1
        
        # Initialize Population using Latin Hypercube Sampling for better coverage
        rng = np.random.default_rng(42)
        
        # LHS
        def lhs(n, dim):
            # n samples in dim dimensions
            # Stratified sampling
            points = np.zeros((n, dim))
            for i in range(dim):
                perm = rng.permutation(n)
                # (perm + U) / n gives uniform [0, 1) stratified
                col = (perm + rng.random(n)) / n
                points[:, i] = col
            # Scale to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        # Initial Population
        pop = lhs(pop_size, dim)
        
        # Evaluate initial population
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            if self.call_count >= budget or time.time() > self.deadline:
                break
            fitness[i] = wrapped_func(pop[i])
            
        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]
        
        # DE Main Loop
        # We reserve the last 10% of the budget for local polish
        polish_budget = max(5, budget // 10)
        de_budget = budget - polish_budget
        
        # JADE state
        F_current = F
        CR_current = CR
        
        gen = 0
        while self.call_count < de_budget and time.time() < self.deadline:
            gen += 1
            # Sort population by fitness
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]
            best_x = pop[0].copy()
            best_f = fitness[0]
            
            # Update JADE parameters
            # Calculate standard deviation of F and CR for successful trials in previous gen
            # For simplicity and speed, we use a simplified JADE update or fixed if too complex.
            # Standard JADE:
            # F_new = F + c1 * N(0.5, 0.2)
            # CR_new = CR + c2 * N(0.1, 0.1)
            # Clipped to [0, 1]
            # We will implement a lightweight version:
            # Keep F and CR fixed but slightly adaptive based on success rate if we tracked it.
            # Given the short budget, fixed F=0.5, CR=0.5 is often very robust.
            # Let's stick to fixed F=0.5, CR=0.5 for stability, or simple randomization.
            # Actually, let's use a simple random perturbation to avoid local optima in parameter space.
            F_current = np.clip(F + 0.1 * rng.standard_normal(), 0.1, 0.9)
            CR_current = np.clip(CR + 0.1 * rng.standard_normal(), 0.1, 0.9)

            new_pop = np.zeros_like(pop)
            new_fitness = np.zeros(pop_size)
            
            # Generate offspring
            for i in range(pop_size):
                if self.call_count >= de_budget or time.time() > self.deadline:
                    break
                    
                # Select 3 distinct random individuals r1, r2, r3 different from i
                # and distinct from each other
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                
                # DE/rand/1
                v = pop[r1] + F_current * (pop[r2] - pop[r3])
                
                # Binary Crossover
                j_rand = rng.integers(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() <= CR_current or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i, j]
                
                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)
                
                # Evaluate
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]
            
            # Replace population
            pop = new_pop
            fitness = new_fitness
            
            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_f:
                self.best_f = fitness[current_best_idx]
                self.best_x = pop[current_best_idx].copy()
        
        # Local Polish with Nelder-Mead
        if self.call_count < budget and time.time() < self.deadline:
            # Use remaining budget
            remaining = budget - self.call_count
            if remaining > 5:
                try:
                    res = minimize(
                        wrapped_func, 
                        self.best_x, 
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining // (dim + 1),
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if np.isfinite(res.fun) and res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = res.x
                except Exception:
                    pass
        
        # Final safety check
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            if self.call_count < budget:
                self.best_f = wrapped_func(self.best_x)
            else:
                self.best_f = 0.0

        return self.best_x, self.best_f
