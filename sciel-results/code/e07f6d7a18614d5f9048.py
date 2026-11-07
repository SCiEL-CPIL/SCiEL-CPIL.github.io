import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin from 330s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initialize population with diverse points
        pop_size = 30  # Population size for DE
        points = []
        values = []
        
        # Center point
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        points.append(x_center)
        values.append(f_center)
        
        # Generate initial population
        for _ in range(pop_size - 1):
            x_rand = rng.uniform(-5, 5, self.dim)
            f_rand = func(x_rand)
            points.append(x_rand)
            values.append(f_rand)
        
        calls_used = len(points)
        best_f = min(values)
        best_x = points[np.argmin(values)]
        
        # Phase 2: Differential Evolution
        # DE parameters
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        max_generations = (self.budget - calls_used) // pop_size
        
        for gen in range(max_generations):
            if time.time() > deadline - 5:
                break
            if calls_used >= self.budget - 10:  # Reserve calls for final refinement
                break
            
            # Sort population by fitness
            sorted_indices = np.argsort(values)
            sorted_points = [points[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            # Update best
            if sorted_values[0] < best_f:
                best_f = sorted_values[0]
                best_x = sorted_points[0]
            
            # Generate offspring for each individual
            new_points = []
            new_values = []
            
            for i in range(pop_size):
                if calls_used >= self.budget - 10:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Select three distinct random indices different from i
                while True:
                    r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
                    if len({r1, r2, r3, i}) == 4:
                        break
                
                # DE/rand/1 mutation
                mutant = points[r1] + F * (points[r2] - points[r3])
                
                # Binary crossover
                trial = np.copy(points[i])
                j_rand = rng.integers(self.dim)
                for j in range(self.dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_used += 1
                
                # Selection: keep better of parent and trial
                if f_trial < values[i]:
                    new_points.append(trial)
                    new_values.append(f_trial)
                else:
                    new_points.append(points[i])
                    new_values.append(values[i])
            
            if new_points:
                points = new_points
                values = new_values
        
        # Phase 3: Local refinement from best point
        remaining_budget = self.budget - calls_used
        if remaining_budget > 20 and time.time() < deadline - 10:
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxfev': remaining_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        return best_x, best_f
