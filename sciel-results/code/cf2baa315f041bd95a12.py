import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution

warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy:
        # 1. Parallel multi-scale DE: three populations (large, medium, small) with adaptive crossover
        # 2. Shared best-so-far archive for information sharing
        # 3. L-BFGS-B polish on top candidates
        # 4. Final refinement

        # Phase 1: Multi-scale Differential Evolution
        # We maintain three populations with different scales:
        # - Large (pop=20): global exploration, high crossover rate
        # - Medium (pop=12): basin identification, medium crossover
        # - Small (pop=6): fine-tuning, low crossover
        
        def run_de_population(pop_size, max_iter, crossover_rate, init_scale, name):
            nonlocal best_x, best_f, calls_made
            
            # Initialize population within init_scale of best_x (or random if no best yet)
            if best_x is not None:
                center = best_x.copy()
            else:
                center = np.zeros(d)
            
            population = []
            for i in range(pop_size):
                if i == 0 and best_x is not None:
                    x = best_x.copy()
                else:
                    x = center + rng.normal(0, init_scale, d)
                    x = np.clip(x, lb, ub)
                f = call_func(x)
                population.append((f, x))
                if calls_made >= self.budget or time.time() > deadline:
                    return population
            
            population.sort(key=lambda item: item[0])
            
            # Run DE iterations
            for it in range(max_iter):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                # Adaptive crossover rate: decrease over iterations
                cr = crossover_rate * (1.0 - 0.5 * it / max_iter)
                
                new_population = []
                for i in range(pop_size):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    # DE/rand/1 mutation
                    # Select three distinct random individuals
                    idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)
                    while idx1 == i or idx2 == i or idx3 == i:
                        idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)
                    
                    donor = population[idx1][1] + cr * (population[idx2][1] - population[idx3][1])
                    
                    # Crossover
                    trial = np.zeros(d)
                    j_rand = rng.integers(d)
                    for j in range(d):
                        if rng.random() < cr or j == j_rand:
                            trial[j] = donor[j]
                        else:
                            trial[j] = population[i][1][j]
                    trial = np.clip(trial, lb, ub)
                    
                    f_trial = call_func(trial)
                    
                    # Selection
                    if f_trial < population[i][0]:
                        new_population.append((f_trial, trial))
                    else:
                        new_population.append(population[i])
                
                if len(new_population) < pop_size:
                    break
                    
                population = new_population
                population.sort(key=lambda item: item[0])
                
                # Update global best
                if population[0][0] < best_f:
                    best_f = population[0][0]
                    best_x = population[0][1].copy()
            
            return population

        # Run the three populations in parallel (sequentially but with different scales)
        # Large population for global exploration
        if calls_made < self.budget and time.time() < deadline:
            run_de_population(pop_size=20, max_iter=5, crossover_rate=0.9, init_scale=3.0, name="large")
        
        # Medium population for basin identification
        if calls_made < self.budget and time.time() < deadline:
            run_de_population(pop_size=12, max_iter=8, crossover_rate=0.7, init_scale=1.5, name="medium")
        
        # Small population for fine-tuning
        if calls_made < self.budget and time.time() < deadline:
            run_de_population(pop_size=6, max_iter=10, crossover_rate=0.5, init_scale=0.5, name="small")
        
        # Phase 2: Local Refinement on best candidates
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made
            if remaining > 15:
                try:
                    minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': min(30, remaining),
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Final Polish with Nelder-Mead
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made
            if remaining > 10:
                try:
                    minimize(
                        call_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': min(20, remaining),
                            'xatol': 1e-12,
                            'fatol': 1e-12
                        }
                    )
                except Exception:
                    pass
        
        # Phase 4: If budget remains, do another round of DE from the best
        if calls_made < self.budget - 20 and time.time() < deadline and best_x is not None:
            run_de_population(pop_size=8, max_iter=5, crossover_rate=0.6, init_scale=0.8, name="restart")
            
            # Final polish
            if calls_made < self.budget and time.time() < deadline:
                remaining = self.budget - calls_made
                if remaining > 10:
                    try:
                        minimize(
                            call_func,
                            best_x,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxfev': min(20, remaining),
                                'ftol': 1e-15,
                                'gtol': 1e-12
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
