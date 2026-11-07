import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # 20s buffer before 300s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Differential Evolution for global search
        # DE is robust, simple, and avoids numerical instability of CMA-ES.
        # We use a wrapper that respects the budget and time limit.
        
        def de_func(x):
            return evaluate(x)

        # Bounds are [-5, 5] for each dimension
        bounds = [(-5, 5)] * self.dim
        
        # We allocate roughly 60% of the budget to DE.
        # DE population size is typically 15 * N. For N=5, that's 75.
        # Max iterations = (budget_allocated - initial_pop) / pop_size
        # Let's just let DE run until it hits the budget or time limit via the wrapper.
        # However, DE in scipy has its own iteration logic. 
        # It's safer to control the loop manually or set a high maxiter and rely on the wrapper.
        # But the wrapper raises RuntimeError when budget is hit, which scipy might catch or crash on.
        # Let's implement a simple DE loop manually to have full control over the budget.
        
        N = self.dim
        pop_size = max(15, 3 * N)  # 15 is standard, min 15
        F = 0.8
        CR = 0.9
        
        # Initialize population
        pop = np.random.uniform(-5, 5, (pop_size, N))
        pop_f = np.array([evaluate(x) for x in pop])
        
        # Sort initial population
        order = np.argsort(pop_f)
        pop = pop[order]
        pop_f = pop_f[order]
        
        # DE Loop
        gen = 0
        while calls_made < self.budget and time.time() - start_time < time_limit:
            gen += 1
            new_pop = np.zeros((pop_size, N))
            new_pop_f = np.zeros(pop_size)
            
            # First individual is the current best (elitism)
            new_pop[0] = pop[0]
            new_pop_f[0] = pop_f[0]
            
            for i in range(1, pop_size):
                # Select 3 distinct random individuals
                while True:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)
                    if a != b and b != c and c != a and a != i and b != i and c != i:
                        break
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                trial = np.random.randint(0, N)
                for j in range(N):
                    if np.random.rand() < CR or j == trial:
                        trial_x = v[j]
                    else:
                        trial_x = pop[i][j]
                    
                    # Clip to bounds
                    trial_x = np.clip(trial_x, -5, 5)
                    new_pop[i][j] = trial_x
                
                # Evaluate trial
                try:
                    f_trial = evaluate(new_pop[i])
                except (TimeoutError, RuntimeError):
                    # If we hit budget/time, stop immediately
                    break
                
                # Selection
                if f_trial <= pop_f[i]:
                    new_pop[i] = new_pop[i]
                    new_pop_f[i] = f_trial
                else:
                    new_pop[i] = pop[i]
                    new_pop_f[i] = pop_f[i]
                
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
            
            # Update population
            pop = new_pop
            pop_f = new_pop_f
            
            # Sort
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            # Check for convergence (optional)
            if pop_f[0] - pop_f[-1] < 1e-10:
                break

        # Phase 2: Local Polish with Nelder-Mead
        # Use remaining budget to refine the best solution found by DE.
        if best_x is not None and calls_made < self.budget - 10:
            try:
                from scipy.optimize import minimize
                
                def wrapper(x):
                    nonlocal calls_made, best_x, best_f
                    if calls_made >= self.budget or time.time() - start_time > time_limit:
                        raise RuntimeError("Stop")
                    calls_made += 1
                    f_val = func(x)
                    if f_val < best_f:
                        best_f = f_val
                        best_x = np.array(x)
                    return f_val

                # Initial simplex around best_x
                initial_simplex = [best_x]
                step_size = 0.05
                for i in range(self.dim):
                    x_i = best_x.copy()
                    x_i[i] += step_size
                    initial_simplex.append(x_i)
                
                # Nelder-Mead
                res = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': (self.budget - calls_made) // (self.dim + 1), 
                             'xatol': 1e-8, 
                             'fatol': 1e-8,
                             'initial_simplex': initial_simplex}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except Exception:
                pass

        return best_x, best_f
