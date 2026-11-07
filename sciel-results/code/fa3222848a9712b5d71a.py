import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Multi-start CMA-ES with adaptive restarts and L-BFGS-B polish for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325

        best_x = None
        best_f = np.inf
        calls = 0

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)

        # Shared archive of best solutions found
        archive = []  # List of (f, x) tuples, sorted by f

        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if x.shape != (dim,):
                x = x.reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
                # Add to archive (keep top 10)
                archive.append((val, x.copy()))
                archive.sort(key=lambda t: t[0])
                if len(archive) > 10:
                    archive.pop()
            return val

        def polish(x_start, maxfev=20):
            """Run L-BFGS-B polish from a starting point."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 30,
                        'maxfev': maxfev,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Initial sampling with structured points
        n_initial = min(20, budget // 15)
        
        # Structured initial points
        init_points = [
            np.zeros(dim),
            np.full(dim, -5.0),
            np.full(dim, 5.0),
        ]
        # Add random points
        for i in range(n_initial - len(init_points)):
            init_points.append(rng.uniform(-5, 5, dim))

        for x in init_points:
            if calls >= budget or time.time() > deadline:
                break
            wrapped_func(x)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Quick polish from initial best
        polish(best_x, maxfev=max(10, (budget - calls) // 10))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-scale Differential Evolution
        # Three populations: large (global), medium (basin), small (fine-tuning)
        
        def run_de(pop_size, max_generations, crossover_rate=0.7, mutation_scale=1.0):
            """Run differential evolution with given parameters."""
            nonlocal calls, best_f, best_x, archive
            
            if calls >= budget or time.time() > deadline:
                return
            
            # Initialize population from archive and random points
            pop = []
            for i in range(pop_size):
                if i < len(archive) and i % 3 == 0:
                    # Use archived solution
                    _, x_arch = archive[i % len(archive)]
                    x = x_arch + rng.normal(0, 0.1, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                x = np.clip(x, -5, 5)
                pop.append(x)
            
            # Evaluate initial population
            pop_f = []
            for x in pop:
                if calls >= budget or time.time() > deadline:
                    return
                f = wrapped_func(x)
                pop_f.append(f)
            
            if len(pop_f) < pop_size:
                return
            
            # DE parameters
            F = mutation_scale  # Mutation scale
            
            for gen in range(max_generations):
                if calls >= budget or time.time() > deadline:
                    break
                
                new_pop = []
                new_pop_f = []
                
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Select three distinct random individuals
                    while True:
                        a, b, c = rng.choice(pop_size, 3, replace=False)
                        if len({a, b, c, i}) == 4:
                            break
                    
                    # Mutation
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, -5, 5)
                    
                    # Crossover
                    crossover = rng.random(dim) < crossover_rate
                    if not crossover.any():
                        crossover[rng.randint(dim)] = True
                    
                    trial = np.where(crossover, mutant, pop[i])
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate trial
                    f_trial = wrapped_func(trial)
                    
                    # Selection
                    if f_trial <= pop_f[i]:
                        new_pop.append(trial)
                        new_pop_f.append(f_trial)
                    else:
                        new_pop.append(pop[i])
                        new_pop_f.append(pop_f[i])
                
                if len(new_pop_f) < pop_size:
                    break
                
                pop = new_pop
                pop_f = new_pop_f
                
                # Adaptive crossover rate
                improvement = sum(1 for i in range(pop_size) if new_pop_f[i] < pop_f[i])
                if improvement > pop_size * 0.3:
                    crossover_rate = min(0.9, crossover_rate * 1.1)
                elif improvement < pop_size * 0.1:
                    crossover_rate = max(0.5, crossover_rate * 0.9)
            
            # Update archive with best from this DE run
            best_idx = np.argmin(pop_f)
            if pop_f[best_idx] < best_f:
                best_f = pop_f[best_idx]
                best_x = pop[best_idx].copy()
                archive.append((pop_f[best_idx], pop[best_idx].copy()))
                archive.sort(key=lambda t: t[0])
                if len(archive) > 10:
                    archive.pop()

        # Run three DE populations in parallel (sequentially but with different scales)
        remaining = budget - calls
        
        # Large population for global exploration
        if remaining > 60:
            run_de(pop_size=24, max_generations=10, crossover_rate=0.8, mutation_scale=1.5)
        
        # Medium population for basin identification
        remaining = budget - calls
        if remaining > 40:
            run_de(pop_size=16, max_generations=8, crossover_rate=0.7, mutation_scale=1.0)
        
        # Small population for fine-tuning
        remaining = budget - calls
        if remaining > 20:
            run_de(pop_size=8, max_generations=10, crossover_rate=0.6, mutation_scale=0.5)

        # Phase 3: Multiple local search runs (10 vs 5 for better basin coverage)
        if best_x is not None and calls < budget and time.time() < deadline:
            n_local_searches = 10
            for ls in range(n_local_searches):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Choose starting point: mix of best found and random
                if ls == 0:
                    x_start = best_x.copy()
                elif ls < len(archive) // 2:
                    # Use archived solutions
                    _, x_arch = archive[ls % max(1, len(archive) // 2)]
                    x_start = x_arch + rng.normal(0, 0.2, dim)
                else:
                    x_start = best_x + rng.normal(0, 0.5, dim)
                
                x_start = np.clip(x_start, -5, 5)
                
                # Dynamic budget allocation for local search
                remaining = budget - calls
                maxfev = max(5, remaining // (n_local_searches - ls))
                
                polish(x_start, maxfev=maxfev)

        # Phase 4: Final high-precision polish
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            if remaining > 5:
                try:
                    minimize(
                        wrapped_func,
                        best_x.copy(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': 50,
                            'maxfev': remaining,
                            'ftol': 1e-14,
                            'gtol': 1e-10,
                            'disp': False
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f
