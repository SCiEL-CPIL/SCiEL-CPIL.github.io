import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start with diverse initial points
        # Use stratified sampling for better coverage in 5D
        initial_points = []
        
        # 1. Center
        initial_points.append(np.zeros(dim))
        
        # 2. Stratified random sampling: divide each dimension into 3 bins
        # This ensures we don't cluster too much
        n_strat = 15
        for _ in range(n_strat):
            # Pick a random bin for each dimension
            bins = rng.integers(0, 3, dim)
            x = np.zeros(dim)
            for d in range(dim):
                # Uniform within the bin
                bin_start = -5.0 + bins[d] * (10.0 / 3.0)
                bin_end = bin_start + (10.0 / 3.0)
                x[d] = rng.uniform(bin_start, bin_end)
            initial_points.append(x)
            
        # 3. A few pure random points
        for _ in range(5):
            initial_points.append(rng.uniform(-5, 5, dim))

        # Evaluate all initial points
        evaluated = []
        for x0 in initial_points:
            if calls_made >= budget or remaining_time() < 10.0:
                break
            x0 = clip(x0)
            fx = evaluate(x0)
            evaluated.append((fx, x0.copy()))
        
        if not evaluated:
            # Safety
            x0 = np.zeros(dim)
            evaluate(x0)
            evaluated.append((best_f, best_x.copy()))

        # Sort by fitness
        evaluated.sort(key=lambda t: t[0])

        # Phase 2: Differential Evolution (DE/rand/1)
        # We will maintain a population of size N
        # Budget allocation for Phase 2
        # Reserve 20 calls for final polish
        reserve_budget = 20
        available_for_refinement = budget - calls_made - reserve_budget
        
        if available_for_refinement > 0:
            # Population size
            N = 50
            # F is the differential weight, CR is crossover rate
            F = 0.8
            CR = 0.9
            
            # Initialize population with the top candidates and random fill
            # We have 'evaluated' list of (fx, x). 
            # We need N points. If we have less, we generate random ones.
            # Actually, we can just use the evaluated points as the initial population if N <= len(evaluated)
            # Otherwise, mix them.
            
            pop_x = []
            pop_f = []
            
            # Use the best ones found so far
            for fx, x in evaluated:
                if len(pop_x) >= N:
                    break
                pop_x.append(x.copy())
                pop_f.append(fx)
                
            # Fill the rest with random points
            while len(pop_x) < N:
                x_new = rng.uniform(-5, 5, dim)
                fx_new = evaluate(x_new)
                if calls_made >= budget or remaining_time() < 5.0:
                    break
                pop_x.append(x_new)
                pop_f.append(fx_new)
            
            # If we didn't fill the population due to budget/time, adjust N
            N = len(pop_x)
            if N < 4: # Need at least 4 for DE/rand/1
                # Fallback to simple random search or (1+1) if population is too small
                # But given budget 500 and initial 21, N should be 50.
                pass
            
            # Main DE Loop
            while calls_made < budget - reserve_budget and remaining_time() > 5.0:
                # Find best index
                idx_best = np.argmin(pop_f)
                
                # For each individual, generate a trial vector
                for i in range(N):
                    if calls_made >= budget - reserve_budget or remaining_time() < 2.0:
                        break
                    
                    # Select 3 distinct random individuals different from i
                    # DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
                    # We need r1, r2, r3 distinct and != i
                    candidates = [j for j in range(N) if j != i]
                    if len(candidates) < 3:
                        continue
                    
                    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                    
                    v = pop_x[r1] + F * (pop_x[r2] - pop_x[r3])
                    
                    # Crossover
                    j_rand = rng.integers(0, dim)
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.uniform() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = pop_x[i][j]
                    
                    trial = clip(trial)
                    f_trial = evaluate(trial)
                    
                    # Selection
                    if f_trial <= pop_f[i]:
                        pop_x[i] = trial
                        pop_f[i] = f_trial

        # Phase 3: Final fine-tuning with very small steps on the global best
        # Use a simple (1+1)-ES or Nelder-Mead on the best found
        if calls_made < budget and remaining_time() > 5.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05 # Start small
            
            # Use remaining calls
            while calls_made < budget and remaining_time() > 1.0:
                y = clip(x + sigma * rng.standard_normal(dim))
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                else:
                    sigma *= 0.95
                    if sigma < 1e-6:
                        break

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
