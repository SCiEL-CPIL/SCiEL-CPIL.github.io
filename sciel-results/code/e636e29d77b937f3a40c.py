# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """
    Multi-objective optimizer for RE37 using a combination of:
    1. Uniform random initialization
    2. NSGA-II style evolutionary search with crowding distance
    3. Local refinement via scipy's differential_evolution on aggregated objective
    4. Gradient-free local search (Nelder-Mead) on promising points
    
    Strategy:
    - Use a significant portion of budget for global exploration (random + DE)
    - Use remaining budget for local refinement of best points
    - Track all evaluated points to compute hypervolume contribution
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Track all evaluations
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f
        
        # Phase 1: Random initialization - use ~30% of budget
        n_init = max(10, self.budget // 3)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)
        
        # Phase 2: Differential Evolution for global exploration - use ~40% of budget
        # DE minimizes a scalar aggregate of objectives
        def aggregate_objective(x):
            f = problem(np.clip(x, lo, hi))
            # Sum of objectives (simple aggregate)
            return np.sum(f)
        
        # Use DE with limited iterations to stay within budget
        # DE internally uses population_size * generations evaluations
        # We need to be careful about budget
        remaining = self.budget - used
        
        if remaining > 20:
            # Try DE with small population and few generations
            # DE population size is 2*dim by default, we can't easily control iterations
            # Instead, let's do a manual DE-like search
            
            # Manual DE: population-based with mutation
            pop_size = min(20, max(8, remaining // 10))
            n_de_iter = max(1, (remaining - pop_size) // pop_size)
            
            # Initialize DE population from existing points + random
            de_pop = [np.array(all_X[i]) for i in range(min(pop_size, len(all_X)))]
            while len(de_pop) < pop_size:
                de_pop.append(lo + (hi - lo) * rng.random(self.dim))
            
            de_F = [evaluate(x) for x in de_pop]
            
            for gen in range(n_de_iter):
                if used >= self.budget:
                    break
                new_pop = []
                for i in range(pop_size):
                    if used >= self.budget:
                        break
                    # Select 3 distinct random indices
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    a, b, c = rng.choice(idxs, size=3, replace=False)
                    
                    # DE mutation
                    F_cr = 0.8  # Differential weight
                    CR = 0.5    # Crossover rate
                    mutant = de_pop[a] + F_cr * (de_pop[b] - de_pop[c])
                    mutant = np.clip(mutant, lo, hi)
                    
                    # Crossover
                    trial = de_pop[i].copy()
                    for j in range(self.dim):
                        if rng.random() < CR:
                            trial[j] = mutant[j]
                    
                    f_trial = evaluate(trial)
                    
                    # Selection: keep better of trial and current
                    # Use dominance: if trial dominates current, replace
                    # Else if current dominates trial, keep current
                    # Else, keep current (simple)
                    if np.all(f_trial <= de_F[i]) and np.any(f_trial < de_F[i]):
                        de_pop[i] = trial
                        de_F[i] = f_trial
                    elif not (np.all(de_F[i] <= f_trial) and np.any(de_F[i] < f_trial)):
                        # Non-dominated or incomparable: keep trial if it's better in sum
                        if np.sum(f_trial) < np.sum(de_F[i]):
                            de_pop[i] = trial
                            de_F[i] = f_trial
                # End for i
            # End DE loop
        
        # Phase 3: Local refinement of best points - use remaining budget
        remaining = self.budget - used
        
        if remaining > 5 and len(all_X) > 0:
            # Find non-dominated points
            n = len(all_F)
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Identify non-dominated points
            non_dom_idx = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dom_idx.append(i)
            
            # If no non-dominated found, pick best by sum
            if not non_dom_idx:
                non_dom_idx = [int(np.argmin(np.sum(F_arr, axis=1)))]
            
            # Refine top non-dominated points
            # Sort by sum of objectives
            non_dom_idx.sort(key=lambda i: np.sum(F_arr[i]))
            n_refine = min(len(non_dom_idx), max(1, remaining // 5))
            
            for idx in non_dom_idx[:n_refine]:
                if used >= self.budget:
                    break
                x0 = all_X[idx].copy()
                
                # Use Nelder-Mead on aggregated objective
                def obj_func(x):
                    if used >= self.budget:
                        return 1e10
                    x_clipped = np.clip(x, lo, hi)
                    f = problem(x_clipped)
                    # Penalize if out of bounds (shouldn't happen due to clip)
                    return np.sum(f)
                
                try:
                    result = minimize(
                        obj_func, x0, method='Nelder-Mead',
                        options={'maxiter': max(1, (self.budget - used) // 3),
                                 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if result.success or result.fun < np.sum(all_F[idx]):
                        x_best = np.clip(result.x, lo, hi)
                        f_best = evaluate(x_best)
                except Exception:
                    pass
        # End Phase 3
        
        # Return last X (ignored by evaluator, but keep interface)
        if all_X:
            return all_X[-1]
        else:
            return lo.copy()
# EVOLVE-BLOCK-END
