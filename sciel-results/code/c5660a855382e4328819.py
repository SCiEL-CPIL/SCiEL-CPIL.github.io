# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for RE37 (4 variables, 3 objectives, bounds [0,1]).
    Strategy:
      1. Phase 1: Differential Evolution (DE) for global exploration.
         - Uses Latin Hypercube Sampling for initialization.
         - DE/rand/1/bin strategy.
         - Pareto-based selection to maintain diversity and convergence.
      2. Phase 2: Local Polish using Nelder-Mead on diverse front points.
         - Refines the best solutions found by DE.
      3. Deadline guard to stay within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a fixed seed for the internal RNG to ensure reproducibility of the search strategy
        # The problem function itself is deterministic given x, so this is safe.
        rng = np.random.RandomState(42)

        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            if time.time() > deadline:
                return None
            fx = np.array(problem(x), dtype=float)
            eval_count += 1
            return fx

        def is_dominated(f, other_f):
            """Returns True if f is dominated by other_f."""
            return np.all(other_f <= f) and np.any(other_f < f)

        def is_non_dominated(f, population_f):
            """Returns True if f is not dominated by any point in population_f."""
            for pf in population_f:
                if is_dominated(f, pf):
                    return False
            return True

        # --- Phase 1: Differential Evolution ---
        
        # 1. Initialization with Latin Hypercube Sampling
        pop_size = 20
        n_init = min(pop_size, budget)
        
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            lhs_points[:, d] = (perm + rng.random(n_init)) / n_init
        lhs_points = lo + lhs_points * (hi - lo)
        
        population_x = lhs_points.copy()
        population_f = []
        
        for i in range(n_init):
            f = evaluate(population_x[i])
            if f is None:
                break
            population_f.append(f)
        
        population_f = np.array(population_f)
        
        # DE Parameters
        CR = 0.9
        F = 0.8
        max_generations = (budget - n_init) // pop_size
        
        for gen in range(max_generations):
            if time.time() > deadline or eval_count >= budget:
                break
            
            for i in range(pop_size):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Select 3 distinct random indices different from i
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Create trial vector using DE/rand/1/bin
                # v = x_r1 + F * (x_r2 - x_r3)
                v = population_x[r1] + F * (population_x[r2] - population_x[r3])
                
                # Apply binomial crossover
                j_rand = rng.randint(dim)
                trial_x = population_x[i].copy()
                for j in range(dim):
                    if rng.random() <= CR or j == j_rand:
                        trial_x[j] = v[j]
                
                # Clip to bounds
                trial_x = np.clip(trial_x, lo, hi)
                
                f_trial = evaluate(trial_x)
                if f_trial is None:
                    break
                
                # Selection: 
                # If trial dominates current, replace.
                # If current dominates trial, keep current.
                # If non-dominated, keep current (to avoid losing diversity too fast, 
                # or we could replace if trial is closer to ideal, but simple dominance is safer).
                # To improve convergence, we can also replace if trial is non-dominated 
                # and current is dominated by some other point? 
                # Standard DE uses single-objective comparison. For MO, we use Pareto.
                
                if is_dominated(population_f[i], f_trial):
                    # Trial is better than current
                    population_x[i] = trial_x
                    population_f[i] = f_trial
                elif not is_dominated(f_trial, population_f[i]):
                    # They are non-dominated. 
                    # Strategy: Keep current to maintain stability, 
                    # OR replace with probability 0.5 to explore.
                    # Let's stick to keeping current unless strictly dominated to ensure we don't lose good points.
                    # However, this might stagnate. 
                    # Better: If non-dominated, we can check if trial improves hypervolume? 
                    # Too expensive. 
                    # Let's use a simple rule: Replace if trial is non-dominated and 
                    # the current point is "worse" in terms of distance to the ideal point (0,0,0)?
                    # No, that biases. 
                    # Let's just keep current for simplicity and rely on the large number of generations.
                    pass
                # If current dominates trial, do nothing.

        # --- Phase 2: Local Polish ---
        
        # Extract current Pareto Front
        front_indices = []
        for i in range(len(population_f)):
            if is_non_dominated(population_f[i], population_f):
                front_indices.append(i)
        
        if not front_indices:
            # Fallback if something went wrong
            return lo

        # Select diverse points from the front for polishing
        # We want to polish points that are likely to be on the "knee" or extremes.
        # Simple selection: Pick the point with min obj0, min obj1, min obj2, and the one with min sum.
        candidates_to_polish = []
        
        if len(front_indices) > 0:
            # Min each objective
            for obj_idx in range(n_obj):
                min_val = float('inf')
                min_idx = -1
                for idx in front_indices:
                    if population_f[idx][obj_idx] < min_val:
                        min_val = population_f[idx][obj_idx]
                        min_idx = idx
                if min_idx != -1 and min_idx not in candidates_to_polish:
                    candidates_to_polish.append(min_idx)
            
            # Min sum (balanced)
            min_sum = float('inf')
            min_sum_idx = -1
            for idx in front_indices:
                s = np.sum(population_f[idx])
                if s < min_sum:
                    min_sum = s
                    min_sum_idx = idx
            if min_sum_idx != -1 and min_sum_idx not in candidates_to_polish:
                candidates_to_polish.append(min_sum_idx)

            # If we have very few front points, just take all
            if len(candidates_to_polish) < 3 and len(front_indices) > len(candidates_to_polish):
                for idx in front_indices:
                    if idx not in candidates_to_polish:
                        candidates_to_polish.append(idx)
                    if len(candidates_to_polish) >= 3:
                        break

        # Polish up to 3-4 points with Nelder-Mead
        # Budget remaining
        remaining_budget = budget - eval_count
        if remaining_budget > 0:
            # Allocate roughly equal budget to each polish run
            n_polish = min(4, len(candidates_to_polish))
            if n_polish > 0:
                evals_per_polish = max(5, remaining_budget // n_polish)
                
                for idx in candidates_to_polish[:n_polish]:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    
                    x0 = population_x[idx].copy()
                    
                    # Define a scalar objective that tries to minimize the distance to the ideal point 
                    # or a weighted sum. Since we don't know the ideal, minimizing the sum is a good heuristic 
                    # for the "balanced" region, but for extremes, we should minimize the specific objective?
                    # Nelder-Mead minimizes a single function.
                    # To improve the Pareto front, we want to move the point along the front or towards the ideal.
                    # A common trick is to minimize the Tchebycheff distance or just the sum.
                    # Let's minimize the sum of objectives. This tends to pull points towards the "knee".
                    # To cover extremes, we already picked min-obj points. Polishing them with sum-minimization
                    # might move them away from the extreme. 
                    # Better: For the point that is min-obj-k, minimize obj-k.
                    # For the balanced point, minimize sum.
                    
                    # Let's assign a specific weight vector based on the point's characteristics
                    # If this point was selected as min-obj-k, use weight [0,0,0,1] for obj k.
                    # If balanced, use [1,1,1].
                    
                    # Simpler: Just use a weighted sum where weights are inverse of the current objective values?
                    # Or just minimize the sum. It's a safe bet for hypervolume improvement in the center.
                    # Let's use a weighted sum where the weight for the objective this point is best at is higher.
                    
                    f_current = population_f[idx]
                    # Normalize weights to sum to 1
                    # If f is [0.1, 0.5, 0.5], weights should emphasize obj 0? 
                    # Actually, to improve HV, we want to push the point "inwards" (lower values).
                    # Minimizing sum is generally good.
                    
                    def objective(x):
                        nonlocal eval_count
                        if time.time() > deadline:
                            return 1e10
                        if eval_count >= budget:
                            return 1e10
                        fx = np.array(problem(x), dtype=float)
                        eval_count += 1
                        return np.sum(fx)
                    
                    try:
                        res = minimize(objective, x0, method='Nelder-Mead',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 20, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                        
                        if res.success or res.fun < 1e9:
                            x_best = np.clip(res.x, lo, hi)
                            # Re-evaluate to get exact f and ensure it's recorded
                            f_best = evaluate(x_best)
                            if f_best is not None:
                                # We don't strictly need to update the population array for scoring, 
                                # as the evaluator records every call. 
                                # But updating it helps if we do further steps (we don't).
                                pass
                    except Exception:
                        pass

        # Return any valid point (ignored by evaluator, but must return something)
        if population_x is not None and len(population_x) > 0:
            return population_x[0]
        return lo

# EVOLVE-BLOCK-END
