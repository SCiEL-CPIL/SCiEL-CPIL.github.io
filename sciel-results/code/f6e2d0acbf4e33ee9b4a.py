"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using adaptive DE search with local polish for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        
        # Latin Hypercube Sampling for initial population
        def lhs(n, d):
            """Generate Latin Hypercube samples in [0,1]^d."""
            samples = np.zeros((n, d))
            for dd in range(d):
                perm = np.arange(n)
                np.random.shuffle(perm)
                samples[:, dd] = (perm + np.random.random(n)) / n
            return samples
        
        # Smaller initial population to free budget for more generations
        pop_size = min(24, max(12, budget // 8))
        
        X = lo + (hi - lo) * lhs(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Check time after initial evaluation
        if time.time() > deadline:
            return X
        
        # DE-style evolutionary loop with adaptive crossover
        base_cr = 0.7  # Crossover rate
        base_f = 0.8   # Scaling factor
        decay_rate = 0.97  # Scaling factor decay per generation
        cr_decay = 0.995  # Crossover rate decay
        
        f_scale = base_f
        cr = base_cr
        gen = 0
        
        while used < budget and time.time() < deadline:
            gen += 1
            
            # Find non-dominated points
            non_dom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom_mask[i] = False
                        break
            
            # Select parents from non-dominated set
            if np.any(non_dom_mask):
                parents_idx = np.where(non_dom_mask)[0]
            else:
                parents_idx = np.arange(len(F))
            
            # Generate children with DE/rand/1 mutation
            n_children = min(budget - used, max(1, pop_size // 2))
            
            for _ in range(n_children):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                    
                # Select 3 distinct parents for DE/rand/1
                n_parents = len(parents_idx)
                if n_parents >= 3:
                    p1_idx, p2_idx, p3_idx = np.random.choice(n_parents, size=3, replace=False)
                    p1 = X[parents_idx[p1_idx]]
                    p2 = X[parents_idx[p2_idx]]
                    p3 = X[parents_idx[p3_idx]]
                    # DE/rand/1: v = x1 + F * (x2 - x3)
                    v = p1 + f_scale * (p2 - p3)
                else:
                    parent_idx = parents_idx[np.random.randint(n_parents)]
                    parent = X[parent_idx]
                    v = parent + np.random.randn(dim) * 0.1 * (hi - lo)
                
                # Binomial crossover
                trial = parent.copy() if n_parents >= 1 else v
                if n_parents >= 1:
                    parent_idx = parents_idx[np.random.randint(n_parents)]
                    parent = X[parent_idx]
                    trial = parent.copy()
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < cr or j == j_rand:
                            trial[j] = v[j]
                
                trial = np.clip(trial, lo, hi)
                
                fc = problem(trial)
                used += 1
                
                # Replace worst point (by sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = trial
                F[worst_idx] = fc
            
            # Decay parameters
            f_scale *= decay_rate
            cr *= cr_decay
            f_scale = max(f_scale, 0.3)
            cr = max(cr, 0.3)
        
        # Final local polish on top Pareto points
        if used < budget and time.time() < deadline - 30:
            # Find non-dominated points
            non_dom_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        non_dom_mask[i] = False
                        break
            
            if np.any(non_dom_mask):
                pareto_idx = np.where(non_dom_mask)[0]
            else:
                pareto_idx = np.arange(len(F))
            
            # Sort by sum of objectives and take top 5
            sums = np.sum(F[pareto_idx], axis=1)
            top_order = np.argsort(sums)[:min(5, len(pareto_idx))]
            top_idx = pareto_idx[top_order]
            
            # Local polish with Nelder-Mead on composite objective
            for idx in top_idx:
                if used >= budget or time.time() > deadline - 10:
                    break
                
                x0 = X[idx].copy()
                
                # Use a scalarized objective for local optimization
                # Tchebycheff weighting to approximate Pareto front
                def scalarized(x):
                    fx = problem(x)
                    # Weighted sum with small perturbation for diversity
                    return 0.5 * fx[0] + 0.5 * fx[1]
                
                # Use bounded Nelder-Mead
                try:
                    result = minimize(
                        scalarized,
                        x0,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result.success or result.fun < np.sum(F[idx]):
                        new_x = np.clip(result.x, lo, hi)
                        new_f = problem(new_x)
                        used += 1
                        # Only accept if it's non-dominated or better in some way
                        if np.any(new_f < F[idx]) or np.all(new_f <= F[idx]):
                            X[idx] = new_x
                            F[idx] = new_f
                except Exception:
                    pass
        
        return X
# EVOLVE-BLOCK-END
