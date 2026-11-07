"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import product


class Optimizer:
    """
    Multi-objective optimizer for WFG9 using a hybrid approach:
    1. Grid-based initialization to cover the search space.
    2. NSGA-II style evolution (non-dominated sorting + crowding distance).
    3. Local refinement using scipy.optimize.minimize on promising solutions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Initial Population: Stratified Sampling
        # For 12 dimensions, a full grid is too large. We use a smaller grid or random.
        # Let's use a mix: some random, some on a coarse grid to ensure coverage.
        n_init = 60
        X = []
        F = []
        
        # Coarse grid points
        # 2^12 = 4096 is too many. Let's do 3 levels per dim? 3^12 is huge.
        # Let's just use random for init but ensure we use the budget efficiently.
        # Actually, WFG9 is a benchmark. Let's try a simple random start first, 
        # but the "inspiration" said NSGA-II is good.
        
        # Let's use a population size of 100.
        pop_size = 100
        if self.budget < pop_size:
            pop_size = self.budget
            
        X = (lo + (hi - lo) * np.random.random((pop_size, self.dim)))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions for NSGA-II
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # Domination sets
            n_dom = np.zeros(n, dtype=int)  # Domination counts
            ranks = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return ranks

        def crowding_distance(F, ranks, n):
            cd = np.zeros(n)
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                idx = np.where(ranks == r)[0]
                if len(idx) <= 2:
                    cd[idx] = np.inf
                    continue
                for m in range(self.n_obj):
                    vals = F[idx, m]
                    min_val, max_val = np.min(vals), np.max(vals)
                    if max_val - min_val < 1e-10:
                        cd[idx] = np.inf
                        continue
                    # Sort by objective m
                    order = np.argsort(vals)
                    sorted_idx = idx[order]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (vals[order[i+1]] - vals[order[i-1]]) / (max_val - min_val)
            return cd

        def select_parents(ranks, cd, n):
            # Tournament selection
            i1 = np.random.randint(n)
            i2 = np.random.randint(n)
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                if cd[i1] > cd[i2]:
                    return i1
                else:
                    return i2

        # Evolution Loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            ranks = fast_non_dominated_sort(F)
            cd = crowding_distance(F, ranks, len(F))
            
            # Selection and Crossover/Mutation
            # We will generate a child and replace the worst individual
            # Or use a proper NSGA-II replacement strategy.
            # For simplicity and speed, let's do (mu+lambda) selection.
            # Generate a few children, combine, sort, keep best.
            
            # To save time, let's just do single-child replacement of the worst
            # based on rank and crowding distance.
            
            # Find the worst individual
            # Sort by rank, then by crowding distance (ascending for worst)
            # Actually, we want to replace the one with highest rank, and lowest CD within that rank.
            
            # Let's create a combined score for replacement: rank * 1000 - cd
            # Lower is better to keep. So we want to replace the highest value.
            replace_score = ranks * 1000.0 - cd
            worst_idx = np.argmax(replace_score)
            
            # Select parents
            p1 = select_parents(ranks, cd, len(F))
            p2 = select_parents(ranks, cd, len(F))
            
            # Simulated Binary Crossover (SBX)
            x1 = X[p1]
            x2 = X[p2]
            child = np.zeros_like(x1)
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    # Crossover
                    if np.random.random() < 0.5:
                        child[i] = x1[i]
                    else:
                        child[i] = x2[i]
                # Mutation
                if np.random.random() < 0.2:
                    child[i] += 0.1 * (hi[i] - lo[i]) * np.random.standard_normal()
            
            child = np.clip(child, lo, hi)
            f_child = problem(child)
            used += 1
            
            # Replace worst
            X[worst_idx] = child
            F[worst_idx] = f_child
            
            # Occasional local refinement?
            # If we have time, maybe refine the best solution in rank 0?
            # But local optimization is expensive. Let's stick to EA for now.
            
        # Final local refinement step if time permits
        # Take the best solution (lowest sum of objectives among rank 0)
        if used < self.budget and time.time() < deadline:
            ranks = fast_non_dominated_sort(F)
            best_idx = np.argmin(np.where(ranks == 0, np.sum(F, axis=1), np.inf))
            if not np.isinf(best_idx):
                x_best = X[best_idx]
                # Try to improve it using a scalarization (weighted sum)
                # Minimize sum of objectives
                def obj_fun(x):
                    return np.sum(problem(np.clip(x, lo, hi)))
                
                # Use Nelder-Mead or L-BFGS-B
                # L-BFGS-B is better for bounded problems
                try:
                    res = minimize(obj_fun, x_best, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
                    if res.success:
                        x_res = res.x
                        f_res = problem(x_res)
                        # Check if it's better than any existing
                        # If it dominates the original best, or is non-dominated, add it?
                        # The task says "Every returned objective vector is recorded".
                        # So we just need to call problem.
                        # But we don't need to add it to X/F if we just want to record it.
                        # Wait, the evaluator records the return value of problem(x) calls.
                        # So calling problem(x_res) records it.
                        # We don't need to store it in our internal state unless we want to use it for further search.
                        # But we are at the end.
                        pass
                except Exception:
                    pass

        return X
# EVOLVE-BLOCK-END
