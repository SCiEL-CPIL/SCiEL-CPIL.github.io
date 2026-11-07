"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using NSGA-II style approach with adaptive budget allocation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Store all evaluated points and their objective vectors
        X = []  # list of x vectors
        F = []  # list of f vectors
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            return f

        def non_dominated_sort(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            if n == 0:
                return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    # Check if j dominates i
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
                    # Check if i dominates j
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
            return [i for i in range(n) if not dominated[i]]

        # Phase 1: Latin hypercube sampling for better initial coverage
        n_initial = min(60, self.budget // 4)
        for i in range(n_initial):
            if time.time() > deadline:
                break
            # Latin hypercube-like sampling
            x = np.array([(i + rng.random()) / n_initial for _ in range(self.dim)])
            x = lo + (hi - lo) * x
            eval_point(x)

        # Phase 2: NSGA-II style evolutionary algorithm
        if used < self.budget and time.time() < deadline:
            # Population size
            pop_size = max(20, min(40, (self.budget - used) // 8))
            
            # Build initial population from non-dominated points
            F_arr = np.array(F)
            nd_idx = non_dominated_sort(F_arr)
            
            # If we don't have enough non-dominated points, add random ones
            while len(nd_idx) < pop_size and used < self.budget:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = eval_point(x)
                if f is not None:
                    nd_idx.append(used - 1)
            
            # Take first pop_size points
            pop_idx = nd_idx[:pop_size]
            
            # Main evolutionary loop
            while used < self.bbudget and time.time() < deadline:
                # Select parents using tournament selection from non-dominated set
                F_arr = np.array(F)
                nd_idx = non_dominated_sort(F_arr)
                if len(nd_idx) < 2:
                    nd_idx = list(range(len(F_arr)))
                
                # Pick two parents via tournament selection
                p1 = nd_idx[rng.integers(len(nd_idx))]
                p2 = nd_idx[rng.integers(len(nd_idx))]
                
                # Simulated binary crossover (SBX)
                x1 = X[p1]
                x2 = X[p2]
                child = np.zeros(self.dim)
                for d in range(self.dim):
                    if rng.random() < 0.5:
                        # SBX for this dimension
                        if abs(x1[d] - x2[d]) > 1e-10:
                            beta = 1.0
                            if rng.random() < 0.5:
                                beta = (2 * rng.random()) ** (1.0 / 15.0)
                            else:
                                beta = (1.0 / (2.0 * (1.0 - rng.random()))) ** (1.0 / 15.0)
                            child[d] = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                        else:
                            child[d] = x1[d]
                    else:
                        if abs(x1[d] - x2[d]) > 1e-10:
                            beta = 1.0
                            if rng.random() < 0.5:
                                beta = (2 * rng.random()) ** (1.0 / 15.0)
                            else:
                                beta = (1.0 / (2.0 * (1.0 - rng.random()))) ** (1.0 / 15.0)
                            child[d] = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                        else:
                            child[d] = x2[d]
                
                # Polynomial mutation
                mutation_rate = 0.2
                for d in range(self.dim):
                    if rng.random() < mutation_rate:
                        # Polynomial mutation
                        if hi[d] - lo[d] > 1e-10:
                            if rng.random() < 0.5:
                                delta = (2 * rng.random()) ** (1.0 / 20.0)
                                child[d] += delta * (hi[d] - lo[d])
                            else:
                                delta = (2 * rng.random()) ** (1.0 / 20.0)
                                child[d] -= delta * (hi[d] - lo[d])
                
                child = np.clip(child, lo, hi)
                fc = eval_point(child)
                if fc is None:
                    break

        # Phase 3: Local refinement of best non-dominated points
        if used < self.budget and time.time() < deadline:
            F_arr = np.array(F)
            nd_idx = non_dominated_sort(F_arr)
            
            # Refine top non-dominated points with local search
            n_refine = min(3, len(nd_idx), max(1, (self.budget - used) // 15))
            for idx in nd_idx[:n_refine]:
                if time.time() > deadline or used >= self.budget:
                    break
                x0 = X[idx]
                
                # Try optimizing each objective separately with tighter bounds
                for obj_i in range(self.n_obj):
                    if time.time() > deadline or used >= self.budget:
                        break
                    
                    def objective(x, _obj_i=obj_i):
                        nonlocal used
                        if used >= self.budget:
                            return 0.0
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x))
                        X.append(x)
                        F.append(f)
                        used += 1
                        return f[_obj_i]
                    
                    try:
                        res = minimize(objective, x0, method='L-BFGS-B',
                                     bounds=list(zip(lo, hi)),
                                     options={'maxiter': 15, 'maxfun': 20})
                    except Exception:
                        pass

        return X
# EVOLVE-BLOCK-END
