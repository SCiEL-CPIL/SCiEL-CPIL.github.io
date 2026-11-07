"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer with time guard and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Larger initial population for better coverage
        pop = min(40, self.budget // 3)
        X = lo + span * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop

        def is_dominated(f, F_arr):
            """Check if f is dominated by any point in F_arr."""
            return np.any(np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1))

        def get_non_dominated_idx(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            nd_idx = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            return np.array(nd_idx, dtype=int) if nd_idx else np.array([0], dtype=int)

        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline - 30.0:
                break  # Reserve time for polish

            nd_idx = get_non_dominated_idx(F)
            parents = X[nd_idx]
            
            # Tournament selection from non-dominated
            t_size = min(3, len(parents))
            candidates = rng.choice(len(parents), size=t_size, replace=False)
            # Pick the one with best sum (proxy for quality)
            best_cand = candidates[np.argmin(np.sum(parents[candidates], axis=1))]
            parent = parents[best_cand]
            
            # Adaptive mutation: smaller steps as we approach deadline
            progress = used / self.budget
            step_size = 0.15 * (1.0 - 0.5 * progress)
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point (highest sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc

        # Local polish on non-dominated points
        if time.time() < deadline - 5.0:
            nd_idx = get_non_dominated_idx(F)
            for idx in nd_idx:
                if time.time() > deadline - 2.0:
                    break
                x0 = X[idx]
                f0 = F[idx]
                
                # Try to improve each objective individually
                for obj_i in range(self.n_obj):
                    if time.time() > deadline - 1.0:
                        break
                    
                    # Minimize objective obj_i
                    def obj_func(x, i=obj_i):
                        return problem(x)[i]
                    
                    try:
                        res = minimize(
                            obj_func, x0, method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 50, 'ftol': 1e-10}
                        )
                        if res.success and res.fun < f0[obj_i]:
                            # Check if new point is still non-dominated or improves HV
                            new_f = problem(res.x)
                            if not is_dominated(new_f, F):
                                F[idx] = new_f
                                X[idx] = res.x
                    except Exception:
                        pass
        
        return X
# EVOLVE-BLOCK-END
