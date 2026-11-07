# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    """Multi-objective Optimizer using LHS initialization and Tchebycheff Local Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        # 330s limit. Safety margin of 50s.
        self.deadline = self.start_time + 280

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        def add_point(x, f):
            if f is not None and len(f) == self.n_obj and np.all(np.isfinite(f)):
                x_clipped = np.clip(x, lo, hi)
                X_list.append(x_clipped)
                F_list.append(f)

        # Phase 1: Initialization using Latin Hypercube Sampling
        n_init = 40
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        samples = sampler.random(n=n_init)
        samples = lo + samples * (hi - lo)
        
        for x in samples:
            if self.eval_count >= self.budget or self._check_time():
                break
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
                
        if not F_list:
            x0 = (lo + hi) / 2
            f0 = self._evaluate(x0, problem)
            if f0 is not None: add_point(x0, f0)
            return X_list if X_list else [x0]

        def get_nondominated_indices(F):
            n = len(F)
            if n == 0:
                return []
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if is_dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        is_dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        F = np.array(F_list)
        X = np.array(X_list)
        
        # Generate weight vectors for Tchebycheff aggregation
        # 12 well-distributed points on the simplex
        weight_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([1.0/3, 1.0/3, 1.0/3]),
            np.array([0.7, 0.2, 0.1]),
            np.array([0.7, 0.1, 0.2]),
            np.array([0.2, 0.7, 0.1]),
            np.array([0.2, 0.1, 0.7]),
            np.array([0.1, 0.7, 0.2]),
            np.array([0.1, 0.2, 0.7])
        ]
        
        # 12 searches. 160 budget / 12 = 13.3
        maxfev_per_search = 12
        
        nd_indices = get_nondominated_indices(F)
        
        for i, w in enumerate(weight_vectors):
            if self.eval_count >= self.budget or self._check_time():
                break
            
            F_nd = F[nd_indices] if len(nd_indices) > 0 else F
            z = np.min(F_nd, axis=0)
            
            scores = np.max(w * (F - z), axis=1)
            best_idx = np.argmin(scores)
            x_start = X[best_idx].copy()
            
            def objective(x, w=w, z=z):
                f = self._evaluate(x, problem)
                if f is None:
                    return 1e6
                return np.max(w * (f - z))
            
            try:
                res = minimize(
                    objective, 
                    x_start, 
                    method='Nelder-Mead', 
                    bounds=list(zip(lo, hi)),
                    options={
                        'maxfev': maxfev_per_search, 
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if res.x is not None:
                    f_final = self._evaluate(res.x, problem)
                    if f_final is not None:
                        add_point(res.x, f_final)
                        F = np.array(F_list)
                        X = np.array(X_list)
                        nd_indices = get_nondominated_indices(F)
                        
            except Exception:
                pass

        return X_list
# EVOLVE-BLOCK-END
