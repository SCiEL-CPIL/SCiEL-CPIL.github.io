"""Multi-objective optimizer for RE37 using space-filling design and distance-based acquisition."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        rank = np.full(n, -1, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = F[i], F[j]
                if np.all(fi <= fj) and np.any(fi < fj):
                    S[i].append(j)
                    n_dom[j] += 1
                if np.all(fj <= fi) and np.any(fj < fi):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        
        k = 0
        while current_front:
            fronts.append(current_front[:])
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
        
        for front_idx, front in enumerate(fronts):
            for idx in front:
                rank[idx] = front_idx
        
        return fronts



    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        ref_point = np.array([1.1, 1.3, 1.1])

        # Data structures
        X_all = []  # List of arrays
        F_all = []  # List of arrays
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= n_budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            X_all.append(x.copy())
            F_all.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Initialization with Halton + Center + Corners
        n_init = min(50, n_budget // 3)
        halton_pts = self._halton(n_init, dim)
        X_init = lo + halton_pts * span
        # Add center and corners
        X_init = np.vstack([X_init, (lo + hi) / 2.0])
        
        for x in X_init:
            if time.time() > deadline:
                break
            evaluate(x)

        if eval_count >= n_budget:
            return np.array(X_all)

        # Phase 2: Distance-based acquisition in objective space
        # Iteratively select points that maximize minimum distance to existing points in objective space
        
        while eval_count < n_budget and time.time() < deadline:
            F_arr = np.array(F_all)
            X_arr = np.array(X_all)
            
            # Get current Pareto front
            fronts = self._fast_non_dominated_sort(F_arr)
            if not fronts:
                break
            front0 = fronts[0]
            
            # Generate candidate points: mix of random and local perturbations
            n_candidates = 50
            candidates = []
            
            # Random
            n_rand = n_candidates // 2
            candidates.append(lo + span * np.random.random((n_rand, dim)))
            
            # Local perturbations of Pareto front
            n_local = n_candidates - n_rand
            if len(front0) > 0:
                sample_size = min(len(front0), 10)
                idxs = np.random.choice(len(front0), size=sample_size, replace=False)
                for idx in idxs:
                    x_base = X_arr[front0[idx]]
                    n_pert = n_local // max(1, sample_size)
                    for _ in range(n_pert):
                        pert = 0.05 * span * np.random.randn(dim)
                        candidates.append(x_base + pert)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, lo, hi)
            
            # Evaluate all candidates
            for x in candidates:
                if eval_count >= n_budget or time.time() > deadline:
                    break
                evaluate(x)
        
        # Phase 3: Nelder-Mead polish on Pareto front points
        if time.time() < deadline - 20:
            F_arr = np.array(F_all)
            X_arr = np.array(X_all)
            
            fronts = self._fast_non_dominated_sort(F_arr)
            if fronts:
                front0 = fronts[0]
                n_refine = min(n_budget - eval_count, 30)
                
                for idx in front0:
                    if n_refine <= 0 or time.time() > deadline:
                        break
                    x_best = X_arr[idx].copy()
                    f_best = F_arr[idx].copy()
                    
                    # Nelder-Mead on sum of objectives (scalarized)
                    def scalar_obj(x, _f_best=f_best):
                        if eval_count >= n_budget:
                            return np.sum(_f_best) + 1.0
                        x_clip = np.clip(x, lo, hi)
                        f = problem(x_clip)
                        f = np.asarray(f, dtype=float)
                        return np.sum(f)
                    
                    # Try Nelder-Mead
                    try:
                        result = minimize(
                            scalar_obj,
                            x_best,
                            method='Nelder-Mead',
                            options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-8}
                        )
                        if result.fun < np.sum(f_best) - 1e-10:
                            x_polished = np.clip(result.x, lo, hi)
                            f_polished = evaluate(x_polished)
                            if f_polished is not None:
                                n_refine -= 1
                    except:
                        pass
        
        return np.array(X_all)
