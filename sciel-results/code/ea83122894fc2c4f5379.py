# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Sobol

class Optimizer:
    """Multi-objective optimizer using Sobol global sampling + local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, self.n_obj

    @staticmethod
    def _fast_non_dominated_sort(F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    @staticmethod
    def _crowding_distance(F, indices):
        """Compute crowding distance for points at given indices."""
        m = len(indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        n_obj = F.shape[1]
        for k in range(n_obj):
            vals = F[indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget
        
        # 1. Global Sampling with Sobol Sequence
        n_global = 150
        n_polish = budget - n_global
        
        # Generate Sobol points
        sobol = Sobol(d=dim, scramble=True)
        n_pow2 = 1
        while n_pow2 < n_global:
            n_pow2 *= 2
            
        sobol_points = sobol.random(n_pow2)
        sobol_points = sobol_points[:n_global]
        
        # Map to bounds
        X_global = lo + (hi - lo) * sobol_points
        
        # Evaluate global points
        F_global = np.array([problem(x) for x in X_global])
        used = n_global
        
        X_all = X_global
        F_all = F_global
        
        # 2. Local Polish
        if used < budget:
            fronts = self._fast_non_dominated_sort(F_all)
            if fronts:
                front0 = fronts[0]
                max_polish_points = min(len(front0), 5)
                
                if len(front0) > max_polish_points:
                    cd = self._crowding_distance(F_all, front0)
                    top_idx = np.argsort(-cd)[:max_polish_points]
                    polish_idx = [front0[i] for i in top_idx]
                else:
                    polish_idx = list(front0)
                    
                for idx in polish_idx:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    remaining = budget - used
                    if remaining < 5:
                        break
                    
                    x0 = X_all[idx]
                    f0 = F_all[idx]
                    
                    best_x = x0.copy()
                    best_f = f0.copy()
                    step_size = 0.1
                    evals_used_local = 0
                    
                    for _ in range(3):
                        if used + evals_used_local >= budget:
                            break
                        improved = False
                        for d in range(dim):
                            if used + evals_used_local >= budget:
                                break
                            for sign in [1, -1]:
                                if used + evals_used_local >= budget:
                                    break
                                trial = best_x.copy()
                                trial[d] += sign * step_size * (hi[d] - lo[d])
                                trial = np.clip(trial, lo, hi)
                                ft = problem(trial)
                                evals_used_local += 1
                                used += 1
                                
                                if np.all(ft <= best_f) and np.any(ft < best_f):
                                    best_x = trial
                                    best_f = ft
                                    improved = True
                                    break
                            if improved:
                                break
                        if not improved:
                            step_size *= 0.5
                        else:
                            step_size *= 0.7
                            
                    if np.all(best_f <= F_all[idx]) and np.any(best_f < F_all[idx]):
                        X_all[idx] = best_x
                        F_all[idx] = best_f

        return X_all
# EVOLVE-BLOCK-END
