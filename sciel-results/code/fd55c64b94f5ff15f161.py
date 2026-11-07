import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=int), np.array([])
            
            ranks = np.zeros(n, dtype=int)
            crowding = np.zeros(n)
            
            front_mask = np.ones(n, dtype=bool)
            rank = 0
            
            while front_mask.any():
                front_indices = np.where(front_mask)[0]
                front_F = F[front_indices]
                
                # Vectorized dominance check
                le = np.all(front_F[None, :, :] <= front_F[:, None, :], axis=2)
                lt = np.any(front_F[None, :, :] < front_F[:, None, :], axis=2)
                dominated = le & lt  # dominated[i,j] = j dominates i
                
                is_dominated = dominated.any(axis=1)
                nd_mask = ~is_dominated
                nd_indices = front_indices[nd_mask]
                
                for i in nd_indices:
                    ranks[i] = rank
                
                if len(nd_indices) > 1:
                    front_F_nd = F[nd_indices]
                    nf_nd = len(nd_indices)
                    for k in range(n_obj):
                        sorted_order = np.argsort(front_F_nd[:, k])
                        f_min = front_F_nd[sorted_order[0], k]
                        f_max = front_F_nd[sorted_order[-1], k]
                        if f_max <= f_min:
                            continue
                        for idx in range(nf_nd):
                            i = nd_indices[sorted_order[idx]]
                            if idx == 0 or idx == nf_nd - 1:
                                crowding[i] = np.inf
                            else:
                                prev_val = front_F_nd[sorted_order[idx-1], k]
                                next_val = front_F_nd[sorted_order[idx+1], k]
                                crowding[i] += (next_val - prev_val) / (f_max - f_min)
                
                front_mask[nd_indices] = False
                rank += 1
            
            return ranks, crowding
        
        def tournament_select(X, F, k=2):
            indices = np.random.randint(0, len(F), size=k)
            best = indices[0]
            for i in indices[1:]:
                if np.sum(F[i]) < np.sum(F[best]):
                    best = i
            return X[best]
        
        def sbcrossover(x1, x2, eta=20.0):
            x = np.zeros_like(x1)
            for i in range(dim):
                if np.random.rand() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) - beta * (x1[i] - x2[i]))
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                        x[i] = 0.5 * ((x1[i] + x2[i]) + beta * (x1[i] - x2[i]))
            return np.clip(x, lo, hi)
        
        def polynomial_mutation(x, pmut=1.0/dim, eta=20.0):
            x_new = x.copy()
            for i in range(dim):
                if np.random.rand() < pmut:
                    range_i = hi[i] - lo[i]
                    if range_i < 1e-14:
                        continue
                    y = (x_new[i] - lo[i]) / range_i
                    if y < 0: y = 0
                    if y > 1: y = 1
                    u = np.random.rand()
                    if u < 0.5:
                        delta1 = (1 + 2 * (y - 0)) ** (eta + 1)
                        delta = (1 / delta1) ** (1 / (eta + 1)) - 1
                        y_new = y + delta * (1 - y)
                    else:
                        delta2 = (1 + 2 * (1 - y)) ** (eta + 1)
                        delta = 1 - (1 / delta2) ** (1 / (eta + 1))
                        y_new = y + delta * y
                    x_new[i] = lo[i] + y_new * range_i
            return np.clip(x_new, lo, hi)
        
        # Initialize with larger population
        pop_size = 40
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            n_off = min(pop_size, self.budget - evaluations)
            X_off = []
            for _ in range(n_off):
                p1 = tournament_select(X, F)
                p2 = tournament_select(X, F)
                child = sbcrossover(p1, p2)
                child = polynomial_mutation(child)
                X_off.append(child)
            
            X_off = np.array(X_off)
            F_off = np.array([problem(x) for x in X_off])
            evaluations += n_off
            
            X_all = np.vstack([X, X_off])
            F_all = np.vstack([F, F_off])
            
            ranks, crowding = non_dominated_sort(F_all)
            sort_idx = np.lexsort(((-crowding), ranks))
            X = X_all[sort_idx[:pop_size]]
            F = F_all[sort_idx[:pop_size]]
        
        return np.zeros((1, dim))
