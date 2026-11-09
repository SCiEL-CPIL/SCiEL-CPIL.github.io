# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """NSGA-II multi-objective search for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 300.0 - 20.0  # Safety margin
        budget = int(self.budget)
        dim = int(self.dim)
        n_obj = int(self.n_obj)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        if np.any(span <= 0):
            span = np.where(span <= 0, 1.0, span)

        rng = np.random.default_rng(12345)

        # Population size
        pop_size = 20
        if budget < pop_size * 2:
            pop_size = max(2, budget // 2)

        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        if used >= budget:
            return F

        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                if n_count[i] == 0:
                    rank[i] = 0
            
            I = [i for i in range(n) if rank[i] == 0]
            k = 0
            while len(I) > 0:
                k += 1
                Q = []
                for i in I:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k
                            Q.append(j)
                I = Q
            return rank

        def crowding_distance(F, rank, pop_indices):
            n = len(pop_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            distances = np.zeros(n)
            front_F = F[pop_indices]
            m = self.n_obj
            for m_idx in range(m):
                sorted_idx = np.argsort(front_F[:, m_idx])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                f_min = front_F[sorted_idx[0], m_idx]
                f_max = front_F[sorted_idx[-1], m_idx]
                if f_max - f_min == 0:
                    continue
                for k in range(1, n - 1):
                    distances[sorted_idx[k]] += (front_F[sorted_idx[k+1], m_idx] - front_F[sorted_idx[k-1], m_idx]) / (f_max - f_min)
            return distances

        def select_parents(F, rank, pop_size):
            ranks = {}
            for i in range(pop_size):
                r = rank[i]
                if r not in ranks:
                    ranks[r] = []
                ranks[r].append(i)
            
            cd = np.zeros(pop_size)
            for r, idxs in ranks.items():
                cd[idxs] = crowding_distance(F, rank, idxs)
            
            parents = []
            for _ in range(pop_size):
                i1 = rng.integers(pop_size)
                i2 = rng.integers(pop_size)
                if rank[i1] < rank[i2]:
                    parents.append(i1)
                elif rank[i2] < rank[i1]:
                    parents.append(i2)
                else:
                    if cd[i1] > cd[i2]:
                        parents.append(i1)
                    else:
                        parents.append(i2)
            return parents

        def sbx_crossover(a, b, eta=20.0):
            if rng.random() < 0.5:
                return a, b
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            for i in range(dim):
                if rng.random() < 0.5:
                    child1[i] = a[i]
                    child2[i] = b[i]
                else:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (2 * (1 - u)) ** (1 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
                    child2[i] = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
            return child1, child2

        def poly_mutation(x, eta=20.0):
            y = x.copy()
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    y[i] = x[i] + delta_q * span[i]
            y = np.clip(y, lo, hi)
            return y

        # Main NSGA-II Loop
        while used < budget:
            if time.time() - t0 > deadline:
                break
            
            rank = fast_non_dominated_sort(F)
            parents = select_parents(F, rank, pop_size)
            
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(pop_size, budget - used)
            
            for _ in range(n_offspring):
                p1_idx = parents[rng.integers(pop_size)]
                p2_idx = parents[rng.integers(pop_size)]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                offspring_X.append(c2)
                offspring_F.append(f2)
                
                if used >= budget:
                    break
            
            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_rank = fast_non_dominated_sort(combined_F)
            
            combined_cd = np.zeros(len(combined_F))
            ranks = {}
            for i in range(len(combined_F)):
                r = combined_rank[i]
                if r not in ranks:
                    ranks[r] = []
                ranks[r].append(i)
            
            for r, idxs in ranks.items():
                combined_cd[idxs] = crowding_distance(combined_F, combined_rank, idxs)
            
            sort_keys = np.array([(combined_rank[i], -combined_cd[i]) for i in range(len(combined_F))])
            sorted_indices = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))
            
            new_indices = sorted_indices[:pop_size]
            
            X = combined_X[new_indices]
            F = combined_F[new_indices]
            
            if used >= budget:
                break

        return F
# EVOLVE-BLOCK-END
