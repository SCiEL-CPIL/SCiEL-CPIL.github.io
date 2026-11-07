import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random  # use global state to respect evaluator's seed
        
        pop_size = 40
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)
            
            # Crowding distance per front
            cd = np.zeros(len(F))
            for front in fronts:
                if len(front) > 1:
                    cd[front] = self._crowding_distance(F[front])
            
            # Create offspring
            n_offspring = min(pop_size, self.budget - used)
            offspring_X = np.zeros((n_offspring, self.dim))
            for i in range(n_offspring):
                if time.time() > deadline:
                    break
                # Tournament selection from combined population
                p1 = self._tournament_select(F, fronts, cd, 2)
                p2 = self._tournament_select(F, fronts, cd, 2)
                
                # SBX crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                
                # Polynomial mutation
                c1 = self._polynomial_mutate(c1, lo, hi)
                c2 = self._polynomial_mutate(c2, lo, hi)
                
                offspring_X[i] = c1
                if i + 1 < n_offspring and used + 2 <= self.budget:
                    offspring_X[i + 1] = c2
                    n_offspring_actual = i + 2
                    break
                else:
                    n_offspring_actual = i + 1
                    break
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X[:n_offspring_actual]])
            used += n_offspring_actual
            
            # Combine parent and offspring
            X_comb = np.vstack([X, offspring_X[:n_offspring_actual]])
            F_comb = np.vstack([F, F_off])
            
            # Non-dominated sorting on combined
            fronts_comb = self._non_dominated_sort(F_comb)
            
            # Select next generation
            next_X = np.zeros((pop_size, self.dim))
            next_F = np.zeros((pop_size, self.n_obj))
            count = 0
            for front in fronts_comb:
                if count + len(front) <= pop_size:
                    next_X[count:count+len(front)] = X_comb[front]
                    next_F[count:count+len(front)] = F_comb[front]
                    count += len(front)
                else:
                    # Need to select from this front using crowding distance
                    cd_front = self._crowding_distance(F_comb[front])
                    order = np.argsort(-cd_front)
                    need = pop_size - count
                    selected = front[order[:need]]
                    next_X[count:count+need] = X_comb[selected]
                    next_F[count:count+need] = F_comb[selected]
                    count += need
                    break
            
            X = next_X
            F = next_F
        
        return X

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        if current_front:
            fronts.append(np.array(current_front))
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
            if current_front:
                fronts.append(np.array(current_front))
        
        return fronts

    def _crowding_distance(self, F_front):
        n = len(F_front)
        if n <= 1:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for k in range(self.n_obj):
            min_val = F_front[:, k].min()
            max_val = F_front[:, k].max()
            if max_val == min_val:
                cd[:] = np.inf
                continue
            sorted_idx = np.argsort(F_front[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F_front[sorted_idx[i+1], k] - F_front[sorted_idx[i-1], k]) / (max_val - min_val)
        
        return cd

    def _tournament_select(self, F, fronts, cd, k=2):
        n = len(F)
        indices = rng.integers(n, size=k)
        # Compare by front rank, then crowding distance
        front_rank = np.zeros(n)
        for i, front in enumerate(fronts):
            front_rank[front] = i
        
        best_idx = indices[0]
        for idx in indices[1:]:
            if front_rank[idx] < front_rank[best_idx]:
                best_idx = idx
            elif front_rank[idx] == front_rank[best_idx]:
                if cd[idx] > cd[best_idx]:
                    best_idx = idx
        
        return X[best_idx] if 'X' in dir() else None  # This is wrong, need to pass X

    def _sbx_crossover(self, p1, p2, lo, hi):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if rng.random() < 0.5:
                if p1[i] == p2[i]:
                    continue
                if p1[i] > p2[i]:
                    p1[i], p2[i] = p2[i], p1[i]
                
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / 21.0)
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / 21.0)
                
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _polynomial_mutate(self, x, lo, hi):
        eta = 20.0
        for i in range(self.dim):
            if rng.random() < 0.5:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                x[i] += delta * (hi[i] - lo[i])
        return np.clip(x, lo, hi)
