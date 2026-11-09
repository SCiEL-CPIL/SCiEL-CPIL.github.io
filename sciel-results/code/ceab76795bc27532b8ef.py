"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Perform fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
        n_count = [0] * n  # n_count[i] = number of points dominating i
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_count[i] += 1
        
        for i in range(n):
            if n_count[i] == 0:
                fronts.append([i])
        
        k = 0
        while k < len(fronts):
            current_front = fronts[k]
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        F_sub = F[front_indices]
        
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            distances[sorted_idx[0]] = float('inf')
            distances[sorted_idx[-1]] = float('inf')
            
            f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
            if f_range == 0:
                continue
            
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / f_range
        
        return distances

    def _select_parents(self, F, N):
        """Select N parents using tournament selection based on non-domination rank and crowding distance."""
        fronts = self._fast_non_dominated_sort(F)
        
        # Assign ranks
        rank = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        # Compute crowding distance for each front
        crowding = np.zeros(len(F))
        for front in fronts:
            cd = self._crowding_distance(F, front)
            for j, idx in enumerate(front):
                crowding[idx] = cd[j]
        
        # Tournament selection
        selected = []
        for _ in range(N):
            i, j = np.random.randint(0, len(F), 2)
            if rank[i] < rank[j]:
                selected.append(i)
            elif rank[j] < rank[i]:
                selected.append(j)
            else:
                if crowding[i] > crowding[j]:
                    selected.append(i)
                else:
                    selected.append(j)
        
        return selected

    def _sbx_crossover(self, x1, x2, lo, hi, eta=15.0):
        """Simulated binary crossover."""
        child1 = x1.copy()
        child2 = x2.copy()
        
        for i in range(len(x1)):
            if np.random.random() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if np.random.random() < 0.5:
                        beta = 1 + 2 * (x1[i] - lo[i]) / (x2[i] - x1[i])
                    else:
                        beta = 1 + 2 * (hi[i] - x2[i]) / (x1[i] - x2[i])
                    alpha = 2 - beta ** (-1 - 1 / eta)
                    
                    if np.random.random() <= 1 / alpha:
                        beta_q = beta ** (1 / (eta + 1))
                    else:
                        beta_q = (1 / (2 - beta * alpha)) ** (1 / (eta + 1))
                    
                    child1[i] = 0.5 * ((1 + beta_q) * x1[i] + (1 - beta_q) * x2[i])
                    child2[i] = 0.5 * ((1 - beta_q) * x1[i] + (1 + beta_q) * x2[i])
        
        return np.clip(child1, lo, hi), np.clip(child2, lo, hi)

    def _polynomial_mutation(self, x, lo, hi, eta=20.0, prob=0.1):
        """Polynomial mutation."""
        child = x.copy()
        for i in range(len(x)):
            if np.random.random() < prob:
                delta = np.random.random()
                if delta < 0.5:
                    val = (2 * delta) ** (1 / (eta + 1)) - 1
                else:
                    val = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))
                
                child[i] += val * (hi[i] - lo[i])
        
        return np.clip(child, lo, hi)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = min(100, max(20, budget // 2))
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5:
                break
            
            # Selection
            parents = self._select_parents(F, 2)
            
            # Crossover
            if np.random.random() < 0.9:  # Crossover probability
                child1, child2 = self._sbx_crossover(X[parents[0]], X[parents[1]], lo, hi)
            else:
                child1 = X[parents[0]].copy()
                child2 = X[parents[1]].copy()
            
            # Mutation
            child1 = self._polynomial_mutation(child1, lo, hi)
            child2 = self._polynomial_mutation(child2, lo, hi)
            
            # Evaluate children
            fc1 = problem(child1)
            used += 1
            if used >= budget or time.time() > deadline - 2:
                # Add only the first child if we're out of budget
                X = np.vstack([X, child1])
                F = np.vstack([F, fc1])
                break
            
            fc2 = problem(child2)
            used += 1
            
            # Combine parent and child populations
            X_new = np.vstack([X, child1, child2])
            F_new = np.vstack([F, fc1, fc2])
            
            # Truncation: keep best pop_size individuals
            fronts = self._fast_non_dominated_sort(F_new)
            
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(X_new[idx])
                        new_F.append(F_new[idx])
                else:
                    # Need to select from this front based on crowding distance
                    cd = self._crowding_distance(F_new, front)
                    sorted_front = sorted(front, key=lambda i: cd[i - front[0]], reverse=True)
                    remaining = pop_size - len(new_X)
                    for idx in sorted_front[:remaining]:
                        new_X.append(X_new[idx])
                        new_F.append(F_new[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        # Return the last population
        return X
# EVOLVE-BLOCK-END
