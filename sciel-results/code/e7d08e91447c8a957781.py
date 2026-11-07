"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting on F (n x m array).
        Returns a list of lists of indices, where each inner list is a front.
        """
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # n_dom[i] = number of points dominating i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        current_front = [i for i in range(n) if n_dom[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of indices in F.
        Returns a dictionary {index: distance}.
        """
        if len(indices) <= 2:
            return {i: float('inf') for i in indices}
        
        distances = {i: 0.0 for i in indices}
        m = self.n_obj
        for k in range(m):
            sorted_indices = sorted(indices, key=lambda i: F[i, k])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            f_min = F[sorted_indices[0], k]
            f_max = F[sorted_indices[-1], k]
            if f_max - f_min == 0:
                continue
                
            for j in range(1, len(sorted_indices) - 1):
                i_prev = sorted_indices[j - 1]
                i_next = sorted_indices[j + 1]
                distances[sorted_indices[j]] += (F[i_next, k] - F[i_prev, k]) / (f_max - f_min)
        
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        # Initialize population with random points
        pop_size = 20
        X = (lo + (hi - lo) * np.random.random((pop_size, self.dim)))
        F = np.array([problem(x) for x in X])
        evaluations = pop_size
        
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Combine parent and offspring
            # Create offspring by crossover and mutation
            offspring_X = np.empty_like(X)
            for i in range(pop_size):
                # Binary tournament selection
                i1 = np.random.randint(pop_size)
                i2 = np.random.randint(pop_size)
                
                # Simple selection: pick the one with better non-domination rank
                # For simplicity, we just pick randomly or use a simple heuristic
                # Let's use a simple random parent selection for now, or better:
                # Pick two parents via tournament on non-domination rank
                # But we haven't sorted yet. Let's just pick random parents.
                p1 = X[np.random.randint(pop_size)]
                p2 = X[np.random.randint(pop_size)]
                
                # Simulated Binary Crossover (SBX) - simplified
                if np.random.random() < 0.9:
                    alpha = np.random.random(self.dim)
                    child = alpha * p1 + (1 - alpha) * p2
                else:
                    child = p1.copy()
                
                # Mutation
                if np.random.random() < 0.9:
                    sigma = 0.1 * (hi - lo)
                    child = child + sigma * np.random.standard_normal(self.dim)
                
                child = np.clip(child, lo, hi)
                offspring_X[i] = child
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            evaluations += pop_size
            
            # Combine parent and offspring
            all_X = np.vstack([X, offspring_X])
            all_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(all_F)
            
            # Assign ranks
            ranks = np.empty(len(all_F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Select next population
            new_pop_size = 0
            new_fronts = []
            for front in fronts:
                if new_pop_size + len(front) <= pop_size:
                    new_fronts.append(front)
                    new_pop_size += len(front)
                else:
                    # Need to truncate this front
                    need = pop_size - new_pop_size
                    cd = self._crowding_distance(all_F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    new_fronts.append(sorted_front[:need])
                    new_pop_size += need
                    break
            
            new_indices = []
            for front in new_fronts:
                new_indices.extend(front)
            
            X = all_X[new_indices]
            F = all_F[new_indices]
        
        return X
# EVOLVE-BLOCK-END
