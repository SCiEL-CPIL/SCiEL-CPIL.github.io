"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective differential evolution for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size: 40 as suggested
        pop_size = 40
        if pop_size > budget:
            pop_size = max(10, budget // 2)
        
        # DE parameters
        F = 0.9  # Mutation scale factor
        CR = 0.9  # Crossover rate

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F_obj = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F_obj):
            """Returns ranks and crowding distances."""
            n = len(F_obj)
            ranks = np.zeros(n, dtype=int)
            crowding = np.zeros(n)
            
            # Non-dominated sorting
            S = [set() for _ in range(n)]  # Dominance set
            n_count = np.zeros(n, dtype=int)  # Number of solutions dominated
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F_obj[i] <= F_obj[j]) and np.any(F_obj[i] < F_obj[j]):
                        S[i].add(j)
                    elif np.all(F_obj[j] <= F_obj[i]) and np.any(F_obj[j] < F_obj[i]):
                        n_count[i] += 1
            
            current_front = set()
            for i in range(n):
                if n_count[i] == 0:
                    ranks[i] = 0
                    current_front.add(i)
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = set()
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            ranks[j] = k + 1
                            next_front.add(j)
                current_front = next_front
                k += 1
            
            # Assign crowding distance
            for front in fronts:
                if len(front) <= 2:
                    for i in front:
                        crowding[i] = float('inf')
                    continue
                
                for m in range(n_obj):
                    sorted_front = sorted(front, key=lambda i: F_obj[i][m])
                    crowding[sorted_front[0]] = float('inf')
                    crowding[sorted_front[-1]] = float('inf')
                    obj_range = F_obj[sorted_front[-1]][m] - F_obj[sorted_front[0]][m]
                    if obj_range > 1e-12:
                        for i in range(1, len(sorted_front) - 1):
                            idx = sorted_front[i]
                            prev_idx = sorted_front[i-1]
                            next_idx = sorted_front[i+1]
                            crowding[idx] += (F_obj[next_idx][m] - F_obj[prev_idx][m]) / obj_range
            
            return ranks, crowding

        def selection(ranks, crowding, F_obj, X, n_select):
            """Select n_select individuals based on rank and crowding distance."""
            n = len(F_obj)
            # Sort by rank, then by crowding distance (descending)
            indices = list(range(n))
            indices.sort(key=lambda i: (ranks[i], -crowding[i]))
            
            selected = []
            for i in indices:
                if len(selected) >= n_select:
                    break
                selected.append(i)
            
            # If we still need more, just take from the rest
            while len(selected) < n_select:
                for i in indices:
                    if i not in selected:
                        selected.append(i)
                        if len(selected) >= n_select:
                            break
                break
            
            return [X[i] for i in selected], [F_obj[i] for i in selected]

        # Main DE loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring using DE/rand/1
            new_X = []
            new_F = []
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Select three distinct parents
                while True:
                    a = rng.integers(pop_size)
                    b = rng.integers(pop_size)
                    c = rng.integers(pop_size)
                    if len({a, b, c}) == 3 and a != i:
                        break
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = X[a] + F * (X[b] - X[c])
                
                # Crossover
                r = rng.random(dim)
                j_rand = rng.integers(dim)
                u = np.where(r <= CR, v, X[i])
                u[j_rand] = v[j_rand]
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                fu = problem(u)
                used += 1
                
                new_X.append(u)
                new_F.append(fu)
            
            if not new_X:
                break
            
            new_X = np.array(new_X)
            new_F = np.array(new_F)
            
            # Combine population and offspring
            combined_X = np.vstack([X, new_X])
            combined_F = np.vstack([F_obj, new_F])
            
            # Non-dominated sort
            ranks, crowding = non_dominated_sort(combined_F)
            
            # Select next generation
            X, F_obj = selection(ranks, crowding, combined_F, combined_X, pop_size)
            
            # If we've used all budget, break
            if used >= budget:
                break
        
        return X
# EVOLVE-BLOCK-END
