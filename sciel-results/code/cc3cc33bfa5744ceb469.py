"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        
        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def non_dominated_sort(F):
            """Return list of fronts (list of indices)."""
            n = len(F)
            S = [[] for _ in range(n)]  # domination count
            n_dom = np.zeros(n, dtype=int)
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # j dominates i?
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
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
            
            # If some points are not assigned (shouldn't happen but safety)
            assigned = set()
            for front in fronts:
                for i in front:
                    assigned.add(i)
            remaining = [i for i in range(n) if i not in assigned]
            if remaining:
                fronts.append(remaining)
            
            return fronts

        def crowding_distance(F, front_indices):
            """Compute crowding distance for given front indices."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            m = self.n_obj
            for k in range(m):
                # Sort by objective k
                sorted_idx = np.argsort(F[front_indices, k])
                sorted_idx = front_indices[sorted_idx]
                
                # Assign infinity to boundary points
                dist[0] = np.inf
                dist[-1] = np.inf
                
                # Normalize range
                min_val = F[sorted_idx[0], k]
                max_val = F[sorted_idx[-1], k]
                range_val = max_val - min_val
                if range_val == 0:
                    continue
                
                for i in range(1, n - 1):
                    idx = sorted_idx[i]
                    pos = np.where(front_indices == idx)[0][0]
                    dist[pos] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / range_val
            
            return dist

        def fast_non_dominated_sort(F):
            """Return rank and crowding distance for each point."""
            n = len(F)
            fronts = non_dominated_sort(F)
            
            ranks = np.zeros(n, dtype=int)
            for rank, front in enumerate(fronts):
                for i in front:
                    ranks[i] = rank
            
            # Compute crowding distances per front
            crowd = np.zeros(n)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowd[idx] = cd[i]
            
            return ranks, crowd

        while used < self.budget:
            # Get ranks and crowding distances
            ranks, crowd = fast_non_dominated_sort(F)
            
            # Select parents using tournament selection
            parents = []
            for _ in range(pop_size):
                # Tournament of size 2
                i = rng.integers(pop_size)
                j = rng.integers(pop_size)
                
                if ranks[i] < ranks[j]:
                    parent_idx = i
                elif ranks[j] < ranks[i]:
                    parent_idx = j
                else:
                    # Same rank, use crowding distance
                    if crowd[i] > crowd[j]:
                        parent_idx = i
                    else:
                        parent_idx = j
                parents.append(X[parent_idx])
            
            # Create children using SBX crossover and polynomial mutation
            children_X = []
            children_F = []
            
            n_children = min(pop_size, self.budget - used)
            
            for _ in range(n_children):
                p1_idx = rng.integers(pop_size)
                p2_idx = rng.integers(pop_size)
                
                # Simple blend crossover (SBX-like)
                alpha = rng.random()
                child1 = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]
                child2 = (1 - alpha) * X[p1_idx] + alpha * X[p2_idx]
                
                # Polynomial mutation
                for child in [child1, child2]:
                    if rng.random() < 0.1:  # mutation rate
                        child = child + 0.05 * span * rng.standard_normal(self.dim)
                        child = np.clip(child, lo, hi)
                
                children_X.append(child1)
                children_X.append(child2)
            
            # Evaluate children
            for cx in children_X:
                if used >= self.budget:
                    break
                cf = problem(cx)
                children_F.append(cf)
                used += 1
            
            # Combine parents and children
            combined_X = np.vstack([X, np.array(children_X)])
            combined_F = np.vstack([F, np.array(children_F)])
            
            # Selection: keep best pop_size individuals
            ranks, crowd = fast_non_dominated_sort(combined_F)
            
            # Sort by rank, then by crowding distance (descending)
            order = np.lexsort(((-crowd, ranks), np.arange(len(combined_X))))
            
            # Select top pop_size
            selected_idx = order[:pop_size]
            X = combined_X[selected_idx]
            F = combined_F[selected_idx]
        
        return X

# EVOLVE-BLOCK-END
