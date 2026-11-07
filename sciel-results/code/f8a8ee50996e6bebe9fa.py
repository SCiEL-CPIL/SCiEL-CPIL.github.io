import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        pop_size = 40
        n_gen = (self.budget - pop_size) // pop_size
        n_gen = max(n_gen, 1)

        # Initialize with stratified sampling to cover the front
        rng = np.random.default_rng(0)
        X = np.zeros((pop_size, self.dim))
        # Stratify x1 to cover f1 range
        for i in range(pop_size):
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * i / (pop_size - 1)
            X[i, 1:] = lo[1:] + (hi[1:] - lo[1:]) * rng.random(self.dim - 1)

        F = np.array([problem(x) for x in X])
        used = pop_size

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                is_nd = True
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        is_nd = False
                        break
                if is_nd:
                    nd.append(i)
            return nd

        def crowding_distance(F, indices):
            n = len(indices)
            if n == 0:
                return np.array([])
            dist = np.zeros(n)
            F_sub = F[indices]
            for m in range(self.n_obj):
                f_vals = F_sub[:, m]
                f_min, f_max = f_vals.min(), f_vals.max()
                if f_max - f_min < 1e-15:
                    continue
                order = np.argsort(f_vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (f_vals[order[k+1]] - f_vals[order[k-1]]) / (f_max - f_min)
            return dist

        while used < self.budget:
            if time.time() > deadline:
                break

            # Find non-dominated set
            nd_idx = non_dominated_indices(F)
            
            # Select parents from non-dominated set using crowding distance
            if len(nd_idx) > 0:
                cd = crowding_distance(F, nd_idx)
                # Select two parents with high crowding distance
                parent_order = np.argsort(-cd)
                p1_idx = nd_idx[parent_order[0]]
                p2_idx = nd_idx[parent_order[min(1, len(parent_order)-1)]]
            else:
                p1_idx = np.random.randint(pop_size)
                p2_idx = np.random.randint(pop_size)

            # Crossover and mutation
            child = np.zeros(self.dim)
            for d in range(self.dim):
                if rng.random() < 0.5:
                    child[d] = X[p1_idx, d]
                else:
                    child[d] = X[p2_idx, d]
            
            # Gaussian mutation
            child += 0.05 * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replace worst individual
            # Use a simple heuristic: replace the individual that is dominated by the child
            replaced = False
            for i in range(pop_size):
                if dominates(fc, F[i]):
                    X[i] = child
                    F[i] = fc
                    replaced = True
                    break
            
            if not replaced:
                # Replace the one with worst sum of objectives
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = child
                F[worst] = fc

        return X
