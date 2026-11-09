"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 combining evolutionary search with DTLZ4-specific exploitation."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size: balance between exploration and evaluation budget
        pop_size = min(30, max(15, self.budget // 6))

        # Initialize population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DTLZ4 specific: position variables (last n_obj-1) should be near 0 or 1
        # For DTLZ4 with 3 objectives, position vars are x[dim-(n_obj-1):]
        n_pos = self.n_obj - 1
        pos_start = self.dim - n_pos

        def fast_non_dominated_sort(F_mat):
            """Returns list of fronts, each containing indices."""
            n = len(F_mat)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F_mat[p] <= F_mat[q]) and np.any(F_mat[p] < F_mat[q]):
                        S[p].append(q)
                        n_dom[p] += 1
                    elif np.all(F_mat[q] <= F_mat[p]) and np.any(F_mat[q] < F_mat[p]):
                        n_dom[q] += 1
                
                if n_dom[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            
            if not fronts[-1]:
                fronts.pop()
            
            return fronts

        def crowding_distance(F_mat, front_indices):
            """Compute crowding distance for indices in a front."""
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            m = self.n_obj
            
            for k in range(m):
                sorted_idx = np.argsort(F_mat[front_indices, k])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F_mat[front_indices[sorted_idx[0]], k]
                f_max = F_mat[front_indices[sorted_idx[-1]], k]
                f_range = f_max - f_min
                
                if f_range > 1e-12:
                    for i in range(1, n - 1):
                        dist[sorted_idx[i]] += (F_mat[front_indices[sorted_idx[i+1]], k] - F_mat[front_indices[sorted_idx[i-1]], k]) / f_range
            
            return dist

        def get_non_dominated(F_mat):
            """Return boolean mask for non-dominated points."""
            n = len(F_mat)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        nd[i] = False
                        break
            return nd

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Find non-dominated points
            nd_mask = get_non_dominated(F)
            if np.any(nd_mask):
                parents = X[nd_mask]
            else:
                parents = X

            # Select a parent using tournament selection from non-dominated set
            if len(parents) > 1:
                # Simple tournament of size 2
                i, j = rng.integers(0, len(parents)), rng.integers(0, len(parents))
                # Prefer parent with lower sum of objectives
                parent = parents[i] if np.sum(parents[i]) <= np.sum(parents[j]) else parents[j]
            else:
                parent = parents[0]

            # Mutation: DTLZ4-specific
            child = parent.copy()
            
            # For position variables: push toward 0 or 1 with some noise
            for i in range(pos_start, self.dim):
                if rng.random() < 0.7:
                    # Set to extreme value
                    if rng.random() < 0.5:
                        child[i] = 0.0
                    else:
                        child[i] = 1.0
                else:
                    # Small perturbation
                    child[i] += 0.05 * range_[i] * rng.standard_normal()
                child[i] = np.clip(child[i], lo[i], hi[i])
            
            # For shape variables: small perturbation
            for i in range(pos_start):
                child[i] += 0.1 * range_[i] * rng.standard_normal()
                child[i] = np.clip(child[i], lo[i], hi[i])

            fc = problem(child)
            used += 1

            # Replace worst point using crowding-aware selection
            # Find the worst point in terms of dominance and crowding
            worst_idx = self._select_replacement_index(F, fc)
            X[worst_idx], F[worst_idx] = child, fc

        return X

    def _select_replacement_index(self, F, new_point):
        """Select index to replace based on dominance and crowding."""
        # First, check if new point dominates any existing points
        n = len(F)
        dominated_by_new = []
        for i in range(n):
            if np.all(new_point <= F[i]) and np.any(new_point < F[i]):
                dominated_by_new.append(i)
        
        if dominated_by_new:
            # Replace the worst dominated point (highest sum)
            return max(dominated_by_new, key=lambda i: np.sum(F[i]))
        
        # If new point doesn't dominate any, replace the worst overall point
        # Use sum of objectives as a simple indicator
        return int(np.argmax(np.sum(F, axis=1)))
# EVOLVE-BLOCK-END
