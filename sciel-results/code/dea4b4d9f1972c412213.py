"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominance(self, a, b):
        """Return True if a dominates b (a <= b in all objectives and a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominance(F[j], F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of points in objective space."""
        n = len(F)
        if n <= 2:
            return np.ones(n)
        
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F[:, m].max() - F[:, m].min()
            if f_range < 1e-10:
                f_range = 1.0
            for i in range(1, n - 1):
                cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / f_range
        
        return cd

    def _tournament(self, population, crowding, size=2):
        """Tournament selection based on crowding distance."""
        candidates = np.random.choice(len(population), size=size, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if crowding[c] > crowding[best]:
                best = c
        return best

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        pop_size = min(80, max(20, budget // 3))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        CR = 0.9  # Crossover rate
        sigma = 0.5  # Mutation std dev
        
        while used < budget:
            # Find non-dominated front
            nondom_idx = self._non_dominated_indices(F)
            
            # Compute crowding distances for the whole population
            cd = self._crowding_distance(F)
            
            # Select parents using tournament selection
            p1 = self._tournament(X, cd)
            p2 = self._tournament(X, cd)
            
            # Create trial vector using SBX-like crossover + Gaussian mutation
            y = X[p1].copy()
            for j in range(dim):
                if rng.random() < CR:
                    y[j] = (X[p1, j] + X[p2, j]) / 2.0
                    y[j] += rng.normal(0, sigma * (hi[j] - lo[j]) / 10.0)
            
            # Clip to bounds
            y = np.clip(y, lo, hi)
            
            # Evaluate
            fy = problem(y)
            used += 1
            
            # Check if y is non-dominated with respect to current population
            is_y_dominated = False
            for i in range(pop_size):
                if self._dominance(F[i], fy):
                    is_y_dominated = True
                    break
            
            if not is_y_dominated:
                # y is non-dominated, find a point to replace
                # Replace the point with the largest crowding distance that is dominated by y
                replace_idx = -1
                max_cd = -1
                for i in range(pop_size):
                    if self._dominance(fy, F[i]):
                        # y dominates i
                        if cd[i] > max_cd:
                            max_cd = cd[i]
                            replace_idx = i
                
                if replace_idx == -1:
                    # No point dominated by y, replace the point with the largest crowding distance
                    replace_idx = np.argmax(cd)
                
                X[replace_idx] = y
                F[replace_idx] = fy
        
        return X
# EVOLVE-BLOCK-END
