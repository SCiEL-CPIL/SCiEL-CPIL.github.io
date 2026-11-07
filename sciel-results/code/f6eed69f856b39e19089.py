"""Improved multi-objective optimizer for DTLZ1 using NSGA-II style approach."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II style multi-objective evolutionary algorithm for DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        ndom = [0] * n
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    ndom[i] += 1
            if ndom[i] == 0:
                if not fronts:
                    fronts.append([])
                fronts[-1].append(i)
        
        k = 0
        while k < len(fronts):
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    ndom[j] -= 1
                    if ndom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        
        return fronts

    def _dominates(self, a, b):
        """Check if a dominates b (all objectives <=, at least one <)."""
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for given indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F_sub[:, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
            if f_range > 0:
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], m] - F_sub[sorted_idx[i-1], m]) / f_range
        
        return dist

    def _tournament_select(self, ranks, dist, pop_size):
        """Tournament selection based on rank and crowding distance."""
        idx1 = np.random.randint(pop_size)
        idx2 = np.random.randint(pop_size)
        
        if ranks[idx1] != ranks[idx2]:
            return idx1 if ranks[idx1] < ranks[idx2] else idx2
        else:
            return idx1 if dist[idx1] > dist[idx2] else idx2

    def _sbx_crossover(self, x1, x2, lo, hi, prob=0.9, eta=20.0):
        """Simulated binary crossover."""
        if np.random.rand() > prob:
            return x1.copy(), x2.copy()
        
        x1c, x2c = x1.copy(), x2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if x1[i] != x2[i]:
                    x_lo = min(x1[i], x2[i])
                    x_hi = max(x1[i], x2[i])
                    if x_hi - x_lo > 1e-14:
                        u = np.random.rand()
                        beta = 1.0 / (1.0 + 2.0 * (x_hi - x_lo) / (hi[i] - lo[i])) if False else 1.0
                        # Standard SBX
                        if u <= 0.5 / (1.0 + 2.0 * beta):
                            beta_q = (2.0 * u * (1.0 + beta)) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 * (1.0 - u) * (1.0 + beta))) ** (1.0 / (eta + 1.0))
                        
                        # Simplified: just use polynomial distribution
                        if u <= 0.5:
                            beta_q = (2.0 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                        
                        x1c[i] = 0.5 * ((1.0 + beta_q) * x1[i] + (1.0 - beta_q) * x2[i])
                        x2c[i] = 0.5 * ((1.0 - beta_q) * x1[i] + (1.0 + beta_q) * x2[i])
        
        x1c = np.clip(x1c, lo, hi)
        x2c = np.clip(x2c, lo, hi)
        return x1c, x2c

    def _mutation(self, x, lo, hi, prob=1.0/self.dim, eta=20.0):
        """Polynomial mutation."""
        x_new = x.copy()
        for i in range(self.dim):
            if np.random.rand() < prob:
                x_lo = (x[i] - lo[i]) / (hi[i] - lo[i])
                x_hi = (hi[i] - x[i]) / (hi[i] - lo[i])
                u = np.random.rand()
                
                if u < 0.5:
                    xy = 1.0 - x_hi
                    val = (2.0 * xy) ** (1.0 / (eta + 1.0))
                else:
                    xy = 1.0 - x_lo
                    val = 1.0 - (2.0 * xy) ** (1.0 / (eta + 1.0))
                
                delta = (val - 1.0) * (hi[i] - lo[i])
                x_new[i] = x[i] + delta
                x_new[i] = np.clip(x_new[i], lo[i], hi[i])
        
        return x_new

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = min(50, self.budget // 4)
        if pop_size < 10:
            pop_size = 10
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume
        all_X = X.copy()
        all_F = F.copy()
        
        # NSGA-II main loop
        while used < self.budget:
            # Evaluate how many we can do this generation
            remaining = self.budget - used
            n_offspring = min(2 * pop_size, remaining)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring // 2):
                # Selection
                p1 = self._tournament_select(
                    self._fast_non_dominated_sort(F), 
                    self._crowding_distance(F, list(range(pop_size))), 
                    pop_size
                )
                p2 = self._tournament_select(
                    self._fast_non_dominated_sort(F), 
                    self._crowding_distance(F, list(range(pop_size))), 
                    pop_size
                )
                
                # Crossover
                child1, child2 = self._sbx_crossover(X[p1], X[p2], lo, hi)
                
                # Mutation
                child1 = self._mutation(child1, lo, hi)
                child2 = self._mutation(child2, lo, hi)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                used += 2
                
                offspring_X.append(child1)
                offspring_F.append(f1)
                offspring_X.append(child2)
                offspring_F.append(f2)
                
                if used >= self.budget:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_X)
            
            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Assign ranks
            ranks = np.zeros(n_combined, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # Select next generation
            next_X = []
            next_F = []
            next_ranks = []
            
            for i, front in enumerate(fronts):
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_ranks.append(i)
                else:
                    # Need to select from this front using crowding distance
                    n_needed = pop_size - len(next_X)
                    dist = self._crowding_distance(combined_F, front)
                    sorted_idx = np.argsort(-dist)
                    for j in range(min(n_needed, len(sorted_idx))):
                        idx = front[sorted_idx[j]]
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                        next_ranks.append(i)
                    break
            
            if len(next_X) == 0:
                break
            
            X = np.array(next_X)
            F = np.array(next_F)
            
            # Update all evaluated points
            all_X = np.vstack([all_X, offspring_X])
            all_F = np.vstack([all_F, offspring_F])
        
        return X
# EVOLVE-BLOCK-END
