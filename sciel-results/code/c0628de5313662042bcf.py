"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Efficient NSGA-II for WFG7 with vectorized operations."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Vectorized non-dominated sort. Returns (ranks, fronts)."""
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        remaining = set(range(n))
        
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if j != i and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            fronts.append(front)
            for i in front:
                remaining.discard(i)
            for i in front:
                ranks[i] = len(fronts) - 1
            # Assign rank to newly added front members
            for i in front:
                ranks[i] = len(fronts) - 1
        
        return ranks, fronts

    def _crowding_distance(self, F, indices):
        """Crowding distance for a set of indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F_sub[order[-1], m] - F_sub[order[0], m]
            if f_range > 1e-10:
                for i in range(1, n - 1):
                    cd[order[i]] += (F_sub[order[i+1], m] - F_sub[order[i-1], m]) / f_range
        
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20.0):
        """Vectorized SBX crossover."""
        c1 = p1.copy()
        c2 = p2.copy()
        
        # For each dimension, decide whether to cross
        cross_mask = rng.random(self.dim) < 0.5
        cross_mask &= (p1 != p2)  # Only cross where parents differ
        
        if not np.any(cross_mask):
            return c1, c2
        
        u = rng.random(np.sum(cross_mask))
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1.0 / (eta + 1)),
                        (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1)))
        
        idx = np.where(cross_mask)[0]
        x1 = p1[idx]
        x2 = p2[idx]
        
        c1_vals = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2_vals = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        
        c1[idx] = c1_vals
        c2[idx] = c2_vals
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _poly_mutation(self, x, lo, hi, rng, eta=20.0, prob=0.1):
        """Polynomial mutation (vectorized)."""
        x_mut = x.copy()
        mut_mask = rng.random(self.dim) < prob
        if not np.any(mut_mask):
            return x_mut
        
        idx = np.where(mut_mask)[0]
        u = rng.random(len(idx))
        delta = np.where(u < 0.5,
                         (2 * u) ** (1.0 / (eta + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1.0 / (eta + 1)))
        
        x_mut[idx] += delta * (hi[idx] - lo[idx])
        x_mut = np.clip(x_mut, lo, hi)
        return x_mut

    def _stratified_init(self, N, lo, hi, rng):
        """Stratified initialization for better coverage."""
        X = np.zeros((N, self.dim))
        # Use Latin hypercube-like stratification
        for d in range(self.dim):
            strata = np.linspace(lo[d], hi[d], N + 1)
            offsets = (np.arange(N) + rng.random(N)) / N
            X[:, d] = strata[:-1] + offsets * (strata[1] - strata[0])
            X[:, d] = np.clip(X[:, d], lo[d], hi[d])
        return X

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Population size: use ~half budget for init, rest for evolution
        N = 30
        
        # Initialize
        X = self._stratified_init(N, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Cache ranks for tournament selection
        ranks, _ = self._fast_non_dominated_sort(F)
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Determine how many offspring to generate this generation
            n_off = min(N, self.budget - evals)
            if n_off <= 0:
                break
            
            off_X = []
            off_F = []
            
            for i in range(n_off):
                if evals >= self.budget or time.time() > deadline:
                    break
                
                # Binary tournament selection
                t1 = rng.integers(N)
                t2 = rng.integers(N)
                if ranks[t1] < ranks[t2]:
                    p1_idx = t1
                elif ranks[t2] < ranks[t1]:
                    p1_idx = t2
                else:
                    p1_idx = t1 if rng.random() < 0.5 else t2
                
                t3 = rng.integers(N)
                t4 = rng.integers(N)
                if ranks[t3] < ranks[t4]:
                    p2_idx = t3
                elif ranks[t4] < ranks[t3]:
                    p2_idx = t4
                else:
                    p2_idx = t3 if rng.random() < 0.5 else t4
                
                # SBX crossover
                c1, c2 = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi, rng)
                
                # Polynomial mutation
                c1 = self._poly_mutation(c1, lo, hi, rng)
                c2 = self._poly_mutation(c2, lo, hi, rng)
                
                off_X.append(c1)
                off_F.append(problem(c1))
                evals += 1
                
                # If we have budget, also evaluate second child
                if evals < self.budget and time.time() <= deadline:
                    off_X.append(c2)
                    off_F.append(problem(c2))
                    evals += 1
            
            if not off_X:
                break
            
            off_X = np.array(off_X)
            off_F = np.array(off_F)
            
            # Combine parent and offspring
            X_comb = np.vstack([X, off_X])
            F_comb = np.vstack([F, off_F])
            
            # Non-dominated sort on combined population
            ranks_comb, fronts = self._fast_non_dominated_sort(F_comb)
            
            # Select next generation using non-dominated sorting + crowding
            X_next = []
            F_next = []
            
            for front in fronts:
                if len(X_next) + len(front) <= N:
                    for idx in front:
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                else:
                    remaining_slots = N - len(X_next)
                    cd = self._crowding_distance(F_comb, front)
                    sorted_front = np.argsort(-cd)
                    for i in sorted_front[:remaining_slots]:
                        idx = front[i]
                        X_next.append(X_comb[idx])
                        F_next.append(F_comb[idx])
                    break
            
            if len(X_next) < N:
                for i in range(len(X_next), N):
                    X_next.append(X_comb[i % len(X_comb)])
                    F_next.append(F_comb[i % len(F_comb)])
            
            X = np.array(X_next[:N])
            F = np.array(F_next[:N])
            ranks, _ = self._fast_non_dominated_sort(F)
        
        return X
# EVOLVE-BLOCK-END
