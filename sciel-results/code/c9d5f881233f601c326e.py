"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 using a hybrid approach:
    - LHS initialization for coverage
    - NSGA-II style evolution with tournament selection
    - Local refinement via Nelder-Mead on hypervolume contribution
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(123)

        # Phase 1: LHS initialization
        n_init = min(40, self.budget // 5)
        n_init = max(n_init, 10)
        X = self._lhs_sample(n_init, self.dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = n_init

        # Phase 2: NSGA-II evolution
        pop_size = min(60, self.budget // 3)
        pop_size = max(pop_size, 20)
        
        # Resize population if needed
        if len(X) < pop_size:
            extra = pop_size - len(X)
            X_extra = lo + (hi - lo) * rng.random((extra, self.dim))
            X = np.vstack([X, X_extra])
            F = np.vstack([F, np.array([problem(x) for x in X_extra])])
            used += extra
        elif len(X) > pop_size:
            X = X[:pop_size]
            F = F[:pop_size]

        # Evolution loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)
            
            # Selection: tournament on fronts
            parents = self._tournament_select(fronts, F, X, 2, rng)
            
            # Crossover and mutation
            child = self._crossover(parents, lo, hi, rng)
            child = self._mutate(child, lo, hi, rng, 0.1)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst individual
            worst_idx = self._find_worst(F, fronts)
            X[worst_idx] = child
            F[worst_idx] = fc

            # Periodic local refinement
            if used % 20 == 0 and time.time() < deadline - 30:
                X, F, used = self._local_refine(X, F, problem, lo, hi, budget_remaining=self.budget - used, rng=rng, deadline=deadline)

        return X

    def _lhs_sample(self, n, dim, lo, hi, rng):
        """Latin Hypercube Sampling."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return lo + (hi - lo) * samples

    def _non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # Dominated sets
        nd = [0] * n  # Dominance count
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    nd[i] += 1
        
        current_front = [i for i in range(n) if nd[i] == 0]
        fronts.append(current_front)
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts

    def _dominates(self, a, b):
        """Check if a dominates b (a <= b in all objectives, a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _tournament_select(self, fronts, F, X, k, rng):
        """Tournament selection based on front rank and crowding."""
        n = len(F)
        # Assign ranks
        rank = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        # Crowding distance
        cd = self._crowding_distance(F, fronts)
        
        # Tournament
        selected = []
        for _ in range(2):
            candidates = rng.integers(n, size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
                    best = c
            selected.append(best)
        
        return np.array([X[selected[0]], X[selected[1]]])

    def _crowding_distance(self, F, fronts):
        """Calculate crowding distance for each individual."""
        n = len(F)
        cd = np.zeros(n)
        
        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    cd[idx] = np.inf
                continue
            
            for m in range(self.n_obj):
                fvals = F[front, m]
                min_val = np.min(fvals)
                max_val = np.max(fvals)
                if max_val - min_val < 1e-10:
                    for idx in front:
                        cd[idx] = np.inf
                    continue
                
                # Sort by objective m
                sorted_idx = np.array(front)[np.argsort(fvals)]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                for i in range(1, len(sorted_idx) - 1):
                    cd[sorted_idx[i]] += (fvals[sorted_idx[i+1]] - fvals[sorted_idx[i-1]]) / (max_val - min_val)
        
        return cd

    def _crossover(self, parents, lo, hi, rng):
        """BLX-alpha crossover."""
        a, b = parents[0], parents[1]
        child = np.zeros(self.dim)
        for d in range(self.dim):
            lo_c = min(a[d], b[d])
            hi_c = max(a[d], b[d])
            alpha = 0.5
            child[d] = rng.uniform(lo_c - alpha * (hi_c - lo_c), hi_c + alpha * (hi_c - lo_c))
        return np.clip(child, lo, hi)

    def _mutate(self, x, lo, hi, rng, rate):
        """Gaussian mutation."""
        mask = rng.random(self.dim) < rate
        x = x.copy()
        x[mask] += 0.1 * (hi[mask] - lo[mask]) * rng.standard_normal(np.sum(mask))
        return np.clip(x, lo, hi)

    def _find_worst(self, F, fronts):
        """Find the worst individual to replace (last front, lowest crowding)."""
        if not fronts:
            return len(F) - 1
        last_front = fronts[-1]
        if not last_front:
            return len(F) - 1
        # Use sum of objectives as tiebreaker
        worst_in_front = max(last_front, key=lambda i: np.sum(F[i]))
        return worst_in_front

    def _local_refine(self, X, F, problem, lo, hi, budget_remaining, rng, deadline):
        """Local refinement using Nelder-Mead on top candidates."""
        if budget_remaining < 10:
            return X, F, 0

        # Select top 3 non-dominated points for refinement
        fronts = self._non_dominated_sort(F)
        if not fronts or not fronts[0]:
            return X, F, 0

        top_idx = fronts[0][:3]
        used = 0

        for idx in top_idx:
            if used >= budget_remaining or time.time() > deadline - 10:
                break

            # Try small perturbations
            best_x = X[idx].copy()
            best_f = F[idx].copy()
            
            for _ in range(3):
                if used >= budget_remaining or time.time() > deadline - 10:
                    break
                pert = best_x + 0.02 * (hi - lo) * rng.standard_normal(self.dim)
                pert = np.clip(pert, lo, hi)
                fp = problem(pert)
                used += 1
                if np.all(fp <= best_f) and np.any(fp < best_f):
                    best_x = pert
                    best_f = fp

            X[idx] = best_x
            F[idx] = best_f

        return X, F, used
# EVOLVE-BLOCK-END
