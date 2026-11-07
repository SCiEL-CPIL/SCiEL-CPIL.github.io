"""Multi-objective optimizer for RE37 using NSGA-II with local refinement."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, ...
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of solutions that dominate i
        rank = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dom[i] == 0]
        for i in current_front:
            rank[i] = 0
        
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        rank[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            k += 1
        
        # Add any remaining solutions (shouldn't happen but just in case)
        for i in range(n):
            if rank[i] == 0 and i not in fronts[0]:
                if not fronts:
                    fronts.append([])
                fronts[-1].append(i)
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        F_front = F[front_indices]
        
        for m in range(self.n_obj):
            obj = F_front[:, m]
            sorted_idx = np.argsort(obj)
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            obj_range = obj[sorted_idx[-1]] - obj[sorted_idx[0]]
            if obj_range > 1e-15:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj[sorted_idx[i + 1]] - obj[sorted_idx[i - 1]]) / obj_range
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n_budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # Phase 1: Halton sequence initialization
        n_init = min(60, n_budget)
        halton_pts = self._halton(n_init, dim)
        X_pop = lo + halton_pts * span
        F_pop = np.array([problem(x) for x in X_pop])
        used = n_init

        if used >= n_budget:
            return X_pop

        # NSGA-II parameters
        pop_size = 50
        n_gen = max(1, (n_budget - used) // pop_size)
        
        # Ensure we have at least pop_size individuals
        if len(X_pop) < pop_size:
            extra = pop_size - len(X_pop)
            X_extra = lo + span * np.random.random((extra, dim))
            F_extra = np.array([problem(x) for x in X_extra])
            X_pop = np.vstack([X_pop, X_extra])
            F_pop = np.vstack([F_pop, F_extra])
            used += extra

        # Truncate to pop_size if needed
        if len(X_pop) > pop_size:
            X_pop = X_pop[:pop_size]
            F_pop = F_pop[:pop_size]

        # Main NSGA-II loop
        while used < n_budget and time.time() < deadline:
            # Generate offspring
            n_offspring = min(pop_size, n_budget - used)
            if n_offspring <= 0:
                break
            
            X_offspring = np.zeros((n_offspring, dim))
            F_offspring = np.zeros((n_offspring, n_obj))
            
            for i in range(n_offspring):
                if time.time() > deadline:
                    break
                # Select parents using tournament selection
                p1 = self._tournament_select(F_pop, k=3)
                p2 = self._tournament_select(F_pop, k=3)
                
                # Crossover (SBX-like)
                child = self._crossover(p1, p2, lo, hi, span)
                
                # Mutation
                if np.random.random() < 0.8:
                    child = self._mutate(child, lo, hi, span)
                
                X_offspring[i] = child
                F_offspring[i] = problem(child)
                used += 1
            
            if used >= n_budget:
                # Combine and select
                X_combined = np.vstack([X_pop, X_offspring[:n_offspring]])
                F_combined = np.vstack([F_pop, F_offspring[:n_offspring]])
                X_pop, F_pop = self._select_next_gen(X_combined, F_combined, pop_size)
                break
            
            # Combine parents and offspring
            X_combined = np.vstack([X_pop, X_offspring[:n_offspring]])
            F_combined = np.vstack([F_pop, F_offspring[:n_offspring]])
            
            # Select next generation
            X_pop, F_pop = self._select_next_gen(X_combined, F_combined, pop_size)
        
        # Phase 3: Local refinement around Pareto front
        if time.time() < deadline - 5:
            # Get non-dominated solutions
            fronts = self._fast_non_dominated_sort(F_pop)
            if fronts:
                pareto_idx = fronts[0]
                n_refine = min(n_budget - used, 30)
                for idx in pareto_idx:
                    if n_refine <= 0 or time.time() > deadline:
                        break
                    x_best = X_pop[idx]
                    # Try small perturbations
                    for trial in range(min(5, n_refine)):
                        perturbation = 0.05 * span * np.random.randn(dim)
                        x_trial = np.clip(x_best + perturbation, lo, hi)
                        f_trial = problem(x_trial)
                        n_refine -= 1
                        used += 1
                        # Add to population if it improves
                        X_pop = np.vstack([X_pop, x_trial])
                        F_pop = np.vstack([F_pop, f_trial])
        
        return X_pop

    def _tournament_select(self, F, k=3):
        """Tournament selection. Returns index of winner."""
        n = len(F)
        candidates = np.random.choice(n, size=min(k, n), replace=False)
        # Find non-dominated among candidates
        best = candidates[0]
        for c in candidates[1:]:
            if np.all(F[c] <= F[best]) and np.any(F[c] < F[best]):
                best = c
            elif not (np.all(F[best] <= F[c]) and np.any(F[best] < F[c])):
                # If neither dominates, pick random
                if np.random.random() < 0.5:
                    best = c
        return best

    def _crossover(self, p1, p2, lo, hi, span):
        """Simple blend crossover."""
        alpha = 1.0  # SBX parameter
        r = np.random.random(dim)
        child = np.zeros(dim)
        for d in range(dim):
            if r[d] < 0.5:
                child[d] = (1 + 1/alpha) * p1[d] + (1 - 1/alpha) * p2[d]
            else:
                child[d] = (1 + 1/alpha) * p2[d] + (1 - 1/alpha) * p1[d]
        return np.clip(child, lo, hi)

    def _mutate(self, x, lo, hi, span):
        """Polynomial mutation."""
        eta = 20.0
        x_mut = x.copy()
        for d in range(dim):
            if np.random.random() < 0.5:
                u = np.random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                x_mut[d] = x[d] + delta * (hi[d] - lo[d])
        return np.clip(x_mut, lo, hi)

    def _select_next_gen(self, X, F, pop_size):
        """NSGA-II selection: non-dominated sorting + crowding distance."""
        n = len(X)
        if n <= pop_size:
            return X, F
        
        fronts = self._fast_non_dominated_sort(F)
        
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                # Need to pick some from this front based on crowding distance
                cd = self._crowding_distance(F, front)
                sorted_by_cd = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                needed = pop_size - len(selected)
                for i in sorted_by_cd[:needed]:
                    selected.append(front[i])
                break
        
        X_sel = X[selected]
        F_sel = F[selected]
        
        return X_sel, F_sel
