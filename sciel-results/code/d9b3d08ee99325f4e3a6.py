import numpy as np
import time

class Optimizer:
    """DE-based multi-objective optimizer with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310  # Safety margin before 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_X = []
        all_F = []
        eval_count = 0

        def eval_point(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            eval_count += 1
            all_X.append(x.copy())
            all_F.append(f)
            return f

        def dominates(a, b):
            """Check if a dominates b (a is better in all objectives, at least one strictly)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominating = [0] * n  # n[i] = number of solutions dominating i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_dominating[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_dominating[i] += 1
            
            current_front = [i for i in range(n) if n_dominating[i] == 0]
            if not current_front:
                return [list(range(n))]
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominating[j] -= 1
                        if n_dominating[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            # If any points remain unassigned (shouldn't happen), add them
            assigned = set()
            for front in fronts:
                assigned.update(front)
            if len(assigned) < n:
                fronts.append([i for i in range(n) if i not in assigned])
            
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            F_sub = F[indices]
            
            for m in range(n_obj):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / f_range
            
            return dist

        # Population size
        pop_size = min(50, max(20, budget // 4))
        
        # Initialize population
        X = lo + span * rng.random((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            F[i] = eval_point(X[i])
        
        # DE parameters
        CR = 0.9  # Crossover rate
        F_factor = 0.8  # Differential weight
        
        # Main loop
        while eval_count < budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Build mating pool: take from front 0, then front 1, etc.
            mating_pool = []
            for front in fronts:
                if len(mating_pool) + len(front) >= pop_size:
                    # Need to select from this front using crowding distance
                    cd = crowding_distance(F, front)
                    sorted_by_cd = np.argsort(cd)[::-1]  # Descending crowding distance
                    needed = pop_size - len(mating_pool)
                    mating_pool.extend([front[i] for i in sorted_by_cd[:needed]])
                    break
                else:
                    mating_pool.extend(front)
            
            # Generate offspring
            new_X = np.zeros((pop_size, dim))
            new_F = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select 3 distinct random parents from mating pool
                if len(mating_pool) < 3:
                    # Fallback: use random individuals
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                else:
                    r1, r2, r3 = rng.choice(len(mating_pool), 3, replace=False)
                    r1, r2, r3 = mating_pool[r1], mating_pool[r2], mating_pool[r3]
                
                # DE mutation
                v = X[r1] + F_factor * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)
                
                # Binary crossover
                jrand = rng.integers(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                trial = np.clip(trial, lo, hi)
                
                new_X[i] = trial
                new_F[i] = eval_point(trial)
            
            # Selection: combine population and offspring, select best pop_size
            X_combined = np.vstack([X, new_X])
            F_combined = np.vstack([F, new_F])
            
            fronts = non_dominated_sort(F_combined)
            
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) >= pop_size:
                    cd = crowding_distance(F_combined, front)
                    sorted_by_cd = np.argsort(cd)[::-1]
                    needed = pop_size - len(selected_indices)
                    selected_indices.extend([front[i] for i in sorted_by_cd[:needed]])
                    break
                else:
                    selected_indices.extend(front)
            
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
        
        return np.array(all_F)
