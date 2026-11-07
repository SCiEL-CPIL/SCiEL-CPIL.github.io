import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        # 1. Initial Sampling: Latin Hypercube for better coverage
        # Use a larger initial set to ensure good spread
        n_init = min(50, self.budget)
        X_init = np.empty((n_init, dim))
        for i in range(dim):
            # Stratified sampling
            X_init[:, i] = lo[i] + span[i] * (np.random.rand(n_init) + np.arange(n_init)) / n_init
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Helper: Check dominance
        def dominates(a, b):
            # a dominates b if a <= b and a != b
            return np.all(a <= b) and np.any(a < b)
        
        # Helper: Get non-dominated set from F (N x n_obj)
        def get_non_dominated(F):
            N = len(F)
            if N == 0:
                return np.array([], dtype=int)
            is_dominated = np.zeros(N, dtype=bool)
            for i in range(N):
                if is_dominated[i]:
                    continue
                for j in range(N):
                    if i == j or is_dominated[j]:
                        continue
                    if dominates(F[i], F[j]):
                        is_dominated[j] = True
            return np.where(~is_dominated)[0]

        # 2. Setup Archive and Population
        # We will maintain an archive of non-dominated points.
        # The population is used for the evolutionary search.
        
        # Initialize archive with non-dominated points from initial sampling
        nd_idx = get_non_dominated(F_init)
        
        archive_X = [X_init[i].copy() for i in nd_idx]
        archive_F = [F_init[i].copy() for i in nd_idx]
        
        # Population size for DE
        pop_size = 20
        if used + pop_size > self.budget:
            pop_size = max(5, self.budget - used)
            
        # Initialize population from random points (mix of archive and random)
        X_pop = np.empty((pop_size, dim))
        F_pop = np.empty((pop_size, n_obj))
        
        # Fill population
        for i in range(pop_size):
            if i < len(archive_X):
                X_pop[i] = archive_X[i]
                F_pop[i] = archive_F[i]
            else:
                x = np.random.uniform(lo, hi)
                X_pop[i] = x
                F_pop[i] = problem(x)
                used += 1
                # Check if this new point is non-dominated and add to archive
                # (Simplified: we'll update archive at the end of each generation or periodically)
                # For now, just keep it in pop. We'll do a full archive update periodically.
                
        # To be safe and efficient, let's re-evaluate the initial non-dominated set 
        # and ensure pop is diverse.
        # Actually, the above loop might use up budget. 
        # Let's stick to the initial 50 points for archive and pop initialization 
        # without extra evaluations if possible.
        
        # Reset pop to be a subset of initial points to avoid extra evals at start
        # We already have 50 points. Let's pick 20 diverse ones.
        # For simplicity, just use the first 20 non-dominated or random from initial.
        if len(nd_idx) >= pop_size:
            X_pop = X_init[nd_idx[:pop_size]].copy()
            F_pop = F_init[nd_idx[:pop_size]].copy()
        else:
            # Fill remaining with random from initial set
            remaining = pop_size - len(nd_idx)
            all_idx = np.arange(len(X_init))
            nd_set = set(nd_idx.tolist())
            available = [i for i in all_idx if i not in nd_set]
            if len(available) >= remaining:
                fill_idx = np.random.choice(available, remaining, replace=False)
            else:
                fill_idx = np.random.choice(len(X_init), remaining, replace=True)
            X_pop[len(nd_idx):] = X_init[fill_idx]
            F_pop[len(nd_idx):] = F_init[fill_idx]

        # 3. Evolutionary Loop (DE/rand/1/bin with NSGA-II-like selection)
        F_mut = 0.8
        CR = 0.7
        
        # We will update the archive every 10 evaluations or when a point is dominated
        evals_since_archive_update = 0
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate one offspring
            i = np.random.randint(pop_size)
            
            # Select three distinct random indices
            candidates = [j for j in range(pop_size) if j != i]
            if len(candidates) < 3:
                r1, r2, r3 = np.random.choice(pop_size, size=3, replace=True)
            else:
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
            
            # Mutation
            v = X_pop[r1] + F_mut * (X_pop[r2] - X_pop[r3])
            
            # Crossover
            j_rand = np.random.randint(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if np.random.random() <= CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X_pop[i][j]
            
            # Clip
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            evals_since_archive_update += 1
            
            # Selection: Compare trial with X_pop[i]
            # If trial dominates i, replace
            # If i dominates trial, discard
            # If non-dominated, use crowding distance or random
            
            if dominates(f_trial, F_pop[i]):
                X_pop[i] = trial
                F_pop[i] = f_trial
                # Update archive: remove points dominated by trial, add trial if non-dominated
                self._update_archive(archive_X, archive_F, trial, f_trial)
            elif not dominates(F_pop[i], f_trial):
                # Non-dominated
                # Calculate crowding distance for i and trial within the current population
                # To keep it fast, we use a simple heuristic or calculate exact for the pair
                # Let's calculate exact crowding distance for the subset {i, trial} relative to pop?
                # No, crowding distance is usually calculated on the non-dominated front.
                # Here we have a mixed population.
                # Simple heuristic: Replace with probability 0.5, or based on distance to archive.
                
                # Better heuristic: Check if trial is closer to the "ideal" (min of mins)
                # or if it improves the spread.
                # Let's just replace with probability 0.5 to maintain diversity.
                if np.random.random() < 0.5:
                    X_pop[i] = trial
                    F_pop[i] = f_trial
                    self._update_archive(archive_X, archive_F, trial, f_trial)
            else:
                # i dominates trial, do nothing
                pass

            # Periodically update archive with all population points to ensure it's clean
            if evals_since_archive_update >= 10:
                # Rebuild archive from population + current archive
                # This is expensive if done often, but 10 evals is okay for pop_size 20
                # Actually, just adding new points is safer.
                # Let's just ensure the archive contains the current best known non-dominated points.
                # We'll do a full rebuild every 50 evals.
                if used % 50 == 0:
                    all_X = np.vstack([np.array(archive_X), X_pop])
                    all_F = np.vstack([np.array(archive_F), F_pop])
                    nd_idx = get_non_dominated(all_F)
                    archive_X = [all_X[i].copy() for i in nd_idx]
                    archive_F = [all_F[i].copy() for i in nd_idx]
                evals_since_archive_update = 0

        return X_pop

    def _update_archive(self, archive_X, archive_F, new_x, new_f):
        """
        Update the archive with a new point.
        1. Remove points dominated by new_x.
        2. If new_x is not dominated by any remaining point, add it.
        """
        new_archive_X = []
        new_archive_F = []
        is_dominated = False
        
        for i in range(len(archive_X)):
            if dominates(new_f, archive_F[i]):
                # new dominates archive[i], so archive[i] is removed
                continue
            elif dominates(archive_F[i], new_f):
                # archive[i] dominates new, so new is not added
                is_dominated = True
                new_archive_X.append(archive_X[i])
                new_archive_F.append(archive_F[i])
            else:
                # Non-dominated
                new_archive_X.append(archive_X[i])
                new_archive_F.append(archive_F[i])
        
        if not is_dominated:
            new_archive_X.append(new_x)
            new_archive_F.append(new_f)
            
        # Update in place (lists are mutable, but we need to replace the contents)
        # Since we pass lists, we can clear and extend
        archive_X.clear()
        archive_F.clear()
        archive_X.extend(new_archive_X)
        archive_F.extend(new_archive_F)
