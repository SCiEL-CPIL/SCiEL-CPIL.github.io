# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if self._dominates(F[p], F[q]):
                    S[p].append(q)
                elif self._dominates(F[q], F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        fronts = []
        if not front:
            front = [0]
        fronts.append(front)
        
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
        return fronts

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        bounds_arr = np.array(bounds)
        
        # Phase 1: Sobol Initial Sampling
        # Use 30% of budget for initial spread
        n_init = max(10, int(0.3 * self.budget))
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires power of 2, so we generate extra and slice
        n_gen = 1
        while n_gen < n_init:
            n_gen *= 2
        X_sobol = sobol.random(n_gen)[:n_init]
        
        # Map to bounds
        X_sobol = lo + X_sobol * (hi - lo)
        
        all_X = []
        all_F = []
        
        # Evaluate initial points
        for x in X_sobol:
            if time.time() > deadline - 5:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            
        all_X = np.array(all_X)
        all_F = np.array(all_F)
        used = len(all_F)
        
        # Phase 2: Differential Evolution
        if used < self.budget:
            # Population size for DE
            pop_size = max(10, min(20, self.budget - used))
            
            # Initialize DE population from best non-dominated points of Phase 1
            # If we have enough, take random sample from all_F to ensure diversity
            # Better: Take non-dominated points if available, else random
            fronts = self._non_dominated_sort(all_F)
            if fronts:
                nd_idx = fronts[0]
                # If nd_idx is too small, add random points
                if len(nd_idx) < pop_size:
                    others = [i for i in range(len(all_F)) if i not in nd_idx]
                    if others:
                        add_idx = np.random.choice(others, size=pop_size - len(nd_idx), replace=False)
                        nd_idx = np.concatenate([nd_idx, add_idx])
                # Shuffle to avoid bias
                np.random.shuffle(nd_idx)
                start_pop_X = all_X[nd_idx[:pop_size]]
                start_pop_F = all_F[nd_idx[:pop_size]]
            else:
                start_pop_X = all_X[:pop_size]
                start_pop_F = all_F[:pop_size]
            
            # Ensure we have exactly pop_size
            if len(start_pop_X) < pop_size:
                # Pad with random points
                n_pad = pop_size - len(start_pop_X)
                X_pad = lo + np.random.rand(n_pad, self.dim) * (hi - lo)
                F_pad = np.array([problem(x) for x in X_pad])
                used += n_pad
                all_X = np.vstack([all_X, X_pad])
                all_F = np.vstack([all_F, F_pad])
                start_pop_X = np.vstack([start_pop_X, X_pad])
                start_pop_F = np.vstack([start_pop_F, F_pad])
            
            F_DE = 0.8
            CR = 0.9
            
            while used < self.budget and time.time() < deadline - 5:
                # Generate one trial vector per generation (or batch?)
                # To save overhead, let's generate a batch of trial vectors
                # But budget is tight. Let's do 1 per loop iteration to be safe with time checks.
                
                # Select 3 distinct parents
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = start_pop_X[r1] + F_DE * (start_pop_X[r2] - start_pop_X[r3])
                v = np.clip(v, lo, hi)
                
                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = start_pop_X[r1].copy()
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Evaluate
                f_trial = problem(trial)
                used += 1
                
                # Selection: Keep if it dominates target or is non-dominated
                # Simple strategy: Replace if it dominates the target
                if self._dominates(f_trial, start_pop_F[r1]):
                    start_pop_X[r1] = trial
                    start_pop_F[r1] = f_trial
                
                # Add to global archive
                all_X = np.vstack([all_X, trial])
                all_F = np.vstack([all_F, f_trial])
                
                # Optional: Periodically prune global archive to keep only non-dominated
                # This helps if we want to sample from it, but we are sampling from start_pop_X
                # We can update start_pop_X with new non-dominated points occasionally
                if used % 20 == 0:
                     # Re-initialize DE population with best non-dominated from global archive
                     fronts = self._non_dominated_sort(all_F)
                     if fronts:
                         nd_idx = fronts[0]
                         if len(nd_idx) >= pop_size:
                             # Pick random pop_size from non-dominated
                             indices = np.random.choice(nd_idx, pop_size, replace=False)
                             start_pop_X = all_X[indices]
                             start_pop_F = all_F[indices]

        return all_X
# EVOLVE-BLOCK-END
