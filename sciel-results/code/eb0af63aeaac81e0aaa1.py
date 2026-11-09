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

    def _fast_non_dominated_sort(self, F):
        """Efficient non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=np.int32)
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
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

    def _generate_sobol(self, n, dim, lo, hi):
        """Generate n Sobol points mapped to [lo, hi]."""
        sobol = qmc.Sobol(d=dim, scramble=True)
        n_gen = 1
        while n_gen < n:
            n_gen *= 2
        X = sobol.random(n_gen)[:n]
        return lo + X * (hi - lo)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        
        def eval_batch(X_batch):
            """Evaluate a batch of points and add to global archive."""
            nonlocal used
            for x in X_batch:
                if time.time() > deadline - 10:
                    break
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1
            return len(all_F) - used + len(X_batch)  # not used, just for clarity

        # Phase 1: Sobol Initial Sampling (30% of budget)
        n_init = max(10, int(0.3 * self.budget))
        X_sobol = self._generate_sobol(n_init, self.dim, lo, hi)
        
        for x in X_sobol:
            if time.time() > deadline - 10:
                break
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
        
        all_X = np.array(all_X)
        all_F = np.array(all_F)
        used = len(all_F)
        
        # Phase 2: Batch DE with population update every generation
        if used < self.budget:
            pop_size = max(10, min(20, int(0.2 * self.budget)))
            
            # Initialize population from non-dominated points
            fronts = self._fast_non_dominated_sort(all_F)
            if fronts:
                nd_idx = fronts[0]
                if len(nd_idx) < pop_size:
                    others = [i for i in range(len(all_F)) if i not in nd_idx]
                    if others:
                        n_add = pop_size - len(nd_idx)
                        add_idx = np.random.choice(others, size=min(n_add, len(others)), replace=False)
                        nd_idx = np.concatenate([nd_idx, add_idx])
                np.random.shuffle(nd_idx)
                pop_X = all_X[nd_idx[:pop_size]].copy()
                pop_F = all_F[nd_idx[:pop_size]].copy()
            else:
                pop_X = all_X[:pop_size].copy()
                pop_F = all_F[:pop_size].copy()
            
            # Pad if needed
            if len(pop_X) < pop_size:
                n_pad = pop_size - len(pop_X)
                X_pad = lo + np.random.rand(n_pad, self.dim) * (hi - lo)
                F_pad = np.array([problem(x) for x in X_pad])
                used += n_pad
                all_X = np.vstack([all_X, X_pad])
                all_F = np.vstack([all_F, F_pad])
                pop_X = np.vstack([pop_X, X_pad])
                pop_F = np.vstack([pop_F, F_pad])
            
            F_DE = 0.8
            CR = 0.9
            batch_size = max(1, min(5, (self.budget - used) // 2))
            
            while used < self.budget and time.time() < deadline - 10:
                # Generate a batch of trial vectors
                n_batch = min(batch_size, self.budget - used)
                if n_batch <= 0:
                    break
                
                # For each trial, select parents and generate
                trials_X = []
                targets = []
                for _ in range(n_batch):
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    v = pop_X[r1] + F_DE * (pop_X[r2] - pop_X[r3])
                    v = np.clip(v, lo, hi)
                    
                    j_rand = np.random.randint(self.dim)
                    trial = pop_X[r1].copy()
                    for j in range(self.dim):
                        if np.random.rand() < CR or j == j_rand:
                            trial[j] = v[j]
                    trials_X.append(trial)
                    targets.append(r1)
                
                trials_X = np.array(trials_X)
                
                # Evaluate batch
                for i, trial in enumerate(trials_X):
                    if time.time() > deadline - 10:
                        break
                    f_trial = problem(trial)
                    used += 1
                    r1 = targets[i]
                    
                    # Selection: replace if dominates target
                    if self._dominates(f_trial, pop_F[r1]):
                        pop_X[r1] = trial
                        pop_F[r1] = f_trial
                    
                    all_X = np.vstack([all_X, trial])
                    all_F = np.vstack([all_F, f_trial])
                
                # Update population every 5 batches with non-dominated points
                if used % (5 * batch_size) == 0 and len(all_F) > pop_size:
                    fronts = self._fast_non_dominated_sort(all_F)
                    if fronts and len(fronts[0]) >= pop_size:
                        nd_idx = fronts[0]
                        indices = np.random.choice(nd_idx, pop_size, replace=False)
                        pop_X = all_X[indices].copy()
                        pop_F = all_F[indices].copy()
        
        # Phase 3: Local refinement around non-dominated points
        if used < self.budget and time.time() < deadline - 5:
            fronts = self._fast_non_dominated_sort(all_F)
            if fronts:
                nd_idx = fronts[0]
                # Take up to 10 best non-dominated points for refinement
                n_refine = min(10, len(nd_idx), self.budget - used)
                if n_refine > 0:
                    refine_idx = np.random.choice(nd_idx, n_refine, replace=False)
                    sigma = 0.05  # small perturbation scale
                    
                    for idx in refine_idx:
                        if used >= self.budget or time.time() > deadline - 5:
                            break
                        center = all_X[idx]
                        # Generate a few perturbations around this point
                        n_local = min(3, self.budget - used)
                        for _ in range(n_local):
                            if used >= self.budget or time.time() > deadline - 5:
                                break
                            perturbation = np.random.randn(self.dim) * sigma
                            x_local = np.clip(center + perturbation, lo, hi)
                            f_local = problem(x_local)
                            used += 1
                            all_X = np.vstack([all_X, x_local])
                            all_F = np.vstack([all_F, f_local])
        
        return all_X
# EVOLVE-BLOCK-END
