import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # 1. LHS Initialization
        pop_size = 40
        lhs_points = np.zeros((pop_size, dim))
        for d in range(dim):
            bins = np.arange(pop_size) / pop_size
            offsets = rng.random(pop_size) / pop_size
            lhs_points[:, d] = bins + offsets
        for d in range(dim):
            idx = rng.permutation(pop_size)
            lhs_points[:, d] = lhs_points[idx, d]
            
        X = lo + (hi - lo) * lhs_points
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        archive = list(zip(X, F))
        archive = self._prune_archive(archive)
        
        # 2. DE Loop
        F_cr = 0.7
        F_weight = 0.8
        max_archive_size = 50
        
        while used < budget:
            if time.time() > deadline:
                break
            
            arch_X = np.array([a[0] for a in archive])
            arch_F = np.array([a[1] for a in archive])
            n_arch = len(arch_X)
            
            if n_arch < 3:
                break
                
            # Calculate Crowding Distance for selection
            cd = self._calc_crowding(arch_F)
            
            # Weighted selection based on CD
            # Normalize CD to sum to 1
            cd_sum = np.sum(cd)
            if cd_sum == 0:
                weights = np.ones(n_arch) / n_arch
            else:
                weights = cd / cd_sum
                
            # Select 3 parents
            idxs = rng.choice(n_arch, size=3, replace=False, p=weights)
            x_r1 = arch_X[idxs[0]]
            x_r2 = arch_X[idxs[1]]
            x_r3 = arch_X[idxs[2]]
            
            v = x_r1 + F_weight * (x_r2 - x_r3)
            v = np.clip(v, lo, hi)
            
            # Select target
            target_idx = rng.choice(n_arch, p=weights)
            target = arch_X[target_idx]
            
            jrand = rng.integers(dim)
            trial = target.copy()
            for j in range(dim):
                if rng.random() <= F_cr or j == jrand:
                    trial[j] = v[j]
            
            f_new = problem(trial)
            used += 1
            
            # Update Archive
            if self._is_dominated(f_new, arch_F):
                continue
                
            new_archive = []
            for ax, af in archive:
                if not (np.all(f_new <= af) and np.any(f_new < af)):
                    new_archive.append((ax, af))
            
            new_archive.append((trial, f_new))
            
            if len(new_archive) > max_archive_size:
                new_archive = self._prune_archive(new_archive)
            
            archive = new_archive
            
        # 3. Local Polish
        if used < budget and len(archive) > 0:
            arch_F = np.array([a[1] for a in archive])
            cd = self._calc_crowding(arch_F)
            # Pick top 2 by CD
            top_indices = np.argsort(cd)[-2:]
            
            for idx in top_indices:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x0 = np.array(archive[idx][0])
                f0 = np.array(archive[idx][1])
                
                # Simple local search
                for step_size in [0.1, 0.05]:
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    for _ in range(3): # 3 attempts per step size
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        delta = rng.standard_normal(dim) * step_size
                        x_new = np.clip(x0 + delta, lo, hi)
                        f_new = problem(x_new)
                        used += 1
                        
                        if self._is_dominated(f_new, arch_F):
                            continue
                            
                        # Check if f_new dominates f0
                        if np.all(f_new <= f0) and np.any(f_new < f0):
                            x0 = x_new
                            f0 = f_new
                            # Update archive: remove old, add new
                            # This is a bit messy, let's just add to archive and prune
                            archive = self._add_to_archive(archive, x0, f0)
                        elif not (np.all(f0 <= f_new) and np.any(f0 < f_new)):
                            # Non-dominated
                            archive = self._add_to_archive(archive, x_new, f_new)
                            
                        # Update arch_F for domination checks
                        arch_F = np.array([a[1] for a in archive])

        return archive

    def _is_dominated(self, f, archive_F):
        if len(archive_F) == 0:
            return False
        leq = np.all(archive_F <= f, axis=1)
        lt = np.any(archive_F < f, axis=1)
        return np.any(leq & lt)
        
    def _calc_crowding(self, F):
        n = len(F)
        if n == 0:
            return np.array([])
        cd = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            val_range = F[order[-1], m] - F[order[0], m]
            if val_range > 1e-12:
                for i in range(1, n-1):
                    cd[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / val_range
        return cd

    def _prune_archive(self, archive):
        if len(archive) <= 50:
            return archive
        F = np.array([a[1] for a in archive])
        cd = self._calc_crowding(F)
        order = np.argsort(cd)
        remove_count = len(archive) - 50
        keep_indices = set(order[remove_count:])
        return [a for i, a in enumerate(archive) if i in keep_indices]

    def _add_to_archive(self, archive, x, f):
        # Check if dominated
        arch_F = np.array([a[1] for a in archive])
        if self._is_dominated(f, arch_F):
            return archive
            
        # Remove dominated
        new_archive = []
        for ax, af in archive:
            if not (np.all(f <= af) and np.any(f < af)):
                new_archive.append((ax, af))
        new_archive.append((x, f))
        
        if len(new_archive) > 50:
            new_archive = self._prune_archive(new_archive)
            
        return new_archive
