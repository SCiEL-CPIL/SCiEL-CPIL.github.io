import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with adaptive mutation and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        if not front:
            return [list(range(n))]
        
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_idx = sorted(front_indices, key=lambda i: F[i][m])
            cd[sorted_idx[0] - front_indices[0]] = np.inf
            cd[sorted_idx[-1] - front_indices[0]] = np.inf
            
            f_min = F[sorted_idx[0]][m]
            f_max = F[sorted_idx[-1]][m]
            f_range = f_max - f_min
            if f_range < 1e-12:
                continue
            
            for k in range(1, n - 1):
                cd[sorted_idx[k] - front_indices[0]] += (
                    F[sorted_idx[k + 1]][m] - F[sorted_idx[k - 1]][m]
                ) / f_range
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(123)
        
        # Population size
        P = 20
        n_generations = max(1, (budget - P) // P)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((P, dim))
        F = np.array([problem(x) for x in X])
        evals_used = P
        
        # Global archive: all non-dominated points seen
        archive_X = list(X)
        archive_F = list(F)
        
        def is_dominated(f, F_list):
            """Check if f is dominated by any point in F_list."""
            for g in F_list:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False
        
        def update_archive(x, f):
            """Add point to archive if not dominated, remove dominated points."""
            if is_dominated(f, archive_F):
                return
            # Remove points dominated by f
            new_X = [x]
            new_F = [f]
            for i, g in enumerate(archive_F):
                if not (np.all(f <= g) and np.any(f < g)):
                    new_X.append(archive_X[i])
                    new_F.append(g)
            archive_F[:] = new_F
            archive_X[:] = new_X
        
        # Main evolutionary loop
        gen = 0
        while evals_used < budget:
            if time.time() > deadline - 5.0:
                break
            
            gen += 1
            progress = gen / max(1, n_generations)
            
            # Adaptive sigma: start large, decrease over time
            sigma = 0.3 * (1.0 - 0.7 * progress) * (hi - lo)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(P, budget - evals_used)
            
            for _ in range(n_offspring):
                if evals_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # Select parent from archive using tournament
                if len(archive_X) <= 2:
                    parent_idx = int(rng.integers(len(archive_X)))
                else:
                    # Tournament selection: pick 2 random, take non-dominated one
                    i1 = int(rng.integers(len(archive_X)))
                    i2 = int(rng.integers(len(archive_X)))
                    f1, f2 = archive_F[i1], archive_F[i2]
                    if np.all(f1 <= f2) and np.any(f1 < f2):
                        parent_idx = i1
                    elif np.all(f2 <= f1) and np.any(f2 < f1):
                        parent_idx = i2
                    else:
                        parent_idx = i1 if rng.random() < 0.5 else i2
                
                parent = archive_X[parent_idx]
                
                # Mutation: Gaussian + occasional large jump
                if rng.random() < 0.1:
                    # Large jump: reinitialize some dimensions
                    child = parent.copy()
                    n_dims_to_jump = max(1, dim // 3)
                    dims_to_jump = rng.choice(dim, size=n_dims_to_jump, replace=False)
                    for d in dims_to_jump:
                        child[d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                else:
                    # Gaussian mutation
                    child = parent + sigma * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                
                fc = problem(child)
                evals_used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                update_archive(child, fc)
            
            if not offspring_X:
                break
            
            # Update population: combine parent + offspring, select best P
            # Use non-dominated sorting + crowding distance
            all_X = archive_X[:P] if len(archive_X) >= P else archive_X[:]
            all_F = archive_F[:P] if len(archive_F) >= P else archive_F[:]
            
            # Add offspring to consideration
            for x, f in zip(offspring_X, offspring_F):
                all_X.append(x)
                all_F.append(f)
            
            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(np.array(all_F))
            
            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= P:
                    for idx in front:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                else:
                    # Need crowding distance to select
                    cd = self._crowding_distance(np.array(all_F), front)
                    sorted_front = sorted(front, key=lambda i: cd[i - front[0]], reverse=True)
                    needed = P - len(new_X)
                    for idx in sorted_front[:needed]:
                        new_X.append(all_X[idx])
                        new_F.append(all_F[idx])
                    break
            
            # Replace population with selected
            # Keep archive as all non-dominated points
            # But for efficiency, limit archive size
            if len(archive_X) > 50:
                # Keep only non-dominated points
                nd_fronts = self._fast_non_dominated_sort(np.array(archive_F))
                if nd_fronts:
                    first_front = nd_fronts[0]
                    if len(first_front) > 30:
                        # Use crowding to trim
                        cd = self._crowding_distance(np.array(archive_F), first_front)
                        sorted_front = sorted(first_front, key=lambda i: cd[i - first_front[0]], reverse=True)
                        keep = sorted_front[:30]
                        archive_X = [archive_X[i] for i in keep]
                        archive_F = [archive_F[i] for i in keep]
                    else:
                        archive_X = [archive_X[i] for i in first_front]
                        archive_F = [archive_F[i] for i in first_front]
        
        # Local refinement of best points
        if evals_used < budget and time.time() < deadline - 10.0:
            # Sort archive by hypervolume contribution (approximate: by first objective)
            # Do local search on top few points
            top_k = min(5, len(archive_X))
            if top_k > 0:
                # Pick diverse top points
                candidates = archive_X[:top_k]
                for cx in candidates:
                    if evals_used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Local search: try small perturbations
                    local_sigma = 0.02 * (hi - lo)
                    best_x = cx
                    best_f = problem(cx)
                    evals_used += 1
                    if is_dominated(best_f, archive_F):
                        continue
                    
                    for _ in range(5):
                        if evals_used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        
                        trial = best_x + local_sigma * rng.standard_normal(dim)
                        trial = np.clip(trial, lo, hi)
                        tf = problem(trial)
                        evals_used += 1
                        
                        if not is_dominated(tf, archive_F):
                            best_x = trial
                            best_f = tf
                            update_archive(trial, tf)
        
        if archive_X:
            return np.array(archive_X)
        return np.array(X)
