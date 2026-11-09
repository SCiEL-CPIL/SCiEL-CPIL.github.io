# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using Differential Evolution
    with Pareto front maintenance."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(0)

        def dominates(a, b):
            """Returns True if a dominates b (all objectives <= and at least one <)."""
            return np.all(a <= b) and np.any(a < b)

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # Dominated sets
            n_doms = [0] * n  # Number of dominators
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_doms[i] += 1
            
            # First front
            front = [i for i in range(n) if n_doms[i] == 0]
            if not front:
                return [list(range(n))]
            
            fronts.append(front)
            
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                front = next_front
            
            return fronts

        def crowding_distance(front_indices, F):
            """Calculate crowding distance for points in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            distances = [0.0] * n
            m = len(F[front_indices[0]])  # Number of objectives
            
            for k in range(m):
                # Sort by k-th objective
                sorted_indices = sorted(front_indices, key=lambda i: F[i][k])
                
                # Assign infinite distance to boundary points
                dist_arr = np.array(distances)
                dist_arr[sorted_indices[0]] = float('inf')
                dist_arr[sorted_indices[-1]] = float('inf')
                
                # Calculate range
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    dist_arr[idx] += (F[sorted_indices[i + 1]][k] - F[sorted_indices[i - 1]][k]) / f_range
            
            return list(dist_arr)

        # Initialize with stratified sampling
        n_init = 20
        X = []
        F = []
        used = 0

        # Use stratified sampling for better initial coverage
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = rng.uniform(lo, hi)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        # Main Optimization Loop: Differential Evolution with archive
        F_de = 0.8  # Differential weight
        CR = 1.0    # Crossover rate
        
        # Maintain an archive of non-dominated points
        archive = []  # List of (x, f) tuples
        
        # Add initial points to archive
        for i in range(len(X)):
            archive.append((X[i], F[i]))
        
        MAX_ARCHIVE = 100
        
        def hv_contribution(f_point, archive_F, ref_point):
            """Compute the hypervolume contribution of a point given the archive."""
            # A point contributes if it's not dominated and is better than ref in all dims
            if np.any(f_point >= ref_point):
                return 0.0
            # Check if dominated by any point in archive
            for f_arch in archive_F:
                if dominates(f_arch, f_point):
                    return 0.0
            # Approximate contribution: volume of the box from f_point to ref_point
            # that isn't already covered by other non-dominated points
            # Simplified: just use the box volume as a proxy
            box = np.prod(ref_point - f_point)
            return box

        def prune_archive(archive, ref_point):
            """Prune archive to MAX_ARCHIVE size using hypervolume contribution."""
            if len(archive) <= MAX_ARCHIVE:
                return archive
            
            archive_F = [f for _, f in archive]
            
            # Compute contributions
            contributions = []
            for i, (x, f) in enumerate(archive):
                # Remove this point from consideration when computing others
                other_F = [f_j for j, (_, f_j) in enumerate(archive) if j != i]
                contrib = hv_contribution(f, other_F, ref_point)
                contributions.append(contrib)
            
            # Keep points with highest contributions
            order = np.argsort(contributions)[::-1]
            kept = [archive[i] for i in order[:MAX_ARCHIVE]]
            return kept

        def update_archive(archive, x_new, f_new):
            """Update archive with new point, removing dominated points."""
            # Check if new point is dominated by any in archive
            for _, f_arch in archive:
                if dominates(f_arch, f_new):
                    return archive  # New point is dominated, don't add
            
            # Remove points dominated by new point
            new_archive = []
            for x_arch, f_arch in archive:
                if not dominates(f_new, f_arch):
                    new_archive.append((x_arch, f_arch))
            
            # Add new point
            new_archive.append((x_new, f_new))
            
            # Prune if needed
            ref_point = np.array([1.1, 1.3, 1.1])
            if len(new_archive) > MAX_ARCHIVE:
                new_archive = prune_archive(new_archive, ref_point)
            
            return new_archive

        while used < budget:
            if time.time() > deadline:
                break

            n_pop = len(X)
            if n_pop < 3:
                x = rng.uniform(lo, hi)
                x = np.clip(x, lo, hi)
                f = np.array(problem(x), dtype=float)
                X = np.vstack([X, x.reshape(1, -1)])
                F = np.vstack([F, f.reshape(1, -1)])
                used += 1
                continue

            # Use archive for parent selection
            if len(archive) >= 3:
                # Select 3 distinct parents from archive
                p1, p2, p3 = rng.choice(len(archive), size=3, replace=False)
                X_p1 = archive[p1][0]
                X_p2 = archive[p2][0]
                X_p3 = archive[p3][0]
            else:
                # Fallback to population
                p1, p2, p3 = rng.choice(n_pop, size=3, replace=False)
                X_p1 = X[p1]
                X_p2 = X[p2]
                X_p3 = X[p3]

            # Generate mutant with adaptive F
            F_adaptive = 0.5 + 0.3 * np.random.random()  # Vary F between 0.5 and 0.8
            v = X_p1 + F_adaptive * (X_p2 - X_p3)
            v = np.clip(v, lo, hi)

            # Evaluate
            f_v = np.array(problem(v), dtype=float)
            used += 1

            # Update archive
            archive = update_archive(archive, v, f_v)

            # Add to population
            X = np.vstack([X, v.reshape(1, -1)])
            F = np.vstack([F, f_v.reshape(1, -1)])

            # Periodically prune population to maintain diversity
            if len(X) > 50:
                fronts = fast_non_dominated_sort(F)
                if fronts:
                    # Keep first front (non-dominated) with crowding distance
                    nds_indices = fronts[0]
                    if len(nds_indices) > 30:
                        cd = crowding_distance(nds_indices, F)
                        # Keep top 30 by crowding distance
                        order = np.argsort(cd)[::-1][:30]
                        nds_indices = [nds_indices[i] for i in order]
                    X = X[nds_indices]
                    F = F[nds_indices]

        # Final polish: local search around archive points
        ref_point = np.array([1.1, 1.3, 1.1])
        polish_budget = min(20, budget - used)
        
        if polish_budget > 0 and len(archive) > 0:
            # Pick a few diverse points from archive for local refinement
            archive_F = np.array([f for _, f in archive])
            # Select points that are spread out
            if len(archive_F) > 5:
                # Use greedy selection for diversity
                selected = [0]
                remaining = list(range(1, len(archive_F)))
                for _ in range(min(4, len(archive_F) - 1)):
                    best_idx = -1
                    best_min_dist = -1
                    for idx in remaining:
                        min_dist = min(np.linalg.norm(archive_F[idx] - archive_F[s]) for s in selected)
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = idx
                    if best_idx >= 0:
                        selected.append(best_idx)
                        remaining.remove(best_idx)
            else:
                selected = list(range(len(archive_F)))
            
            for idx in selected:
                if used >= budget or time.time() > deadline:
                    break
                base_x = archive[idx][0].copy()
                # Try small perturbations
                for _ in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    step = 0.05 * np.random.random()
                    direction = rng.normal(0, 1, dim)
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    x_trial = base_x + step * direction
                    x_trial = np.clip(x_trial, lo, hi)
                    f_trial = np.array(problem(x_trial), dtype=float)
                    used += 1
                    archive = update_archive(archive, x_trial, f_trial)
                    # Update base if this is better (in terms of being non-dominated)
                    if not any(dominates(f, f_trial) for _, f in archive if f is not f_trial):
                        base_x = x_trial

        return X
# EVOLVE-BLOCK-END
