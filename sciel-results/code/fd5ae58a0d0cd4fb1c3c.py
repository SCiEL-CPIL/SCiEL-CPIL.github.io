import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        
        # Use Halton sequence for space-filling initial design
        def halton(dim, n, skip=1):
            points = np.zeros((n, dim))
            for d in range(dim):
                # Prime numbers for each dimension
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                p = primes[d % len(primes)]
                for i in range(n):
                    x = 0.0
                    f = 1.0 / p
                    idx = i + skip
                    while idx > 0:
                        idx, rem = divmod(idx, p)
                        x += f * rem
                        f /= p
                    points[i, d] = x
            return points
        
        # Phase 1: Space-filling exploration with Halton + local refinement
        n_initial = min(100, self.budget // 3)
        
        # Generate Halton points
        halton_points = halton(dim, n_initial)
        X_initial = lo + (hi - lo) * halton_points
        F_initial = np.array([problem(x) for x in X_initial])
        used = n_initial
        
        if used >= self.budget or time.time() > deadline:
            return X_initial
        
        # Phase 2: Greedy coverage in objective space
        # Start with the best spread points from initial design
        # Use non-dominated sort to get front
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
            
            i = 0
            while True:
                i += 1
                found = False
                for a in range(n):
                    if rank[a] == i - 1:
                        for b in S[a]:
                            n_dom[b] -= 1
                            if n_dom[b] == 0:
                                rank[b] = i
                                found = True
                if not found:
                    break
            return rank
        
        # Keep track of all evaluated points
        all_X = [X_initial]
        all_F = [F_initial]
        
        # Select diverse seed points from initial front
        rank_init = non_dominated_sort(F_initial)
        front0_idx = np.where(rank_init == 0)[0]
        
        # Pick up to 10 diverse points from front0
        max_seeds = min(10, len(front0_idx))
        if max_seeds > 0:
            # Simple greedy selection for diversity
            selected = [front0_idx[0]]
            for _ in range(max_seeds - 1):
                best_idx = -1
                best_min_dist = -1
                for idx in front0_idx:
                    if idx in selected:
                        continue
                    # Min distance to already selected
                    min_dist = np.inf
                    for s in selected:
                        d = np.linalg.norm(F_initial[idx] - F_initial[s])
                        min_dist = min(min_dist, d)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_idx = idx
                if best_idx >= 0:
                    selected.append(best_idx)
            seed_indices = selected
        else:
            seed_indices = [0]
        
        # Build active set for greedy coverage
        active_X = np.array([X_initial[i] for i in seed_indices])
        active_F = np.array([F_initial[i] for i in seed_indices])
        
        # Phase 3: Greedy farthest-point sampling in objective space
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find the point in active_F that is farthest from all others in objective space
            # Then generate candidates around that region
            
            if len(active_F) < 2:
                # Just do random sampling
                n_candidates = min(10, self.budget - used)
                X_cand = lo + (hi - lo) * np.random.default_rng(used).random((n_candidates, dim))
            else:
                # Compute pairwise distances in objective space
                n_active = len(active_F)
                dist_matrix = np.zeros((n_active, n_active))
                for i in range(n_active):
                    for j in range(i + 1, n_active):
                        d = np.linalg.norm(active_F[i] - active_F[j])
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d
                
                # For each active point, find min distance to others
                min_dists = np.min(dist_matrix, axis=1)
                
                # Select the point with largest min distance (most isolated)
                target_idx = np.argmax(min_dists)
                target_point = active_X[target_idx]
                target_F = active_F[target_idx]
                
                # Generate candidates around target with decreasing radius
                n_candidates = min(5, self.budget - used)
                X_cand = np.zeros((n_candidates, dim))
                for i in range(n_candidates):
                    # Local perturbation
                    radius = 0.05 + 0.05 * i
                    direction = np.random.default_rng(used + i).standard_normal(dim)
                    direction = direction / (np.linalg.norm(direction) + 1e-12)
                    X_cand[i] = target_point + radius * (hi - lo) * direction
                    X_cand[i] = np.clip(X_cand[i], lo, hi)
            
            # Evaluate candidates
            for i in range(len(X_cand)):
                if used >= self.budget or time.time() > deadline:
                    break
                F_cand = problem(X_cand[i])
                used += 1
                all_X.append(X_cand[i].reshape(1, -1))
                all_F.append(F_cand.reshape(1, -1))
                
                # Add to active set if non-dominated or improves coverage
                # Simple check: if it's not dominated by any active point
                is_dominated = False
                for j in range(len(active_F)):
                    if np.all(active_F[j] <= F_cand) and np.any(active_F[j] < F_cand):
                        is_dominated = True
                        break
                if not is_dominated:
                    # Check if it dominates any active points, remove them
                    new_active_X = []
                    new_active_F = []
                    for j in range(len(active_F)):
                        if not (np.all(F_cand <= active_F[j]) and np.any(F_cand < active_F[j])):
                            new_active_X.append(active_X[j])
                            new_active_F.append(active_F[j])
                    active_X = np.array(new_active_X + [X_cand[i]])
                    active_F = np.array(new_active_F + [F_cand.reshape(1, -1)])
                else:
                    # Still add to active set for coverage diversity if it's far enough
                    if len(active_F) < 50:
                        active_X = np.vstack([active_X, X_cand[i]])
                        active_F = np.vstack([active_F, F_cand.reshape(1, -1)])
            
            if time.time() > deadline:
                break
        
        # Combine all evaluated points
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)
        
        return all_X
