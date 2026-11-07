import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        archive = []  # List of (x, f) tuples for non-dominated points
        all_evals = [] # List of all evaluated (x, f) to avoid re-evaluation if needed, though grid/DE shouldn't repeat exactly

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return [[]]
            S = [set() for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].add(j)
                        n_d[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].add(i)
                        n_d[i] += 1
            
            for i in range(n):
                if n_d[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            fronts.pop()
            return fronts

        def crowding_distance(F):
            n = len(F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                idx = np.argsort(F[:, m])
                cd[idx[0]] = np.inf
                cd[idx[-1]] = np.inf
                range_m = F[idx[-1], m] - F[idx[0], m]
                if range_m == 0:
                    continue
                for i in range(1, n - 1):
                    cd[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / range_m
            return cd

        def update_archive(x, f):
            nonlocal archive
            is_dominated = False
            new_archive = []
            for ax, af in archive:
                if dominates(af, f):
                    is_dominated = True
                    new_archive.append((ax, af))
                    break
            if is_dominated:
                archive = new_archive
                return False
            
            # Remove points in archive dominated by new point
            for ax, af in archive:
                if not dominates(f, af):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive = new_archive
            return True

        # Phase 1: Grid Search
        # 3^4 = 81 points
        grid_size = 3
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    for l in range(grid_size):
                        x = np.array([i, j, k, l], dtype=float) / (grid_size - 1)
                        x = lo + x * (hi - lo)
                        grid_points.append(x)
        
        # Shuffle to make it less deterministic in order, though grid is fixed
        np.random.shuffle(grid_points)

        for x in grid_points:
            if used >= budget or time.time() > deadline:
                break
            f = problem(x)
            f = np.array(f)
            all_evals.append((x, f))
            used += 1
            update_archive(x, f)

        # Phase 2: DE Refinement
        if used < budget and time.time() < deadline:
            # Initialize DE population
            pop_size = 20
            # Use archive points to seed population
            if len(archive) == 0:
                # Fallback to random
                X = np.random.rand(pop_size, dim)
                X = lo + X * (hi - lo)
                F = np.array([problem(x) for x in X])
                used += pop_size
                for i in range(pop_size):
                    update_archive(X[i], F[i])
            else:
                # Sample from archive and add noise
                X = []
                F = []
                # Ensure we have enough points for pop_size
                # Repeat archive points if necessary
                while len(X) < pop_size and used < budget:
                    idx = np.random.randint(len(archive))
                    x_base, f_base = archive[idx]
                    # Add Gaussian noise
                    noise = np.random.normal(0, 0.05, dim)
                    x_new = np.clip(x_base + noise, lo, hi)
                    # Check if already evaluated (simple check)
                    dup = False
                    for ex, ef in all_evals:
                        if np.allclose(x_new, ex, atol=1e-6):
                            dup = True
                            break
                    if not dup:
                        f_new = problem(x_new)
                        f_new = np.array(f_new)
                        all_evals.append((x_new, f_new))
                        used += 1
                        X.append(x_new)
                        F.append(f_new)
                        update_archive(x_new, f_new)
                    else:
                        # If duplicate, just pick another random point
                        x_new = np.random.rand(dim)
                        x_new = lo + x_new * (hi - lo)
                        f_new = problem(x_new)
                        f_new = np.array(f_new)
                        all_evals.append((x_new, f_new))
                        used += 1
                        X.append(x_new)
                        F.append(f_new)
                        update_archive(x_new, f_new)
                
                X = np.array(X)
                F = np.array(F)

            # DE Loop
            F_mut = 0.8
            CR = 0.9
            
            while used < budget and time.time() < deadline:
                # Select target
                p_idx = np.random.randint(pop_size)
                x_p = X[p_idx]
                f_p = F[p_idx]
                
                # Select 3 distinct parents
                while True:
                    a, b, c = np.random.randint(pop_size, size=3)
                    if a != b and b != c and a != c and a != p_idx and b != p_idx and c != p_idx:
                        break
                
                # DE Mutation
                v = X[a] + F_mut * (X[b] - X[c])
                v = np.clip(v, lo, hi)
                
                # Crossover
                u = x_p.copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        u[j] = v[j]
                
                # Evaluate
                f_u = problem(u)
                f_u = np.array(f_u)
                used += 1
                all_evals.append((u, f_u))
                
                # Update Archive
                is_new = update_archive(u, f_u)
                
                # Update Population
                X_new = np.vstack([X, u.reshape(1, -1)])
                F_new = np.vstack([F, f_u.reshape(1, -1)])
                
                fronts = non_dominated_sort(F_new)
                
                X_sel = []
                F_sel = []
                for front in fronts:
                    if len(X_sel) + len(front) <= pop_size:
                        for idx in front:
                            X_sel.append(X_new[idx])
                            F_sel.append(F_new[idx])
                    else:
                        need = pop_size - len(X_sel)
                        front_F = F_new[front]
                        cd = crowding_distance(front_F)
                        sorted_indices = np.argsort(-cd)
                        for i in range(need):
                            idx = front[sorted_indices[i]]
                            X_sel.append(X_new[idx])
                            F_sel.append(F_new[idx])
                        break
                
                X = np.array(X_sel)
                F = np.array(F_sel)

        # Phase 3: Local Polish
        if used < budget and time.time() < deadline:
            # Sort archive by potential hypervolume contribution (simple proxy: sum of objectives)
            if archive:
                # Sort by sum of objectives (lower is better)
                archive_sorted = sorted(archive, key=lambda xf: np.sum(xf[1]))
                
                # Take top few points for polishing
                n_polish = min(10, len(archive_sorted))
                polish_points = [x for x, f in archive_sorted[:n_polish]]
                
                # Local search around each point
                for x_base in polish_points:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Try small perturbations in each dimension
                    for d in range(dim):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        # Try a few step sizes
                        for step in [0.01, 0.005, 0.001]:
                            if used >= budget or time.time() > deadline:
                                break
                            
                            # Positive direction
                            x_try = x_base.copy()
                            x_try[d] = min(hi[d], x_base[d] + step)
                            if not np.allclose(x_try, x_base):
                                f_try = problem(x_try)
                                f_try = np.array(f_try)
                                used += 1
                                all_evals.append((x_try, f_try))
                                update_archive(x_try, f_try)
                            
                            # Negative direction
                            x_try = x_base.copy()
                            x_try[d] = max(lo[d], x_base[d] - step)
                            if not np.allclose(x_try, x_base):
                                f_try = problem(x_try)
                                f_try = np.array(f_try)
                                used += 1
                                all_evals.append((x_try, f_try))
                                update_archive(x_try, f_try)

        return [x for x, f in archive]
