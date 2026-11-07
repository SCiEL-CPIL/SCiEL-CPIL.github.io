import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = [0]  # Mutable counter for nested functions
        archive_X = []
        archive_F = []
        
        def dominates(f1, f2):
            """Check if f1 dominates f2 (f1 is better in at least one obj and not worse in any)"""
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f):
            f = np.array(f)
            # Check if dominated by any existing point in archive
            dominated = False
            for af in archive_F:
                if dominates(af, f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            return True

        def safe_eval(x):
            if used[0] >= self.budget:
                return None
            try:
                f = problem(x)
                used[0] += 1
                return f
            except Exception:
                return None

        def non_dominated_sort(F):
            """Return ranks for non-dominated sorting (NSGA-II style)"""
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_count = [0] * n  # n_count[i] = number of points that dominate i
            ranks = [0] * n
            front = [0]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_count[i] += 1
            
            # Assign rank 0 to non-dominated points
            for i in range(n):
                if n_count[i] == 0:
                    ranks[i] = 0
                    front[0] += 1
            
            # Assign subsequent ranks
            rank = 0
            while front[rank] > 0:
                rank += 1
                front.append(0)
                next_front = []
                for i in range(n):
                    if ranks[i] == rank - 1:
                        for j in S[i]:
                            n_count[j] -= 1
                            if n_count[j] == 0:
                                ranks[j] = rank
                                next_front.append(j)
                front[rank] = len(next_front)
            
            return ranks

        def crowding_distance(F, indices):
            """Compute crowding distance for points with given indices"""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            distances = [0.0] * n
            F_idx = [F[i] for i in indices]
            
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = sorted(range(n), key=lambda i: F_idx[i][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = F_idx[sorted_indices[0]][m]
                f_max = F_idx[sorted_indices[-1]][m]
                range_val = f_max - f_min
                
                if range_val < 1e-10:
                    continue
                
                for k in range(1, n - 1):
                    i = sorted_indices[k]
                    i_prev = sorted_indices[k - 1]
                    i_next = sorted_indices[k + 1]
                    distances[i] += (F_idx[i_next][m] - F_idx[i_prev][m]) / range_val
            
            return distances

        # Initialize population using Latin Hypercube Sampling
        pop_size = 30  # Population size for DE
        n_init = min(pop_size, self.budget)
        
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * (hi - lo)
        
        population_X = []
        population_F = []
        
        for x in X_init:
            if time.time() > deadline or used[0] >= self.budget:
                break
            f = safe_eval(x)
            if f is not None:
                population_X.append(x)
                population_F.append(f)
                add_to_archive(x, f)
        
        # Differential Evolution with non-dominated selection
        CR = 0.8  # Crossover rate
        F_mut = 0.5  # Mutation factor
        
        while used[0] < self.budget and time.time() < deadline:
            if len(population_X) < 3:
                break
            
            n_pop = len(population_X)
            
            # Generate offspring for each individual
            offspring_X = []
            offspring_F = []
            
            for i in range(n_pop):
                if used[0] >= self.budget or time.time() > deadline:
                    break
                
                # Select three distinct individuals different from i
                candidates = [j for j in range(n_pop) if j != i]
                if len(candidates) < 3:
                    break
                
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_a + F*(x_b - x_c)
                v = population_X[a] + F_mut * (population_X[b] - population_X[c])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover
                j_rand = np.random.randint(0, self.dim)
                trial_x = np.array(population_X[i])
                for j in range(self.dim):
                    if np.random.random() < CR or j == j_rand:
                        trial_x[j] = v[j]
                
                trial_x = np.clip(trial_x, lo, hi)
                
                f_trial = safe_eval(trial_x)
                if f_trial is not None:
                    offspring_X.append(trial_x)
                    offspring_F.append(f_trial)
                    add_to_archive(trial_x, f_trial)
            
            # Combine parent and offspring populations
            combined_X = population_X + offspring_X
            combined_F = population_F + offspring_F
            
            if len(combined_F) == 0:
                break
            
            # Non-dominated sort
            ranks = non_dominated_sort(combined_F)
            
            # Select next population
            next_X = []
            next_F = []
            
            for rank in range(max(ranks) + 1):
                if len(next_X) >= pop_size:
                    break
                
                # Get indices with this rank
                rank_indices = [i for i in range(len(combined_F)) if ranks[i] == rank]
                
                if len(next_X) + len(rank_indices) <= pop_size:
                    for i in rank_indices:
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
                else:
                    # Need to select based on crowding distance
                    distances = crowding_distance(combined_F, rank_indices)
                    # Sort by crowding distance (descending)
                    sorted_indices = sorted(range(len(rank_indices)), key=lambda k: distances[k], reverse=True)
                    for k in sorted_indices:
                        if len(next_X) >= pop_size:
                            break
                        i = rank_indices[k]
                        next_X.append(combined_X[i])
                        next_F.append(combined_F[i])
            
            population_X = next_X
            population_F = next_F
        
        # Final polish: use remaining budget for local refinement of archive points
        if archive_F and used[0] < self.budget:
            # Sort archive by sum of objectives (proxy for "goodness")
            archive_indices = list(range(len(archive_F)))
            archive_indices.sort(key=lambda i: np.sum(archive_F[i]))
            
            # Polish top few points with small perturbations
            n_polish = min(5, len(archive_indices))
            for idx in archive_indices[:n_polish]:
                if used[0] >= self.budget or time.time() > deadline:
                    break
                
                x0 = archive_X[idx].copy()
                # Try small random perturbations
                for _ in range(3):
                    if used[0] >= self.budget or time.time() > deadline:
                        break
                    perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
                    x_trial = np.clip(x0 + perturbation, lo, hi)
                    f_trial = safe_eval(x_trial)
                    if f_trial is not None:
                        add_to_archive(x_trial, f_trial)
        
        return archive_X
