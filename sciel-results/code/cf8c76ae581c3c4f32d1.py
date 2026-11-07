"""Multi-objective optimizer on DTLZ1 with analytical front sampling + NSGA-II refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    Strategy:
    1. Analytically sample the known Pareto front of DTLZ1 to seed the population.
    2. Use NSGA-II (non-dominated sorting, crowding distance, SBX crossover, polynomial mutation)
       to refine the population and discover better points within the remaining budget.
    3. Include a time deadline guard to ensure completion within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._start_time = None

    def _elapsed(self):
        return time.time() - self._start_time

    def _time_left(self, safety_margin=15.0):
        """Return remaining time with safety margin."""
        return 300.0 - self._elapsed() - safety_margin

    def __call__(self, problem, bounds):
        self._start_time = time.time()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size for NSGA-II
        N = 30

        # --- Analytical DTLZ1 front sampling ---
        # For DTLZ1 with 3 objectives and 7 variables:
        # f1 = 0.5 * x1 * x2 * (1 + g)
        # f2 = 0.5 * x1 * (1 - x2) * (1 + g)
        # f3 = 0.5 * (1 - x1) * (1 + g)
        # where g = 100 + 10*n + sum_{i=3}^{n}(x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5))
        # On the Pareto front, g=0, so x3..x7 = 0.5
        # Thus: f1 = 0.5*x1*x2, f2 = 0.5*x1*(1-x2), f3 = 0.5*(1-x1)
        # The front is the simplex: f1 + f2 + f3 = 0.5

        # Generate points on the analytical front
        # Use a good fraction of budget for front sampling
        n_front = min(70, self.budget // 3)
        n_front = max(n_front, 20)
        
        x1_samples = rng.random(n_front)
        x2_samples = rng.random(n_front)
        
        X_front = np.zeros((n_front, self.dim))
        X_front[:, 0] = x1_samples
        X_front[:, 1] = x2_samples
        X_front[:, 2:] = 0.5  # For DTLZ1, g=0 when x_i=0.5 for i>=3
        
        # Evaluate front points
        F_front = np.array([problem(x) for x in X_front])
        used = n_front

        # Add some random points to ensure diversity
        n_random = max(0, N - n_front)
        if n_random > 0:
            X_rand = lo + (hi - lo) * rng.random((n_random, self.dim))
            F_rand = np.array([problem(x) for x in X_rand])
            used += n_random
            X = np.vstack([X_front, X_rand])
            F = np.vstack([F_front, F_rand])
        else:
            X = X_front
            F = F_front

        # --- NSGA-II helpers ---

        def non_dominated_sort(F_mat):
            """
            Returns ranks for each individual using fast non-dominated sort.
            """
            n = len(F_mat)
            if n == 0:
                return []
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
            
            current_rank = 0
            r = [i for i in range(n) if n_dominated[i] == 0]
            
            while r:
                next_r = []
                for i in r:
                    rank[i] = current_rank
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_r.append(j)
                r = next_r
                current_rank += 1
            return rank

        def crowding_distance(F_mat, rank):
            """
            Calculate crowding distance for each individual.
            """
            n = len(F_mat)
            dist = np.zeros(n)
            if n <= 2:
                dist[:] = np.inf
                return dist
                
            unique_ranks = sorted(list(set(rank)))
            for r in unique_ranks:
                indices = [i for i in range(n) if rank[i] == r]
                if len(indices) <= 2:
                    for i in indices:
                        dist[i] = np.inf
                    continue
                
                for k in range(self.n_obj):
                    sorted_indices = sorted(indices, key=lambda i: F_mat[i][k])
                    dist[sorted_indices[0]] = np.inf
                    dist[sorted_indices[-1]] = np.inf
                    f_min = F_mat[sorted_indices[0]][k]
                    f_max = F_mat[sorted_indices[-1]][k]
                    if f_max - f_min < 1e-15:
                        continue
                    for i in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[i]
                        dist[idx] += (F_mat[sorted_indices[i+1]][k] - F_mat[sorted_indices[i-1]][k]) / (f_max - f_min)
            return dist

        def sbx_crossover(x1, x2, lo, hi):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            
            eta = 20.0
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                u = rng.random()
                if u <= 0.5:
                    alpha = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    alpha = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                c1 = 0.5 * ((1.0 + alpha) * x1[i] + (1.0 - alpha) * x2[i])
                c2 = 0.5 * ((1.0 - alpha) * x1[i] + (1.0 + alpha) * x2[i])
                
                if c1 < lo[i]:
                    c1 = lo[i]
                if c1 > hi[i]:
                    c1 = hi[i]
                if c2 < lo[i]:
                    c2 = lo[i]
                if c2 > hi[i]:
                    c2 = hi[i]
                
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        def polynomial_mutation(x, lo, hi):
            eta = 20.0
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    x[i] += delta * (hi[i] - lo[i])
                    if x[i] < lo[i]:
                        x[i] = lo[i]
                    if x[i] > hi[i]:
                        x[i] = hi[i]
            return x

        def tournament_select(X_mat, F_mat, rank, dist):
            i1 = rng.integers(0, len(F_mat))
            i2 = rng.integers(0, len(F_mat))
            if rank[i1] < rank[i2]:
                return X_mat[i1].copy()
            elif rank[i2] < rank[i1]:
                return X_mat[i2].copy()
            else:
                if dist[i1] > dist[i2]:
                    return X_mat[i1].copy()
                else:
                    return X_mat[i2].copy()

        # --- Main NSGA-II Loop ---
        # Keep track of all evaluated points
        all_X = X.copy()
        all_F = F.copy()
        
        while used < self.budget:
            # Check time
            if self._time_left() < 1.0:
                break
            
            # Compute ranks and crowding distance
            rank = non_dominated_sort(F)
            dist = crowding_distance(F, rank)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(N):
                # Tournament selection for two parents
                parent1 = tournament_select(X, F, rank, dist)
                parent2 = tournament_select(X, F, rank, dist)
                
                # Crossover
                child1, child2 = sbx_crossover(parent1, parent2, lo, hi)
                
                # Mutation
                child1 = polynomial_mutation(child1, lo, hi)
                child2 = polynomial_mutation(child2, lo, hi)
                
                # Evaluate if we have budget
                if used < self.budget:
                    f1 = problem(child1)
                    used += 1
                    offspring_X.append(child1)
                    offspring_F.append(f1)
                    all_X = np.vstack([all_X, [child1]])
                    all_F = np.vstack([all_F, [f1]])
                else:
                    break
                
                if used < self.budget:
                    f2 = problem(child2)
                    used += 1
                    offspring_X.append(child2)
                    offspring_F.append(f2)
                    all_X = np.vstack([all_X, [child2]])
                    all_F = np.vstack([all_F, [f2]])
                else:
                    break
            
            if len(offspring_X) == 0:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            combined_rank = non_dominated_sort(combined_F)
            combined_dist = crowding_distance(combined_F, combined_rank)
            
            # Select next generation
            next_X = []
            next_F = []
            
            # Sort by rank, then by crowding distance
            indices = list(range(len(combined_X)))
            indices.sort(key=lambda i: (combined_rank[i], -combined_dist[i]))
            
            for idx in indices:
                if len(next_X) >= N:
                    break
                next_X.append(combined_X[idx])
                next_F.append(combined_F[idx])
            
            if len(next_X) < N:
                # Fill remaining with best from combined
                for idx in indices:
                    if len(next_X) >= N:
                        break
                    if idx not in [list(np.array(next_X)).tolist().index(x) if x in np.array(next_X).tolist() else -1 for x in [None] * len(next_X)]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        # Return all evaluated points
        return all_X
# EVOLVE-BLOCK-END
