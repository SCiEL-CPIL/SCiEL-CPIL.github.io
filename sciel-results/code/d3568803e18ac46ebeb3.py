"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using NSGA-II-like approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size: use a reasonable fraction of budget
        # For budget=200, we want enough diversity but not too many
        pop_size = 40
        
        # Initialize population
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume (given in task)
        ref_point = np.array([3.0, 5.0, 7.0])
        
        def dominates(a, b):
            """Check if a dominates b (minimizing all objectives)."""
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            n_count = np.zeros(n, dtype=int)  # n_count[i] = number of points dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_count[i] += 1
            
            front1 = [i for i in range(n) if n_count[i] == 0]
            for i in front1:
                rank[i] = 0
            
            k = 1
            current_front = front1
            fronts = [current_front]
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                    current_front = next_front
                    k += 1
            return fronts
        
        def crowding_distance(F, indices):
            """Calculate crowding distance for points in indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            idx_arr = np.array(indices)
            F_sub = F[idx_arr]
            
            for m in range(n_obj):
                # Sort by objective m
                order = np.argsort(F_sub[:, m])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                min_val = F_sub[order[0], m]
                max_val = F_sub[order[-1], m]
                range_val = max_val - min_val
                if range_val > 1e-15:
                    for i in range(1, n - 1):
                        dist[order[i]] += (F_sub[order[i+1], m] - F_sub[order[i-1], m]) / range_val
            return dist
        
        def tournament_select(X, F, fronts, k=2):
            """Tournament selection based on rank and crowding distance."""
            # Compute crowding distance for each front
            front_crowd = {}
            for front in fronts:
                if len(front) > 0:
                    front_crowd[tuple(front)] = crowding_distance(F, front)
            
            def get_key(i):
                # Find which front i is in
                rank_val = 0
                for idx, front in enumerate(fronts):
                    if i in front:
                        rank_val = idx
                        break
                # Get crowding distance
                front_tuple = tuple(front)
                crowd = front_crowd.get(front_tuple, np.array([0.0]))
                # Find position of i in front
                pos = front.index(i)
                return (rank_val, -crowd[pos])
            
            candidates = rng.integers(len(X), size=k)
            best = min(candidates, key=get_key)
            return X[best].copy()
        
        def crossover(parent1, parent2):
            """Simulated binary crossover (SBX)."""
            eta = 20.0
            child1 = parent1.copy()
            child2 = parent2.copy()
            for i in range(dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            return child1, child2
        
        def mutate(x, rate=0.2, sigma=0.1):
            """Polynomial mutation."""
            eta = 20.0
            child = x.copy()
            for i in range(dim):
                if rng.random() < rate:
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    range_i = hi[i] - lo[i]
                    child[i] += delta * range_i
            child = np.clip(child, lo, hi)
            return child
        
        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute fronts
            fronts = non_dominated_sort(F)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used < self.budget:
                # Select two parents via tournament
                p1 = tournament_select(X, F, fronts, k=2)
                p2 = tournament_select(X, F, fronts, k=2)
                
                # Crossover
                c1, c2 = crossover(p1, p2)
                
                # Mutate
                c1 = mutate(c1, rate=0.3, sigma=0.1)
                c2 = mutate(c2, rate=0.3, sigma=0.1)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= self.budget:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            # Combine parents and offspring
            X_all = np.vstack([X, np.array(offspring_X)])
            F_all = np.vstack([F, np.array(offspring_F)])
            
            # Selection: pick pop_size best from combined
            fronts_all = non_dominated_sort(F_all)
            
            # Compute crowding for each front
            front_crowd_all = {}
            for front in fronts_all:
                if len(front) > 0:
                    front_crowd_all[tuple(front)] = crowding_distance(F_all, front)
            
            # Select individuals
            selected_indices = []
            for front in fronts_all:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Need to select some from this front based on crowding
                    remaining = pop_size - len(selected_indices)
                    front_arr = np.array(front)
                    crowd_vals = front_crowd_all[tuple(front)]
                    # Select top 'remaining' by crowding distance
                    order = np.argsort(crowd_vals)[::-1]
                    selected_indices.extend(front_arr[order[:remaining]].tolist())
                    break
            
            X = X_all[selected_indices]
            F = F_all[selected_indices]
        
        return X
# EVOLVE-BLOCK-END
