"""Multi-objective optimizer on DTLZ7 using NSGA-II with Memetic Local Search."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using NSGA-II with a Memetic Local Search component.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        eval_count = 0
        all_points = []
        all_objectives = []
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            eval_count += 1
            f = np.array(problem(x))
            all_points.append(x.copy())
            all_objectives.append(f)
            return f

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def fast_non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns a list of fronts, where each front is a list of indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = [0] * n  # n_d[i] = number of individuals dominating i
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_d[i] += 1
                        # Note: The loop structure above is slightly inefficient for large n, 
                        # but for n=20 it is fine. 
                        # Wait, the standard algorithm checks all pairs. 
                        # Let's stick to a simpler O(n^2) implementation for clarity and small n.
            
            # Re-doing fast_non_dominated_sort properly for clarity
            S = [[] for _ in range(n)]
            n_d = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_d[i] += 1
            
            front = []
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    front.append(i)
                    
            fronts = []
            i = 0
            while front:
                fronts.append(front)
                next_front = []
                for k in front:
                    for m in S[k]:
                        n_d[m] -= 1
                        if n_d[m] == 0:
                            rank[m] = i + 1
                            next_front.append(m)
                front = next_front
                i += 1
            return fronts, rank

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for individuals in a front.
            """
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            distances = [0.0] * n
            for m in range(self.n_obj):
                # Sort front by objective m
                sorted_indices = sorted(front_indices, key=lambda i: F[i][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                
                for k in range(1, n - 1):
                    i_prev = sorted_indices[k - 1]
                    i_next = sorted_indices[k + 1]
                    distances[sorted_indices[k]] += (F[i_next][m] - F[i_prev][m]) / range_f
            
            return distances

        # Parameters
        pop_size = 10
        eta_c = 20.0  # Distribution index for SBX
        eta_m = 20.0  # Distribution index for Mutation
        p_c = 0.9     # Crossover probability
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        for i in range(pop_size):
            F[i] = eval_point(X[i])
        
        # Main Loop
        while eval_count < self.budget:
            # Generate offspring
            Q_X = []
            Q_F = []
            
            while len(Q_X) < pop_size:
                if eval_count >= self.budget:
                    break
                    
                # Select parents using Tournament Selection
                def tournament_select(X, F, rank, k=2):
                    idxs = rng.choice(len(F), size=k, replace=False)
                    best_idx = idxs[0]
                    for idx in idxs[1:]:
                        if rank[idx] < rank[best_idx]:
                            best_idx = idx
                        elif rank[idx] == rank[best_idx]:
                            # Tie-breaker: random or crowding (simplified to random for speed)
                            if rng.random() < 0.5:
                                best_idx = idx
                    return best_idx

                # We need ranks for selection. Calculate for current P.
                fronts, rank = fast_non_dominated_sort(F)
                
                i = tournament_select(X, F, rank)
                j = tournament_select(X, F, rank)
                
                parent1 = X[i].copy()
                parent2 = X[j].copy()
                
                # SBX Crossover
                if rng.random() < p_c:
                    child1 = np.zeros_like(parent1)
                    child2 = np.zeros_like(parent2)
                    for k in range(self.dim):
                        if rng.random() < 0.5:
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1.0 / (eta_c + 1.0))
                            else:
                                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0))
                            
                            x1, x2 = parent1[k], parent2[k]
                            child1[k] = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2)
                            child2[k] = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2)
                        else:
                            child1[k] = parent1[k]
                            child2[k] = parent2[k]
                    # Clip to bounds
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                # Polynomial Mutation
                def polynomial_mutate(x):
                    delta1 = (1.0 - (1.0 - (x - lo) / (hi - lo)) ** (eta_m + 1.0)) / eta_m
                    delta2 = 1.0 - (1.0 - (hi - x) / (hi - lo)) ** (eta_m + 1.0) / eta_m
                    # Actually, standard polynomial mutation formula:
                    # delta = (2u)^{1/(eta+1)} or (1-2(1-u))^{1/(eta+1)}
                    # Let's use a simpler random walk if budget is tight, but SBX is better.
                    # For simplicity and robustness in this snippet, let's use a simple gaussian mutation
                    # scaled by the range.
                    return x
                
                # Let's implement a simple random mutation to avoid complex SBX mutation bugs
                for k in range(self.dim):
                    if rng.random() < 0.5:
                        child1[k] += rng.normal(0, 0.1 * (hi[k] - lo[k]))
                    if rng.random() < 0.5:
                        child2[k] += rng.normal(0, 0.1 * (hi[k] - lo[k]))
                
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                f1 = eval_point(child1)
                if f1 is not None:
                    Q_X.append(child1)
                    Q_F.append(f1)
                    
                if len(Q_X) < pop_size:
                    f2 = eval_point(child2)
                    if f2 is not None:
                        Q_X.append(child2)
                        Q_F.append(f2)
            
            if not Q_X:
                break
                
            Q_F = np.array(Q_F)
            
            # Combine P and Q
            R_X = np.vstack([X, np.array(Q_X)])
            R_F = np.vstack([F, Q_F])
            
            # Non-dominated sorting on R
            fronts, rank = fast_non_dominated_sort(R_F)
            
            # Select P from R
            new_X = []
            new_F = []
            new_rank = [0] * len(R_X)
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                        new_rank.append(rank[idx])
                else:
                    # Need to select from this front based on crowding distance
                    cd = crowding_distance(R_F, front)
                    # Sort front indices by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[i - len(R_X) // 2 if i >= len(R_X) else i], reverse=True)
                    # Wait, cd is calculated for the front indices relative to R_F? 
                    # My crowding_distance function takes F (the full array) and front_indices (indices into F).
                    # So cd[i] corresponds to R_F[i].
                    
                    # We need to pick (pop_size - len(new_X)) individuals from this front
                    needed = pop_size - len(new_X)
                    # Sort front by cd descending
                    front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in front_sorted[:needed]:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                        new_rank.append(rank[idx])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
            
            # Memetic Local Search
            # Apply local search to the best individual (lowest rank, highest crowding?)
            # Let's pick the individual with the best hypervolume contribution proxy: 
            # Low rank, and if rank 0, lowest sum of objectives.
            
            if eval_count < self.budget - 2: # Keep some budget
                # Find best individual
                best_idx = 0
                best_val = float('inf')
                for i in range(len(F)):
                    # Prefer rank 0
                    if rank[i] == 0:
                        val = np.sum(F[i])
                        if val < best_val:
                            best_val = val
                            best_idx = i
                    elif rank[i] < rank[best_idx]:
                        best_idx = i
                        best_val = np.sum(F[i])
                
                # Local search on best_idx
                current_x = X[best_idx].copy()
                current_f = F[best_idx].copy()
                
                # Try 3 random perturbations
                for _ in range(3):
                    if eval_count >= self.budget:
                        break
                    perturbation = rng.normal(0, 0.05 * (hi - lo), self.dim)
                    new_x = np.clip(current_x + perturbation, lo, hi)
                    new_f = eval_point(new_x)
                    if new_f is None:
                        break
                    
                    # Accept if dominates current or is non-dominated and has better "quality"
                    if dominates(new_f, current_f):
                        current_x = new_x
                        current_f = new_f
                    elif not dominates(current_f, new_f):
                        # Non-dominated: accept with 50% probability to maintain diversity
                        if rng.random() < 0.5:
                            current_x = new_x
                            current_f = new_f
                
                # Update population with the improved individual
                X[best_idx] = current_x
                F[best_idx] = current_f

        return all_points
# EVOLVE-BLOCK-END
