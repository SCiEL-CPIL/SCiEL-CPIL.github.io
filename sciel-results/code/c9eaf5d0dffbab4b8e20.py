"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II Multi-Objective Optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Parameters
        pop_size = 20
        # NSGA-II parameters
        cx_prob = 0.9  # Crossover probability
        mut_prob = 0.1  # Mutation probability (per gene)
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        
        # Initialize Population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def non_dominated_sort(F_set):
            n = len(F_set)
            S = [set() for _ in range(n)]
            n_count = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F_set[i], F_set[j]):
                        S[i].add(j)
                    elif dominates(F_set[j], F_set[i]):
                        n_count[i] += 1
            
            front = [i for i in range(n) if n_count[i] == 0]
            if not front:
                return [list(range(n))], [0.0] * n
                
            fronts.append(front)
            
            while front:
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                front = next_front
                
            crowding = [0.0] * n
            for front in fronts:
                if len(front) <= 2:
                    for idx in front:
                        crowding[idx] = float('inf')
                    continue
                for m in range(self.n_obj):
                    sorted_front = sorted(front, key=lambda idx: F_set[idx][m])
                    crowding[sorted_front[0]] = float('inf')
                    crowding[sorted_front[-1]] = float('inf')
                    f_min = F_set[sorted_front[0]][m]
                    f_max = F_set[sorted_front[-1]][m]
                    f_range = f_max - f_min
                    if f_range == 0:
                        continue
                    for k in range(1, len(sorted_front) - 1):
                        idx = sorted_front[k]
                        prev_idx = sorted_front[k-1]
                        next_idx = sorted_front[k+1]
                        crowding[idx] += (F_set[next_idx][m] - F_set[prev_idx][m]) / f_range
                        
            return fronts, crowding

        def tournament_select(fronts, crowding, X, F):
            # Binary tournament
            i1, i2 = rng.integers(0, len(F), size=2)
            
            # Get rank
            rank1 = next(idx for idx, f in enumerate(fronts) if i1 in f)
            rank2 = next(idx for idx, f in enumerate(fronts) if i2 in f)
            
            if rank1 < rank2:
                return i1
            elif rank2 < rank1:
                return i2
            else:
                # Same rank, pick higher crowding distance
                if crowding[i1] > crowding[i2]:
                    return i1
                else:
                    return i2

        def crossover(x1, x2):
            if rng.random() > cx_prob:
                return x1.copy(), x2.copy()
            
            # Simulated Binary Crossover (SBX)
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            
            for i in range(dim):
                if rng.random() < 0.5:
                    # Calculate SBX
                    x1_i, x2_i = x1[i], x2[i]
                    if x1_i == x2_i:
                        child1[i] = x1_i
                        child2[i] = x2_i
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                        
                        c1 = 0.5 * ((1 + beta) * x1_i + (1 - beta) * x2_i)
                        c2 = 0.5 * ((1 - beta) * x1_i + (1 + beta) * x2_i)
                        
                        # Clip
                        c1 = np.clip(c1, lo[i], hi[i])
                        c2 = np.clip(c2, lo[i], hi[i])
                        
                        child1[i] = c1
                        child2[i] = c2
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
                    
            return child1, child2

        def mutation(x):
            if rng.random() < mut_prob:
                for i in range(dim):
                    if rng.random() < 0.5:
                        # Polynomial mutation
                        u = rng.random()
                        if u < 0.5:
                            delta1 = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                        else:
                            delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                        
                        x[i] += delta1 * (hi[i] - lo[i])
                        x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while used < budget:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if used >= budget:
                    break
                    
                # Select parents
                p1_idx = tournament_select(*non_dominated_sort(F), F)
                p2_idx = tournament_select(*non_dominated_sort(F), F)
                
                # Crossover
                child1, child2 = crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                child1 = mutation(child1)
                
                # Evaluate
                f_child = problem(child1)
                used += 1
                
                offspring_X.append(child1)
                offspring_F.append(f_child)
                
            if not offspring_X:
                break
                
            # Combine
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Sort and Select
            fronts, crowding = non_dominated_sort(combined_F)
            
            # Select best pop_size individuals
            new_X = []
            new_F = []
            current_rank = 0
            while len(new_X) < pop_size and current_rank < len(fronts):
                front = fronts[current_rank]
                if len(new_X) + len(front) <= pop_size:
                    # Take all
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Take only some, sort by crowding distance
                    sorted_front = sorted(front, key=lambda idx: crowding[idx], reverse=True)
                    for idx in sorted_front:
                        if len(new_X) < pop_size:
                            new_X.append(combined_X[idx])
                            new_F.append(combined_F[idx])
                        else:
                            break
                current_rank += 1
                
            X = np.array(new_X)
            F = np.array(new_F)

        return X
# EVOLVE-BLOCK-END
