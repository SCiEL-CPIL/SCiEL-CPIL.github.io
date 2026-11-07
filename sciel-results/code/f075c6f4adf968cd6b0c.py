"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II with DTLZ1-specific initialization and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        pop_size = min(50, budget)
        
        # Initialize population with DTLZ1-specific structure
        # DTLZ1: x = [x1, x2, ..., x_M, y1, y2, ..., y_(n-M)]
        # Where M = n_obj, and the remaining variables are position variables
        # f1 = 0.5 * x1 * prod(1 - f_i)
        # f2 = 0.5 * (1-x1) * x2 * prod(1 - f_i)
        # f3 = 0.5 * (1-x1) * (1-x2) * prod(1 - f_i)
        # where f_i = 1 - (y_i^2 - 10*cos(20*pi*y_i))/(1+21*(y_i^2-10*cos(20*pi*y_i))^2)
        
        # For DTLZ1, the Pareto front is at y_i = 0 for all i
        # So we should initialize y variables near 0 and x variables spread out
        
        X = np.zeros((pop_size, dim))
        # Initialize x variables (first n_obj) uniformly
        X[:, :n_obj] = rng.random((pop_size, n_obj))
        # Initialize y variables (remaining) near 0 with some spread
        X[:, n_obj:] = 0.1 * rng.random((pop_size, dim - n_obj))
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dominated = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominated[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dominated[i] += 1
                
                if n_dominated[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            
            # Remove empty last front if any
            if not fronts[-1]:
                fronts.pop()
                
            return fronts

        # Crowding distance
        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(F[indices, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_min = F[indices[sorted_idx[0]], m]
                f_max = F[indices[sorted_idx[-1]], m]
                if f_max - f_min < 1e-10:
                    continue
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (F[indices[sorted_idx[k + 1]], m] - 
                                          F[indices[sorted_idx[k - 1]], m]) / (f_max - f_min)
            return cd

        # SBX crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = np.zeros_like(x1)
            child2 = np.zeros_like(x2)
            for i in range(len(x1)):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        child1[i] = x1[i]
                        child2[i] = x2[i]
                    else:
                        if rng.random() <= 0.5:
                            beta = (2 * rng.random()) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * rng.random())) ** (1.0 / (eta + 1.0))
                        c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        child1[i] = c1
                        child2[i] = c2
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            return child1, child2

        # Polynomial mutation
        def polynomial_mutation(x, eta=20.0):
            x = x.copy()
            for i in range(len(x)):
                if rng.random() < 1.0 / dim:
                    if x[i] >= hi[i] - 1e-10 or x[i] <= lo[i] + 1e-10:
                        x[i] = lo[i] + (hi[i] - lo[i]) * rng.random()
                    else:
                        delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                        delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                        if rng.random() <= 0.5:
                            xy = 1.0 - (1.0 - delta1) ** (eta + 1.0)
                            deltaq = (2.0 * rng.random() + xy) ** (1.0 / (eta + 1.0)) - xy
                        else:
                            xy = 1.0 - (1.0 - delta2) ** (eta + 1.0)
                            deltaq = xy - (2.0 * rng.random() + xy) ** (1.0 / (eta + 1.0))
                        x[i] = x[i] + deltaq * (hi[i] - lo[i])
                        x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main NSGA-II loop
        while used < budget:
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and used < budget:
                # Select parents using tournament selection
                def tournament_select(F, fronts, pop_size):
                    # Get front ranks
                    rank = np.zeros(pop_size, dtype=int)
                    for front_idx, front in enumerate(fronts):
                        for i in front:
                            rank[i] = front_idx
                    # Tournament size 2
                    i1 = rng.integers(pop_size)
                    i2 = rng.integers(pop_size)
                    if rank[i1] < rank[i2]:
                        return i1
                    elif rank[i2] < rank[i1]:
                        return i2
                    else:
                        # Same front, use crowding distance
                        cd = crowding_distance(F, list(range(pop_size)))
                        if cd[i1] > cd[i2]:
                            return i1
                        else:
                            return i2
                
                p1 = tournament_select(F, fronts, pop_size)
                p2 = tournament_select(F, fronts, pop_size)
                
                if p1 == p2:
                    p2 = (p2 + 1) % pop_size
                
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2])
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= budget:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            combined_n = len(combined_X)
            
            # Non-dominated sorting
            fronts = non_dominated_sort(combined_F)
            
            # Select next population
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.append(combined_X[front])
                    new_F.append(combined_F[front])
                else:
                    # Need to truncate this front using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_front = np.argsort(-cd)
                    selected = front[sorted_front[:remaining]]
                    new_X.append(combined_X[selected])
                    new_F.append(combined_F[selected])
                    break
            
            if new_X:
                X = np.vstack(new_X)
                F = np.vstack(new_F)
            else:
                break

        return X
# EVOLVE-BLOCK-END
