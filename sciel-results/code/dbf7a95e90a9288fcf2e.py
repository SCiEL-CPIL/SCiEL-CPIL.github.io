import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        pop_size = 25
        n_gen = (self.budget - pop_size) // pop_size
        n_gen = max(n_gen, 1)
        
        # Initialize population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
            
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    current_front.append(i)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts
        
        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            for m in range(self.n_obj):
                vals = F[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-12:
                    cd[:] = np.inf
                    continue
                order = np.argsort(vals)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                for k in range(1, n - 1):
                    cd[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / (max_val - min_val)
            return cd
        
        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated Binary Crossover."""
            child1 = np.zeros(self.dim)
            child2 = np.zeros(self.dim)
            for i in range(self.dim):
                if rng.random() < 0.5:
                    beta = 1.0 / (1.0 + 2.0 * rng.random())
                    beta1 = beta ** (1.0 / (eta + 1.0))
                    beta2 = (1.0 / beta) ** (1.0 / (eta + 1.0))
                    child1[i] = 0.5 * ((1 + beta1) * x1[i] + (1 - beta1) * x2[i])
                    child2[i] = 0.5 * ((1 + beta2) * x1[i] + (1 - beta2) * x2[i])
                else:
                    child1[i] = x2[i]
                    child2[i] = x1[i]
            return child1, child2
        
        def polynomial_mutation(x, eta=20.0):
            """Polynomial mutation."""
            child = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    delta = rng.random()
                    if delta < 0.5:
                        delta1 = (2.0 * delta) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta1 = 1.0 - (2.0 * (1.0 - delta)) ** (1.0 / (eta + 1.0))
                    child[i] += delta1 * (hi[i] - lo[i])
            return np.clip(child, lo, hi)
        
        # Main NSGA-II loop
        for gen in range(n_gen):
            # Create offspring
            offspring_X = []
            for _ in range(pop_size // 2):
                # Tournament selection for parent 1
                idx1 = rng.integers(pop_size, 2)
                f1 = non_dominated_sort(F[idx1])
                # Pick from best front
                best_front1 = f1[0]
                p1 = X[best_front1[rng.integers(len(best_front1))]]
                
                # Tournament selection for parent 2
                idx2 = rng.integers(pop_size, 2)
                f2 = non_dominated_sort(F[idx2])
                best_front2 = f2[0]
                p2 = X[best_front2[rng.integers(len(best_front2))]]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = non_dominated_sort(combined_F)
            
            # Select next generation
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate using crowding distance
                    remaining = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    order = np.argsort(-cd)
                    for idx in order[:remaining]:
                        new_X.append(combined_X[front[idx]])
                        new_F.append(combined_F[front[idx]])
                    break
            
            X = np.array(new_X)
            F = np.array(new_F)
        
        return X
