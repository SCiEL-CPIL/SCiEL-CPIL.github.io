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
        
        pop_size = 20
        # Ensure we don't exceed budget
        generations = max(1, self.budget // pop_size)
        
        # Initialization
        # 5 points with position variables (first 2) set to 0 to target g=0
        # 15 points fully random
        X = np.zeros((pop_size, self.dim))
        # Random shape variables for all
        X[:, 2:] = lo[2:] + (hi[2:] - lo[2:]) * rng.random((pop_size, self.dim - 2))
        
        # First 5 have position 0
        X[:5, :2] = 0.0
        # Next 15 have random position
        X[5:, :2] = lo[:2] + (hi[:2] - lo[:2]) * rng.random((pop_size - 5, 2))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II Loop
        while used < self.budget:
            # Generate offspring
            offspring = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Binary tournament selection
                p1, p2 = self._tournament(F, X, rng)
                
                # SBX Crossover
                c1, c2 = self._sbx_crossover(p1, p2, lo, hi, rng)
                
                # Polynomial Mutation
                c1 = self._poly_mutation(c1, lo, hi, rng)
                c2 = self._poly_mutation(c2, lo, hi, rng)
                
                offspring[i] = c1
                # If budget allows, we could store c2, but we only have pop_size slots in offspring array
                # Standard NSGA-II creates pop_size offspring.
                # We evaluate pop_size offspring per generation.
            
            F_off = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sorting
            ranks, crowding = self._nsga2_sort(F_all)
            
            # Select next generation
            # We need to pick pop_size individuals
            # Sort by rank, then by crowding distance (descending)
            indices = np.argsort(ranks, kind='stable')
            # Within same rank, sort by crowding descending
            # Easier: create a list of (rank, -crowding) and sort
            sort_keys = np.array([(ranks[i], -crowding[i]) for i in range(len(ranks))])
            order = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))
            
            X = X_all[order[:pop_size]]
            F = F_all[order[:pop_size]]
            
            # Check time/budget
            if used >= self.budget:
                break

        return X

    def _tournament(self, F, X, rng, k=2):
        idx1 = rng.integers(0, len(F))
        idx2 = rng.integers(0, len(F))
        # Dominance check
        if self._dominates(F[idx1], F[idx2]):
            return X[idx1]
        elif self._dominates(F[idx2], F[idx1]):
            return X[idx2]
        else:
            # Random
            return X[idx1] if rng.random() < 0.5 else X[idx2]

    def _dominates(self, f1, f2):
        # f1 dominates f2 if f1 <= f2 in all objectives and < in at least one
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20.0):
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(len(p1)):
            if rng.random() < 0.5:
                # Calculate beta
                if abs(p2[i] - p1[i]) < 1e-14:
                    beta = 1.0
                else:
                    if rng.random() < 0.5:
                        beta = (2.0 * min(p1[i], p2[i]) - lo[i] + 1e-14) / (p1[i] - p2[i] + 1e-14)
                    else:
                        beta = (hi[i] - max(p1[i], p2[i]) + 1e-14) / (p1[i] - p2[i] + 1e-14)
                    
                    beta = max(0, min(beta, 100)) # Clip beta
                    
                    if beta <= 1.0:
                        beta_q = (0.5 * (beta + 1.0)) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (0.5 * (2.0 - beta)) ** (1.0 / (eta + 1.0))
                    
                    c1[i] = 0.5 * ((1.0 + beta_q) * p1[i] + (1.0 - beta_q) * p2[i])
                    c2[i] = 0.5 * ((1.0 - beta_q) * p1[i] + (1.0 + beta_q) * p2[i])
        
        return c1, c2

    def _poly_mutation(self, x, lo, hi, rng, eta=20.0):
        # Mutate each variable with probability 1/dim
        for i in range(len(x)):
            if rng.random() < 1.0 / len(x):
                y = x[i]
                if hi[i] - lo[i] < 1e-14:
                    continue
                
                delta1 = (y - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - y) / (hi[i] - lo[i])
                
                r = rng.random()
                if r < 0.5:
                    xy = (2.0 * r + (1.0 - 2.0 * r) * (delta1 + 1.0) ** (eta + 1.0)) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - (2.0 * (1.0 - r) + (2.0 * r - 1.0) * (delta2 + 1.0) ** (eta + 1.0)) ** (1.0 / (eta + 1.0))
                
                x[i] = y + xy * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        
        return x

    def _nsga2_sort(self, F):
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Non-dominated sorting
        S = [set() for _ in range(n)] # Dominated sets
        n_dom = np.zeros(n, dtype=int) # Number of dominators
        current_front = list(range(n))
        
        i = 0
        while current_front:
            next_front = []
            for p in current_front:
                for q in range(n):
                    if p == q:
                        continue
                    if self._dominates(F[p], F[q]):
                        S[p].add(q)
                        n_dom[q] += 1
                    elif self._dominates(F[q], F[p]):
                        S[q].add(p)
                        n_dom[p] += 1
                if n_dom[p] == 0:
                    ranks[p] = i
                    next_front.append(p)
            
            current_front = next_front
            i += 1
        
        # Crowding distance
        for i in range(len(ranks)):
            front = [j for j in range(n) if ranks[j] == i]
            if not front:
                continue
            
            if len(front) == 1:
                crowding[front[0]] = np.inf
                continue
            
            for k in range(self.n_obj):
                # Sort by objective k
                front_sorted = sorted(front, key=lambda idx: F[idx][k])
                crowding[front_sorted[0]] = np.inf
                crowding[front_sorted[-1]] = np.inf
                
                f_min = F[front_sorted[0]][k]
                f_max = F[front_sorted[-1]][k]
                range_k = f_max - f_min
                
                if range_k < 1e-14:
                    continue
                    
                for j in range(1, len(front_sorted) - 1):
                    idx = front_sorted[j]
                    idx_prev = front_sorted[j-1]
                    idx_next = front_sorted[j+1]
                    crowding[idx] += (F[idx_next][k] - F[idx_prev][k]) / range_k
        
        return ranks, crowding
