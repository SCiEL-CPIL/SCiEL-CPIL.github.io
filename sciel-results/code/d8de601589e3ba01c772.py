import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 20
        # Initialize with Halton sequence for better spread
        X = self._halton_sequence(pop_size, self.dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        crossover_prob = 0.9
        mutation_prob = 0.1
        eta_c = 15.0
        eta_m = 20.0
        
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            # We generate pop_size offspring to replace the whole population
            # But we only have limited budget. 
            # Standard NSGA-II generates pop_size offspring per generation.
            # With budget 200 and pop 20, we can do 8 generations.
            
            for i in range(pop_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select parents
                p1_idx = self._tournament_select(F, 2)
                p2_idx = self._tournament_select(F, 2)
                
                # Crossover
                if np.random.rand() < crossover_prob:
                    c1, c2 = self._sbx_crossover(X[p1_idx], X[p2_idx], eta_c, lo, hi)
                else:
                    c1 = X[p1_idx].copy()
                    c2 = X[p2_idx].copy()
                
                # Mutation
                c1 = self._polynomial_mutation(c1, eta_m, lo, hi)
                c2 = self._polynomial_mutation(c2, eta_m, lo, hi)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if used >= self.budget:
                    break
                
                if time.time() > deadline:
                    break
                
                f2 = problem(c2)
                used += 1
                offspring_X.append(c2)
                offspring_F.append(f2)
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Selection
            X, F = self._nsga2_select(combined_X, combined_F, pop_size)
            
        # Return the best points (non-dominated)
        # The task says "Every returned objective vector is recorded... return value is ignored"
        # But we should return something reasonable.
        return X

    def _halton_sequence(self, num_points, dim, lo, hi):
        points = []
        for i in range(num_points):
            p = []
            for d in range(dim):
                # Base for halton is d+2
                base = d + 2
                f = 1.0
                r = 0.0
                i_val = i + 1
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                p.append(lo[d] + r * (hi[d] - lo[d]))
            points.append(p)
        return np.array(points)

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_count = [0] * n
        ranks = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                # i dominates j?
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_count[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_count[j] += 1
        
        front = [0]
        current_front = []
        for i in range(n):
            if n_count[i] == 0:
                ranks[i] = 0
                current_front.append(i)
        
        k = 0
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            current_front = next_front
            k += 1
            front.append(k)
            
        return ranks, S

    def _crowding_distance(self, F, ranks, n_points):
        n = len(F)
        distances = np.zeros(n)
        
        # Group by rank
        for r in range(max(ranks) + 1):
            indices = [i for i in range(n) if ranks[i] == r]
            if len(indices) <= 2:
                for i in indices:
                    distances[i] = float('inf')
                continue
                
            for m in range(self.n_obj):
                # Sort by objective m
                sorted_indices = sorted(indices, key=lambda x: F[x][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                min_val = F[sorted_indices[0]][m]
                max_val = F[sorted_indices[-1]][m]
                range_val = max_val - min_val
                if range_val == 0:
                    continue
                    
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    distances[idx] += (F[next_idx][m] - F[prev_idx][m]) / range_val
                    
        return distances

    def _nsga2_select(self, X, F, pop_size):
        ranks, _ = self._fast_non_dominated_sort(F)
        distances = self._crowding_distance(F, ranks, len(F))
        
        # Sort by rank, then by crowding distance (descending)
        # We want to pick the best pop_size individuals
        order = np.argsort(ranks)
        # For same rank, sort by distance descending
        # numpy argsort is stable, so we can sort by -distance first, then rank
        # Actually, easier to just sort indices
        indices = np.arange(len(F))
        
        # Create a composite key for sorting
        # Rank (lower is better), Distance (higher is better)
        # We can sort by (rank, -distance)
        
        # Simple approach: iterate through fronts
        selected_indices = []
        current_front_size = 0
        
        # Get unique ranks
        unique_ranks = sorted(list(set(ranks)))
        
        for r in unique_ranks:
            front_indices = [i for i in range(len(F)) if ranks[i] == r]
            if current_front_size + len(front_indices) <= pop_size:
                selected_indices.extend(front_indices)
                current_front_size += len(front_indices)
            else:
                # Need to select only a subset of this front
                needed = pop_size - current_front_size
                # Sort front_indices by crowding distance descending
                front_indices_sorted = sorted(front_indices, key=lambda x: distances[x], reverse=True)
                selected_indices.extend(front_indices_sorted[:needed])
                break
                
        return X[selected_indices], F[selected_indices]

    def _tournament_select(self, F, k=2):
        n = len(F)
        # Simple tournament on rank and crowding is complex without pre-computed arrays.
        # For small n, we can just pick random and compare dominance.
        # However, standard NSGA-II uses non-dominated sorting.
        # To keep it simple and fast for n=20, we can just do random selection 
        # and rely on the replacement step to maintain quality? 
        # No, selection pressure is important.
        
        # Let's do a quick non-dominated sort for selection
        ranks, _ = self._fast_non_dominated_sort(F)
        distances = self._crowding_distance(F, ranks, n)
        
        candidates = np.random.choice(n, k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best]:
                if distances[c] > distances[best]:
                    best = c
        return best

    def _sbx_crossover(self, x1, x2, eta, lo, hi):
        c1 = x1.copy()
        c2 = x2.copy()
        
        for i in range(len(x1)):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 2.0 * u
                        beta = ((1.0 + 2.0 * (x1[i] - lo[i]) / (x2[i] - lo[i])) ** (eta + 1.0))
                    else:
                        alpha = 2.0 * (1.0 - u)
                        beta = ((1.0 + 2.0 * (hi[i] - x2[i]) / (hi[i] - x1[i])) ** (eta + 1.0))
                    
                    beta = 1.0 / beta
                    g1 = 0.5 * (x1[i] + x2[i]) - 0.5 * (x2[i] - x1[i]) * beta
                    g2 = 0.5 * (x1[i] + x2[i]) + 0.5 * (x2[i] - x1[i]) * beta
                    
                    c1[i] = g1
                    c2[i] = g2
        
        return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

    def _polynomial_mutation(self, x, eta, lo, hi):
        for i in range(len(x)):
            if np.random.rand() < 0.1: # Mutation rate per gene
                u = np.random.rand()
                if u < 0.5:
                    xy = (x[i] - lo[i]) / (hi[i] - lo[i])
                    delta1 = (2.0 * xy) ** (1.0 / (eta + 1.0))
                    delta = 1.0 - (1.0 - delta1) ** (eta + 1.0)
                else:
                    xy = (hi[i] - x[i]) / (hi[i] - lo[i])
                    delta2 = (2.0 * xy) ** (1.0 / (eta + 1.0))
                    delta = 1.0 - (1.0 - delta2) ** (eta + 1.0)
                
                x[i] += (hi[i] - lo[i]) * delta
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
