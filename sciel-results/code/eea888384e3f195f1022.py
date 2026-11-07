"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a lightweight NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Parameters
        pop_size = 20
        n_generations = budget // pop_size
        alpha = 0.5  # BLX-alpha crossover parameter
        sigma_mut = 0.1  # Mutation strength

        # Initialize population
        X = np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        
        # Precompute ranks and crowding for initial pop
        ranks, crowdings = self._compute_rank_crowding(F)
        
        for gen in range(n_generations):
            # Generate offspring
            Y = np.zeros((pop_size, dim))
            for i in range(pop_size):
                # Tournament selection
                p1_idx = self._tournament_select(ranks, crowdings, 2)
                p2_idx = self._tournament_select(ranks, crowdings, 2)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # BLX-alpha Crossover
                child = self._blx_alpha_crossover(p1, p2, alpha)
                
                # Mutation
                child = child + np.random.normal(0, sigma_mut, dim)
                child = np.clip(child, 0, 1)
                
                Y[i] = child
            
            F_y = np.array([problem(x) for x in Y])
            
            # Combine and Select
            X_comb = np.vstack((X, Y))
            F_comb = np.vstack((F, F_y))
            
            # Select N best
            selected_indices = self._nsga2_select(F_comb, pop_size)
            
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]
            
            # Recompute ranks and crowding for next generation
            ranks, crowdings = self._compute_rank_crowding(F)

        return F

    def _dominates(self, a, b):
        """Check if a dominates b."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Returns list of fronts (list of indices) and ranks array."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominated = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=int)
        
        fronts = []
        current_front = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dominated[i] += 1
            if n_dominated[i] == 0:
                rank[i] = 0
                current_front.append(i)
                
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        rank[j] = len(fronts)
                        next_front.append(j)
            current_front = next_front
            
        return fronts, rank

    def _crowding_distance(self, F, front_indices):
        """Calculate crowding distance for points in a front. Returns array of same length as front_indices."""
        n = len(front_indices)
        if n == 0:
            return np.zeros(0)
        if n == 1:
            return np.array([float('inf')])
            
        dist = np.zeros(n)
        F_front = F[front_indices]
        
        # Map global indices to local indices in front
        global_to_local = {idx: i for i, idx in enumerate(front_indices)}
        
        for m in range(self.n_obj):
            sorted_local = np.argsort(F_front[:, m])
            dist[sorted_local[0]] = float('inf')
            dist[sorted_local[-1]] = float('inf')
            
            f_min = F_front[sorted_local[0], m]
            f_max = F_front[sorted_local[-1], m]
            if f_max == f_min:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_local[i]
                prev_idx = sorted_local[i - 1]
                next_idx = sorted_local[i + 1]
                
                dist[idx] += (F_front[next_idx, m] - F_front[prev_idx, m]) / (f_max - f_min)
                
        return dist

    def _compute_rank_crowding(self, F):
        """Compute ranks and crowding distances for all individuals in F."""
        fronts, rank = self._non_dominated_sort(F)
        n = len(F)
        crowding = np.zeros(n)
        
        for front in fronts:
            cd = self._crowding_distance(F, front)
            for i, idx in enumerate(front):
                crowding[idx] = cd[i]
                
        return rank, crowding

    def _nsga2_select(self, F, N):
        """Select N indices from F using NSGA-II selection."""
        fronts, rank = self._non_dominated_sort(F)
        
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= N:
                selected_indices.extend(front)
            else:
                needed = N - len(selected_indices)
                cd = self._crowding_distance(F, front)
                # Sort by crowding distance descending
                sorted_front = sorted(zip(front, cd), key=lambda x: x[1], reverse=True)
                selected_indices.extend([idx for idx, _ in sorted_front[:needed]])
                break
                
        return selected_indices

    def _tournament_select(self, ranks, crowdings, k):
        """Tournament selection based on rank and crowding distance."""
        n = len(ranks)
        participants = np.random.choice(n, k, replace=False)
        best = participants[0]
        for p in participants[1:]:
            # Lower rank is better
            if ranks[p] < ranks[best]:
                best = p
            elif ranks[p] == ranks[best]:
                # Higher crowding distance is better
                if crowdings[p] > crowdings[best]:
                    best = p
        return best

    def _blx_alpha_crossover(self, p1, p2, alpha):
        """BLX-alpha crossover."""
        low = np.minimum(p1, p2)
        high = np.maximum(p1, p2)
        spread = high - low
        l = low - alpha * spread
        u = high + alpha * spread
        child = np.random.uniform(l, u)
        return child

# EVOLVE-BLOCK-END
