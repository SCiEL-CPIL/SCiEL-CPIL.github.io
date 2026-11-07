"""Optimized program for the suite task; implements an efficient NSGA-II for WFG9."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II based multi-objective optimizer with vectorized dominance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin from 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # NSGA-II parameters
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget
        
        # Initialize population
        # Use the global numpy random state since the evaluator sets np.random.seed(seed)
        # This ensures reproducibility based on the seed set by the evaluator.
        
        # Initialize with random points
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        evals_used = pop_size
        
        # Pre-compute ranks and distances for initial population
        ranks, distances = self._compute_rank_distance(F)
        
        while evals_used < self.budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate offspring
            Q = np.empty((pop_size, self.dim))
            FQ = np.empty((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Selection: Tournament selection on parents based on rank and distance
                p1_idx = self._tournament_select(ranks, distances, 2)
                p2_idx = self._tournament_select(ranks, distances, 2)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover: SBX (Simulated Binary Crossover) or BLX-Alpha
                # Let's use a simple blend crossover with mutation
                c = self._crossover(p1, p2, lo, hi)
                
                # Mutation: Gaussian with adaptive sigma
                # Use a small sigma to allow exploration but mostly exploitation
                sigma = 0.1 * range_
                # Mutation rate: each gene mutated with probability 1/dim
                mut_mask = np.random.rand(self.dim) < (1.0 / self.dim)
                c[mut_mask] += np.random.randn(np.sum(mut_mask)) * sigma[mut_mask]
                
                c = np.clip(c, lo, hi)
                
                # Evaluate
                if evals_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                fc = problem(c)
                Q[i] = c
                FQ[i] = fc
                evals_used += 1
            
            # If we broke early due to budget/time, Q might be partially filled
            # We only proceed if we have a full offspring population
            if evals_used - pop_size < pop_size:
                break
            
            # Combine parents and offspring
            R = np.vstack([X, Q])
            FR = np.vstack([F, FQ])
            
            # Non-dominated sorting and Crowding distance
            ranks, distances = self._compute_rank_distance(FR)
            
            # Select next population of size pop_size from R
            # Sort by (rank, -distance)
            # We want to pick the best 'pop_size' individuals
            # Create a list of indices and sort them based on rank and distance
            indices = np.arange(len(R))
            
            # Sort keys: primary rank (ascending), secondary distance (descending)
            # np.lexsort sorts by the last key first.
            # So we pass (-distance, rank) -> sorts by rank, then -distance (i.e., distance desc)
            sorted_indices = np.lexsort((-distances, ranks))
            
            selected_indices = sorted_indices[:pop_size]
            
            X = R[selected_indices]
            F = FR[selected_indices]
            
            # Recompute ranks and distances for the new population for the next selection
            # Actually, we just recomputed them for R. We need them for X (the subset).
            # But the ranks/distances in X are relative to R.
            # For tournament selection in the next generation, we need ranks/distances relative to X.
            # However, NSGA-II typically uses the ranks from the combined population for selection.
            # Once selected, the new population is just the set of individuals.
            # For the next generation's selection, we should recompute dominance within the new population X.
            ranks, distances = self._compute_rank_distance(F)
            
        return X

    def _compute_rank_distance(self, F):
        """
        Compute non-dominated ranks and crowding distances for a set of objective vectors F.
        F: (N, n_obj) array
        Returns:
            ranks: (N,) array of integers (0 is best)
            distances: (N,) array of crowding distances
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        distances = np.zeros(n)
        
        # Fast non-dominated sorting
        # S[i] = set of points dominated by i
        # n_dominates[i] = number of points dominating i
        
        S = [set() for _ in range(n)]
        n_dominates = np.zeros(n, dtype=int)
        
        # Calculate dominance relations
        # Vectorized approach:
        # F[i] <= F[j] for all objectives AND F[i] < F[j] for at least one
        # Let's compute a dominance matrix D where D[i,j] is True if i dominates j
        
        # F: (N, M)
        # F_i <= F_j: (N, 1, M) <= (1, N, M) -> (N, N, M)
        # This might be memory intensive for large N, but N=40 here, so 40*40*3 = 4800 floats. Fine.
        
        # all_leq[i,j] = all(F[i] <= F[j])
        # any_lt[i,j] = any(F[i] < F[j])
        
        all_leq = np.all(F[:, np.newaxis, :] <= F[np.newaxis, :, :], axis=2)
        any_lt = np.any(F[:, np.newaxis, :] < F[np.newaxis, :, :], axis=2)
        
        # i dominates j if all_leq[i,j] and any_lt[i,j]
        dominates = all_leq & any_lt
        
        # n_dominates[j] = sum over i of dominates[i,j]
        n_dominates = np.sum(dominates, axis=0)
        
        # S[i] = indices j where dominates[i,j] is True
        for i in range(n):
            S[i] = set(np.where(dominates[i, :])[0])
            
        # Initial front
        front = np.where(n_dominates == 0)[0]
        current_rank = 0
        
        while len(front) > 0:
            for i in front:
                ranks[i] = current_rank
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        next_front.append(j)
            front = np.array(next_front, dtype=int)
            current_rank += 1
            
        # Crowding distance
        unique_ranks = np.unique(ranks)
        for r in unique_ranks:
            front_indices = np.where(ranks == r)[0]
            m = len(front_indices)
            if m <= 2:
                distances[front_indices] = np.inf
                continue
            
            # Sort front by each objective
            for k in range(self.n_obj):
                obj_indices = np.argsort(F[front_indices, k])
                sorted_front = front_indices[obj_indices]
                min_val = F[sorted_front[0], k]
                max_val = F[sorted_front[-1], k]
                
                if max_val == min_val:
                    continue
                    
                distances[sorted_front[0]] = np.inf
                distances[sorted_front[-1]] = np.inf
                
                for i in range(1, m - 1):
                    d = (F[sorted_front[i+1], k] - F[sorted_front[i-1], k]) / (max_val - min_val)
                    distances[sorted_front[i]] += d
                    
        return ranks, distances

    def _tournament_select(self, ranks, distances, k):
        """
        Select an index using tournament selection.
        """
        n = len(ranks)
        indices = np.random.randint(0, n, size=k)
        
        best_idx = indices[0]
        best_rank = ranks[best_idx]
        best_dist = distances[best_idx]
        
        for idx in indices[1:]:
            r = ranks[idx]
            d = distances[idx]
            if r < best_rank:
                best_idx = idx
                best_rank = r
                best_dist = d
            elif r == best_rank:
                if d > best_dist:
                    best_idx = idx
                    best_rank = r
                    best_dist = d
                    
        return best_idx

    def _crossover(self, p1, p2, lo, hi):
        """
        Simple blend crossover (BLX-Alpha) or uniform.
        """
        c = np.empty_like(p1)
        # Use a random switch for each gene
        # Or just mix them
        for i in range(self.dim):
            low = min(p1[i], p2[i])
            high = max(p1[i], p2[i])
            if low == high:
                c[i] = p1[i]
            else:
                # Sample from [low - 0.5*(high-low), high + 0.5*(high-low)]
                # This is BLX-alpha with alpha=0.5
                alpha = 0.5
                new_low = low - alpha * (high - low)
                new_high = high + alpha * (high - low)
                c[i] = np.random.uniform(new_low, new_high)
        return np.clip(c, lo, hi)
# EVOLVE-BLOCK-END
