"""Multi-objective optimizer on RE37 using NSGA-II with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using NSGA-II with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Fast Non-Dominated Sorting.
        Returns a list of fronts, where each front is a list of indices.
        """
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] contains indices dominated by i
        nd = np.zeros(n, dtype=int)  # Number of solutions dominating i
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]
        
        fronts = [front]
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
                
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        
        return dist

    def _tournament_selection(self, fronts, crowding, size):
        """Tournament selection based on rank and crowding distance."""
        n = len(fronts) if fronts else 0
        if n == 0:
            return np.array([])
        
        # Assign ranks
        rank = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        selected = []
        for _ in range(size):
            # Pick two random candidates
            idx1 = np.random.randint(n)
            idx2 = np.random.randint(n)
            
            # Compare by rank first
            if rank[idx1] < rank[idx2]:
                selected.append(idx1)
            elif rank[idx2] < rank[idx1]:
                selected.append(idx2)
            else:
                # Same rank, compare by crowding distance
                if crowding[idx1] > crowding[idx2]:
                    selected.append(idx1)
                else:
                    selected.append(idx2)
        
        return np.array(selected)

    def _crossover(self, x1, x2, lo, hi, crossover_rate=0.9):
        """Simulated binary crossover (SBX)."""
        if np.random.random() > crossover_rate:
            return x1.copy(), x2.copy()
        
        # Simple blend crossover
        alpha = np.random.random(self.dim)
        x_child1 = alpha * x1 + (1 - alpha) * x2
        x_child2 = (1 - alpha) * x1 + alpha * x2
        
        x_child1 = np.clip(x_child1, lo, hi)
        x_child2 = np.clip(x_child2, lo, hi)
        
        return x_child1, x_child2

    def _mutation(self, x, lo, hi, mutation_rate=0.1, sigma=0.1):
        """Gaussian mutation."""
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.random() < mutation_rate:
                x_mut[i] += sigma * (hi[i] - lo[i]) * np.random.standard_normal()
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        
        return x_mut

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # NSGA-II parameters
        pop_size = 40
        n_evals_used = 0
        
        # Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        n_evals_used += pop_size
        
        # Archive to store all evaluated points
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # NSGA-II iterations
        while n_evals_used < self.budget and time.time() < deadline - 5:
            # Evaluate remaining budget
            remaining = self.budget - n_evals_used
            
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and n_evals_used < self.budget:
                # Non-dominated sorting and crowding distance for current population
                fronts = self._fast_non_dominated_sort(F)
                n = len(F)
                crowding = np.zeros(n)
                for front in fronts:
                    if len(front) > 1:
                        cd = self._crowding_distance(F, front)
                        for i, idx in enumerate(front):
                            crowding[idx] = cd[i]
                
                # Select parents via tournament
                parent1_idx = self._tournament_selection(fronts, crowding, 1)[0]
                parent2_idx = self._tournament_selection(fronts, crowding, 1)[0]
                
                parent1 = X[parent1_idx]
                parent2 = X[parent2_idx]
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2, lo, hi)
                
                # Mutation
                child1 = self._mutation(child1, lo, hi)
                child2 = self._mutation(child2, lo, hi)
                
                # Evaluate offspring
                if n_evals_used < self.budget:
                    f1 = problem(child1)
                    offspring_X.append(child1)
                    offspring_F.append(f1)
                    n_evals_used += 1
                
                if n_evals_used < self.budget:
                    f2 = problem(child2)
                    offspring_X.append(child2)
                    offspring_F.append(f2)
                    n_evals_used += 1
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Selection: non-dominated sorting and crowding distance
            fronts = self._fast_non_dominated_sort(combined_F)
            n_combined = len(combined_F)
            crowding = np.zeros(n_combined)
            
            for front in fronts:
                if len(front) > 1:
                    cd = self._crowding_distance(combined_F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = cd[i]
            
            # Select next generation
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Sort by crowding distance and take top
                    remaining_slots = pop_size - len(selected_indices)
                    if remaining_slots > 0:
                        front_crowding = np.array([crowding[idx] for idx in front])
                        order = np.argsort(front_crowding)[::-1]
                        selected_indices.extend([front[i] for i in order[:remaining_slots]])
                    break
            
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]
            
            # Update archive
            all_X.append(X.copy())
            all_F.append(F.copy())
        
        # Combine all evaluated points
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)
        
        # Return all objective values
        return all_F
# EVOLVE-BLOCK-END
