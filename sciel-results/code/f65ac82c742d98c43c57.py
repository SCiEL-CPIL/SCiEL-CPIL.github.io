# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 20
        self.evals_used = 0
        self.archive = []
        self.bounds = None

    def _evaluate(self, problem, bounds, pop):
        """Evaluate a population and update eval count."""
        F = np.zeros((len(pop), self.n_obj))
        for i, x in enumerate(pop):
            if self.evals_used >= self.budget:
                break
            F[i] = problem(x)
            self.evals_used += 1
        return F

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting on objective matrix F."""
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                    n_dom[p] += 1
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dom[q] += 1
        
        current_front = []
        for p in range(n):
            if n_dom[p] == 0:
                ranks[p] = 0
                current_front.append(p)
        
        i = 0
        while current_front:
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            current_front = next_front
            
        return ranks

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a set of indices."""
        if len(indices) <= 2:
            return np.full(len(indices), np.inf)
        
        dist = np.zeros(len(indices))
        for m in range(self.n_obj):
            f_m = F[indices, m]
            min_f = np.min(f_m)
            max_f = np.max(f_m)
            if max_f - min_f == 0:
                continue
            sorted_idx = np.argsort(f_m)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            for i in range(1, len(sorted_idx) - 1):
                dist[sorted_idx[i]] += (f_m[sorted_idx[i+1]] - f_m[sorted_idx[i-1]]) / (max_f - min_f)
        return dist

    def __call__(self, problem, bounds):
        self.bounds = bounds
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        
        # Initialize population
        pop = np.random.rand(self.pop_size, self.dim) * (ub - lb) + lb
        F = self._evaluate(problem, bounds, pop)
        
        while self.evals_used < self.budget:
            # Selection
            ranks = self._non_dominated_sort(F)
            # Sort by rank, then by crowding distance
            # We need to select parents. Let's use tournament selection.
            
            # To simplify, we'll just create offspring from the best non-dominated front
            # and fill the rest with random if needed, or just use the whole pop.
            
            # Create offspring
            offspring = np.zeros_like(pop)
            off_F = np.zeros_like(F)
            
            for i in range(self.pop_size):
                if self.evals_used >= self.budget:
                    break
                
                # Tournament selection
                # Pick 2 random individuals, choose the one with better rank (and crowding)
                p1_idx = np.random.randint(0, self.pop_size)
                p2_idx = np.random.randint(0, self.pop_size)
                
                # Simple comparison based on rank
                if ranks[p1_idx] < ranks[p2_idx]:
                    parent = pop[p1_idx]
                elif ranks[p2_idx] < ranks[p1_idx]:
                    parent = pop[p2_idx]
                else:
                    # Same rank, pick random or use crowding (simplified: random)
                    parent = pop[np.random.choice([p1_idx, p2_idx])]
                
                # Mutation: Gaussian perturbation
                child = parent + np.random.randn(self.dim) * 0.1 * (ub - lb)
                child = np.clip(child, lb, ub)
                
                offspring[i] = child
                off_F[i] = problem(child)
                self.evals_used += 1
            
            # Combine
            combined_pop = np.vstack([pop, offspring])
            combined_F = np.vstack([F, off_F])
            
            # Sort and select next population
            c_ranks = self._non_dominated_sort(combined_F)
            
            # Select best self.pop_size individuals
            # Sort indices by rank
            sorted_indices = np.argsort(c_ranks)
            
            # If we have more than pop_size, we need to cut off by crowding distance
            # For simplicity in this short budget, we just take the first pop_size from sorted ranks.
            # This might be biased towards early fronts.
            
            # Better: Get all individuals in rank 0, then rank 1, etc.
            selected_indices = []
            current_rank = 0
            while len(selected_indices) < self.pop_size:
                front_indices = [i for i in range(len(combined_pop)) if c_ranks[i] == current_rank]
                if len(front_indices) + len(selected_indices) <= self.pop_size:
                    selected_indices.extend(front_indices)
                else:
                    # Need to select from this front using crowding
                    dist = self._crowding_distance(combined_F, front_indices)
                    # Sort by distance descending
                    sorted_front = np.argsort(-dist)
                    needed = self.pop_size - len(selected_indices)
                    selected_indices.extend([front_indices[i] for i in sorted_front[:needed]])
                    break
                current_rank += 1
                
            pop = combined_pop[selected_indices]
            F = combined_F[selected_indices]
            
        # Return the best points found (the final population or archive)
        # The task says "Every returned objective vector is recorded".
        # The return value is ignored.
        # We should return a representative set or just the last population.
        # Since the evaluator records ALL calls to problem(), the internal
        # evaluations are what matter. The return value of __call__ is ignored.
        # So we just need to ensure we made the calls.
        
        return pop
# EVOLVE-BLOCK-END
