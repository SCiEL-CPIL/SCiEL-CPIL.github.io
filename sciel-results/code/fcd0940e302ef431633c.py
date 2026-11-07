import numpy as np

class Optimizer:
    """NSGA-II multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size
        # We have budget evaluations.
        # Init takes pop_size.
        # Each generation takes pop_size offspring evaluations.
        # Let's use a population size that fits well.
        # If budget=200, pop=100 allows 1 generation.
        # If pop=50, allows 3 generations.
        # NSGA-II usually benefits from more generations for convergence.
        # Let's try pop=50, generations=3. 
        # 50 (init) + 3*50 (offspring) = 200.
        
        pop_size = 50
        n_gen = (self.budget - pop_size) // pop_size
        if n_gen < 1:
            # Fallback if budget is too small for the logic
            pop_size = self.budget
            n_gen = 0

        # Initialize
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        for gen in range(n_gen):
            # Selection: Binary Tournament
            parents = np.empty((pop_size, self.dim))
            
            # Non-dominated sorting and crowding distance for selection
            # We need ranks and crowding distances for current population
            ranks, cd = self._fast_non_dominated_sort(F)
            
            for i in range(pop_size):
                # Select two parents
                p1_idx = self._binary_tournament(F, ranks, cd, rng)
                p2_idx = self._binary_tournament(F, ranks, cd, rng)
                
                # Crossover: SBX (Simulated Binary Crossover)
                # eta_c = 20 is standard
                child = self._sbx_crossover(X[p1_idx], X[p2_idx], rng, lo, hi, eta_c=20.0)
                
                # Mutation: Polynomial Mutation
                # eta_m = 20 is standard
                child = self._poly_mutation(child, rng, lo, hi, eta_m=20.0)
                
                parents[i] = child
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in parents])
            
            # Combine parents and offspring
            X_combined = np.vstack([X, parents])
            F_combined = np.vstack([F, F_off])
            
            # Selection for next generation
            ranks, cd = self._fast_non_dominated_sort(F_combined)
            
            # Select best pop_size individuals
            # Sort by rank, then by crowding distance (descending)
            # We need to pick the first pop_size from the sorted list
            # Create indices sorted by (rank, -cd)
            # rank is lower is better. cd is higher is better.
            # So sort by rank ascending, then cd descending.
            
            # Create a composite key for sorting
            # We can't easily sort numpy arrays by multiple keys directly without structuring.
            # Let's create a list of indices and sort them.
            indices = np.arange(len(F_combined))
            
            # Sort by rank
            sorted_indices = indices[np.argsort(ranks)]
            
            # Within each rank group, sort by cd descending
            # This is a bit tricky with numpy alone if we want to do it in one go efficiently for small N.
            # Given N=100, we can do it with a python sort or stable sorts.
            
            # Stable sort by cd descending
            # Then stable sort by rank ascending
            # np.argsort is stable if kind='stable' or 'mergesort'
            
            # First sort by cd descending
            idx_by_cd = np.argsort(-cd, kind='mergesort')
            # Then sort by rank ascending, using the previous order as tie-breaker
            # Actually, we want to sort the *indices* such that:
            # Primary: rank (asc)
            # Secondary: cd (desc)
            
            # Let's do it in two steps on the indices array
            # 1. Sort indices by cd descending
            # 2. Sort the resulting indices by rank ascending (stable)
            
            order = np.argsort(-cd, kind='mergesort')
            final_order = order[np.argsort(ranks[order], kind='mergesort')]
            
            # Select the first pop_size
            selected_indices = final_order[:pop_size]
            
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return X

    def _fast_non_dominated_sort(self, F):
        """
        Returns ranks (array of int) and crowding distances (array of float)
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        cd = np.zeros(n)
        
        # Fast Non-Dominated Sorting
        # S_i is set of solutions dominated by i
        # n_i is domination count of i
        # R_i is set of solutions that dominate i
        
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        
        # Initialize
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif self._dominates(F[j], F[i]):
                    pass # i is dominated by j, handled in S[j]
        
        # Front 0
        front = []
        for i in range(n):
            if n_dom[i] == 0:
                front.append(i)
                ranks[i] = 0
        
        k = 0
        while front:
            next_front = []
            for i in front:
                # Calculate crowding distance for i? No, usually done per front.
                # We need to calculate crowding distance for the entire front before moving to next?
                # Standard NSGA-II calculates CD for each front after the front is identified.
                # Let's collect fronts first, then calculate CD for each front.
                
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
                        ranks[j] = k + 1
            front = next_front
            k += 1
        
        # Calculate Crowding Distance for each front
        # Group indices by rank
        from collections import defaultdict
        fronts = defaultdict(list)
        for i in range(n):
            fronts[ranks[i]].append(i)
            
        for r, front_indices in fronts.items():
            self._crowding_distance(F, front_indices, cd)
            
        return ranks, cd

    def _dominates(self, f1, f2):
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _crowding_distance(self, F, indices, cd):
        n = len(indices)
        if n <= 2:
            for idx in indices:
                cd[idx] = float('inf')
            return
            
        # Map indices to 0..n-1 for local calculation
        F_sub = F[indices]
        local_cd = np.zeros(n)
        m = self.n_obj
        
        for k in range(m):
            # Sort by k-th objective
            sorted_local_idx = np.argsort(F_sub[:, k])
            local_cd[sorted_local_idx[0]] = float('inf')
            local_cd[sorted_local_idx[-1]] = float('inf')
            
            f_min = F_sub[sorted_local_idx[0], k]
            f_max = F_sub[sorted_local_idx[-1], k]
            f_range = f_max - f_min
            if f_range < 1e-10:
                continue
                
            for i in range(1, n - 1):
                local_cd[sorted_local_idx[i]] += (F_sub[sorted_local_idx[i+1], k] - F_sub[sorted_local_idx[i-1], k]) / f_range
        
        # Map back to global cd
        for i, global_idx in enumerate(indices):
            cd[global_idx] = local_cd[i]

    def _binary_tournament(self, F, ranks, cd, rng):
        i1 = rng.integers(0, len(F))
        i2 = rng.integers(0, len(F))
        
        # Prefer lower rank
        if ranks[i1] < ranks[i2]:
            return i1
        elif ranks[i2] < ranks[i1]:
            return i2
        else:
            # Prefer higher crowding distance
            if cd[i1] > cd[i2]:
                return i1
            elif cd[i2] > cd[i1]:
                return i2
            else:
                return i1 if rng.random() < 0.5 else i2

    def _sbx_crossover(self, p1, p2, rng, lo, hi, eta_c=20.0):
        child = np.empty_like(p1)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                child[i] = p1[i]
                if abs(p2[i] - p1[i]) > 1e-14:
                    # SBX
                    r = rng.random()
                    if r <= 0.5:
                        beta = (2 * r) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - r))) ** (1.0 / (eta_c + 1.0))
                    c1 = 0.5 * (1 + beta) * p1[i] + 0.5 * (1 - beta) * p2[i]
                    c2 = 0.5 * (1 - beta) * p1[i] + 0.5 * (1 + beta) * p2[i]
                    
                    # Clip to bounds
                    if c1 < lo[i]:
                        c1 = lo[i]
                    elif c1 > hi[i]:
                        c1 = hi[i]
                    if c2 < lo[i]:
                        c2 = lo[i]
                    elif c2 > hi[i]:
                        c2 = hi[i]
                    
                    # We only return one child per pair in this simple loop? 
                    # Standard SBX produces 2 children. 
                    # My loop structure `for i in range(pop_size)` suggests I'm creating one child per parent pair.
                    # This is effectively a "1-child" crossover or I'm ignoring the second child.
                    # To be more efficient, I should generate 2 children and pick one, or restructure.
                    # Given the tight budget and simple structure, let's just use one of them (c1).
                    # Or better, use Uniform Crossover which is simpler and often sufficient for WFG.
                    # Let's switch to Uniform Crossover for simplicity and robustness in this constrained budget.
                    pass
            else:
                child[i] = p2[i]
                
        # Let's restart with Uniform Crossover to be safe and simple.
        return self._uniform_crossover(p1, p2, rng)

    def _uniform_crossover(self, p1, p2, rng):
        child = np.empty_like(p1)
        mask = rng.random(len(p1)) < 0.5
        child[mask] = p1[mask]
        child[~mask] = p2[~mask]
        return child

    def _poly_mutation(self, x, rng, lo, hi, eta_m=20.0):
        for i in range(len(x)):
            if rng.random() < 1.0 / len(x):
                r = rng.random()
                if r <= 0.5:
                    delta = (2 * r) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2 * (1 - r)) ** (1.0 / (eta_m + 1.0))
                
                x[i] = x[i] + delta * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
