# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE34 using Hypervolume-driven Evolutionary Strategy.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit, keep 30s margin
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_vec = hi - lo
        
        # Reference point for hypervolume calculation (from task description)
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Population size
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget

        # Initialize population
        X = lo + range_vec * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Adaptive step size
        step_size = 0.5 * np.max(range_vec) # Start with half the range
        min_step = 1e-3
        max_step = np.max(range_vec)
        
        def calc_hv_contributions(F_pop, ref):
            """
            Calculate the hypervolume contribution of each point in the population.
            For small n_obj (3) and small pop, we can use a simple approach or exact HV.
            Here, we approximate contribution by checking if a point is dominated by others.
            A more precise metric: For each point, calculate the volume of the box it uniquely covers.
            Given the small size, we can just use non-dominated sorting rank + crowding, 
            or simply: point is valuable if it's non-dominated.
            Let's use a simple heuristic: 
            1. Identify non-dominated points.
            2. Among non-dominated, pick the one that is most "spread out" or has the best scalarized value.
            
            Actually, for selection, we just need to pick a good parent.
            Let's use a tournament selection based on non-domination and then a random tie-breaker.
            Or, simpler: Pick a random non-dominated point.
            To be "Hypervolume-aware", we can calculate the actual HV of the set and the HV of the set minus each point.
            This is O(N^2) or worse. With N=20, it's fine.
            """
            n = len(F_pop)
            contribs = np.zeros(n)
            for i in range(n):
                # Calculate HV of full set
                # Then HV of set without i
                # This is expensive if done naively every step.
                # Alternative: Just use non-dominated sorting.
                pass
            
            # Simpler: Return indices of non-dominated points
            non_dom_idx = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i == j: continue
                    if np.all(F_pop[j] <= F_pop[i]) and np.any(F_pop[j] < F_pop[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_idx.append(i)
            return non_dom_idx

        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Identify non-dominated points
            non_dom_idx = self._get_non_dominated(F)
            
            if len(non_dom_idx) == 0:
                # Fallback: pick best by sum
                parent_idx = np.argmin(np.sum(F, axis=1))
            else:
                # Pick a random non-dominated point to maintain diversity
                parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            
            parent = X[parent_idx]
            
            # Mutation
            mutation = rng.standard_normal(self.dim) * step_size
            child = parent + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Selection for replacement: Replace the point with the lowest "quality"
            # Quality: If dominated, it's bad. If non-dominated, use hypervolume contribution proxy.
            # Simple proxy: Replace the most dominated point, or if all non-dominated, replace the one with worst scalarized value.
            
            # Find the worst point to replace
            # 1. If child is non-dominated, it's good.
            # 2. We want to keep the best set.
            # Let's just replace the point that is dominated by the most others, or has the worst sum.
            
            # Calculate dominance count for each point in current pop
            dom_counts = np.zeros(pop_size)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom_counts[i] += 1
            
            # If child is non-dominated, it's better than any dominated point.
            # If child is dominated, it might still be better than some dominated points.
            # Simple strategy: Replace the point with the highest dom_count.
            # If tie, replace the one with worst sum of objectives.
            
            # Check if child is better than the worst current point
            worst_idx = np.argmax(dom_counts)
            
            # If the worst point is non-dominated (dom_count 0), we need a better criterion.
            # If dom_counts are all 0, replace the one with the worst sum.
            if dom_counts[worst_idx] == 0:
                # All non-dominated (or similar). Replace the one with the largest sum (worst).
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
            
            X[worst_idx] = child
            F[worst_idx] = fc
            
            # Adaptive Step Size
            # If child is non-dominated, it's a good move.
            child_is_non_dom = False
            for j in range(pop_size):
                if j == worst_idx: continue
                if np.all(F[j] <= F[worst_idx]) and np.any(F[j] < F[worst_idx]):
                    child_is_non_dom = False
                    break
            else:
                child_is_non_dom = True
                
            if child_is_non_dom:
                step_size = min(max_step, step_size * 1.1)
            else:
                step_size = max(min_step, step_size * 0.9)

        return X

    def _get_non_dominated(self, F):
        n = len(F)
        non_dom = []
        for i in range(n):
            is_dom = False
            for j in range(n):
                if i == j: continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom = True
                    break
            if not is_dom:
                non_dom.append(i)
        return non_dom
# EVOLVE-BLOCK-END
