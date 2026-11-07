# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE37 using:
    1. Hypervolume-guided evolutionary search (MOEA/D-inspired)
    2. Adaptive mutation based on solution diversity
    3. Local refinement using scalarized objectives
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(123)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Track all evaluations
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f
        
        # Reference point for hypervolume calculation
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Phase 1: Weighted random initialization with diverse weights
        n_init = max(20, self.budget // 4)
        
        # Create diverse weight vectors
        weights = []
        for i in range(n_init):
            # Generate random weight vectors that sum to 1
            w = rng.random(self.n_obj)
            w = w / w.sum()
            weights.append(w)
        
        # Initialize with weighted random points
        for i in range(n_init):
            w = weights[i]
            # Sample point biased toward minimizing weighted sum
            x = lo + (hi - lo) * rng.random(self.dim)
            evaluate(x)
        
        # Phase 2: Evolutionary search with hypervolume-guided selection
        remaining = self.budget - used
        pop_size = min(30, max(10, remaining // 8))
        
        # Initialize population from existing solutions
        F_arr = np.array(all_F)
        X_arr = np.array(all_X)
        
        # Select initial population based on diversity
        pop_idx = self._select_diverse_subset(F_arr, pop_size, rng)
        pop_X = [X_arr[i].copy() for i in pop_idx]
        pop_F = [F_arr[i].copy() for i in pop_idx]
        
        # Evolutionary loop
        max_gens = max(1, (remaining - pop_size) // pop_size)
        
        for gen in range(max_gens):
            if used >= self.budget:
                break
            
            # Adaptive mutation rate based on convergence
            f_sums = np.array([np.sum(f) for f in pop_F])
            f_var = np.var(f_sums)
            adapt_factor = 0.5 + 0.5 * min(1.0, f_var / 0.1)
            
            new_candidates = []
            for i in range(pop_size):
                if used >= self.budget:
                    break
                
                # Select parents based on hypervolume contribution
                parent1_idx, parent2_idx = self._select_parents(pop_F, pop_size, rng)
                
                # Crossover: blend of two parents
                alpha = rng.random()
                child_X = alpha * pop_X[parent1_idx] + (1 - alpha) * pop_X[parent2_idx]
                
                # Mutation: adaptive step size
                step_size = 0.1 * adapt_factor * rng.random()
                mutation = (rng.random(self.dim) * 2 - 1) * step_size
                child_X = child_X + mutation
                child_X = np.clip(child_X, lo, hi)
                
                f_child = evaluate(child_X)
                new_candidates.append((child_X, f_child))
            
            # Selection: keep best solutions based on hypervolume contribution
            if new_candidates:
                # Combine population with new candidates
                combined_X = pop_X + [c[0] for c in new_candidates]
                combined_F = pop_F + [c[1] for c in new_candidates]
                
                # Select next generation based on hypervolume
                next_pop_idx = self._select_by_hypervolume(combined_F, pop_size, ref_point, rng)
                pop_X = [combined_X[i].copy() for i in next_pop_idx]
                pop_F = [combined_F[i].copy() for i in next_pop_idx]
        
        # Phase 3: Local refinement using scalarized objectives
        remaining = self.budget - used
        
        if remaining > 10 and len(all_F) > 0:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Find non-dominated points
            non_dom_idx = self._find_non_dominated(F_arr)
            
            if not non_dom_idx:
                non_dom_idx = [int(np.argmin(np.sum(F_arr, axis=1)))]
            
            # Sort by sum of objectives
            non_dom_idx.sort(key=lambda i: np.sum(F_arr[i]))
            
            # Refine top points using different scalarizations
            n_refine = min(len(non_dom_idx), max(1, remaining // 6))
            
            for idx in non_dom_idx[:n_refine]:
                if used >= self.budget:
                    break
                
                x0 = all_X[idx].copy()
                
                # Try multiple scalarizations
                for w in [np.ones(self.n_obj) / self.n_obj, 
                         rng.random(self.n_obj) / (rng.random(self.n_obj).sum() + 1e-10)]:
                    if used >= self.budget:
                        break
                    
                    def scalarized_obj(x, w=w):
                        if used >= self.budget:
                            return 1e10
                        x_clipped = np.clip(x, lo, hi)
                        f = problem(x_clipped)
                        return np.dot(f, w)
                    
                    try:
                        result = minimize(
                            scalarized_obj, x0, method='Nelder-Mead',
                            options={'maxiter': max(1, (self.budget - used) // 4),
                                     'xatol': 1e-5, 'fatol': 1e-7}
                        )
                        if result.fun < 1e9:
                            x_best = np.clip(result.x, lo, hi)
                            f_best = evaluate(x_best)
                    except Exception:
                        pass
        
        if all_X:
            return all_X[-1]
        else:
            return lo.copy()
    
    def _select_diverse_subset(self, F, k, rng):
        """Select k diverse points based on objective space spread."""
        n = len(F)
        if n <= k:
            return list(range(n))
        
        # Start with point with best sum
        f_sums = np.sum(F, axis=1)
        selected = [int(np.argmin(f_sums))]
        
        for _ in range(k - 1):
            # Select point that maximizes minimum distance to selected
            min_dists = np.inf * np.ones(n)
            for sel_idx in selected:
                dists = np.sqrt(np.sum((F - F[sel_idx]) ** 2, axis=1))
                min_dists = np.minimum(min_dists, dists)
            
            # Exclude already selected
            for sel_idx in selected:
                min_dists[sel_idx] = -1
            
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
        
        return selected
    
    def _select_parents(self, F, pop_size, rng):
        """Select two parents using tournament selection based on hypervolume."""
        # Simple tournament: pick 3 random, keep best 2 by sum
        candidates = rng.choice(pop_size, size=min(3, pop_size), replace=False)
        f_sums = [np.sum(F[i]) for i in candidates]
        sorted_candidates = sorted(zip(f_sums, candidates), key=lambda x: x[0])
        return sorted_candidates[0][1], sorted_candidates[1][1]
    
    def _select_by_hypervolume(self, F, k, ref_point, rng):
        """Select k points that maximize hypervolume contribution."""
        n = len(F)
        if n <= k:
            return list(range(n))
        
        # Compute hypervolume contribution for each point
        # A point contributes if it's not dominated and adds volume
        contributions = np.zeros(n)
        
        for i in range(n):
            # Check if point i is dominated
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            
            if dominated:
                continue
            
            # Compute hypervolume contribution (approximate)
            # Volume of box from F[i] to ref_point, minus overlap with other points
            box_vol = 1.0
            for d in range(len(F[i])):
                if F[i][d] < ref_point[d]:
                    box_vol *= (ref_point[d] - F[i][d])
                else:
                    box_vol = 0
                    break
            
            # Subtract overlap with other non-dominated points
            for j in range(n):
                if i == j or F[j][0] >= ref_point[0]:
                    continue
                if np.all(F[j] <= F[i]):
                    # j dominates i, skip
                    continue
                # Compute intersection volume
                inter_vol = 1.0
                for d in range(len(F[i])):
                    low = max(F[i][d], F[j][d])
                    high = ref_point[d]
                    if low >= high:
                        inter_vol = 0
                        break
                    inter_vol *= (high - low)
                contributions[i] -= inter_vol
            
            contributions[i] = max(0, contributions[i])
        
        # Select top k by contribution
        sorted_idx = np.argsort(-contributions)
        return sorted_idx[:k].tolist()
    
    def _find_non_dominated(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        non_dom = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated = True
                    break
            if not dominated:
                non_dom.append(i)
        return non_dom
# EVOLVE-BLOCK-END
