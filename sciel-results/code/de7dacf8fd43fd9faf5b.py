"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using NSGA-II-like approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Population size: use a reasonable fraction of budget
        pop_size = 20
        if self.budget < 200:
            pop_size = max(5, self.budget // 10)
            
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])
        
        def non_dominated_sort(F_mat):
            """Non-dominated sort returning fronts as lists of indices."""
            n_pts = len(F_mat)
            fronts = []
            remaining = set(range(n_pts))
            
            while remaining:
                front = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i != j:
                            if np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                                dominated = True
                                break
                    if not dominated:
                        front.append(i)
                
                if not front:
                    front = list(remaining)
                
                fronts.append(front)
                remaining -= set(front)
                
            return fronts
        
        def crowding_distance(F_mat, indices):
            """Calculate crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            F_sub = F_mat[indices]
            
            for k in range(m):
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                
                if F_sub[order[-1], k] - F_sub[order[0], k] > 1e-10:
                    for i in range(1, len(order) - 1):
                        dist[order[i]] += (F_sub[order[i+1], k] - F_sub[order[i-1], k]) / \
                                           (F_sub[order[-1], k] - F_sub[order[0], k])
            
            return dist
        
        def select_parents(F_mat, n_select):
            """Tournament selection based on rank and crowding distance."""
            fronts = non_dominated_sort(F_mat)
            n_pts = len(F_mat)
            
            ranks = np.zeros(n_pts, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            cd = np.zeros(n_pts)
            for front in fronts:
                if len(front) > 0:
                    cd_sub = crowding_distance(F_mat, front)
                    for i, idx in enumerate(front):
                        cd[idx] = cd_sub[i]
            
            parents = []
            for _ in range(n_select):
                i, j = rng.integers(0, n_pts, size=2)
                if ranks[i] < ranks[j]:
                    parents.append(i)
                elif ranks[j] < ranks[i]:
                    parents.append(j)
                else:
                    if cd[i] > cd[j]:
                        parents.append(i)
                    else:
                        parents.append(j)
            
            return parents
        
        def crossover(p1, p2):
            """SBX crossover."""
            eta = 20.0
            u = rng.random(n)
            c1 = np.zeros(n)
            c2 = np.zeros(n)
            
            for i in range(n):
                if u[i] <= 0.5:
                    beta = (2 * u[i]) ** (1.0 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u[i]))) ** (1.0 / (eta + 1))
                
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            return c1, c2
        
        def mutate(x, eta=20.0):
            """Polynomial mutation."""
            u = rng.random(n)
            x_mut = x.copy()
            
            for i in range(n):
                if u[i] < 0.5:
                    delta = (2 * u[i]) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1.0 / (eta + 1))
                
                x_mut[i] += delta * (hi[i] - lo[i])
            
            return np.clip(x_mut, lo, hi)
        
        def refine_around(x_center, f_center, budget_for_refine, step_scale=0.05):
            """
            Refine around a center point, maintaining a small diverse archive.
            Returns list of (x, f) pairs that are potentially useful.
            """
            if budget_for_refine < 1:
                return []
            
            results = [(x_center.copy(), f_center.copy())]
            step_size = step_scale * (hi - lo)
            used_here = 0
            
            # Adaptive step: start moderate, shrink as we find improvements
            for iteration in range(min(budget_for_refine, 20)):
                if used_here >= budget_for_refine or time.time() > deadline - 2:
                    break
                
                # Generate several candidates per iteration
                n_candidates = min(3, budget_for_refine - used_here)
                for _ in range(n_candidates):
                    if used_here >= budget_for_refine or time.time() > deadline - 2:
                        break
                    
                    # Random perturbation with adaptive step
                    delta = (rng.random(n) * 2 - 1) * step_size * (0.5 + rng.random())
                    new_x = np.clip(x_center + delta, lo, hi)
                    new_f = problem(new_x)
                    used_here += 1
                    
                    # Add to results if not dominated by existing results
                    is_dominated = False
                    for _, f_existing in results:
                        if np.all(f_existing <= new_f) and np.any(f_existing < new_f):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        # Remove any results dominated by new_f
                        results = [(x_r, f_r) for x_r, f_r in results 
                                   if not (np.all(new_f <= f_r) and np.any(new_f < f_r))]
                        results.append((new_x, new_f))
                    
                    # Shrink step if we found a non-dominated point
                    if not is_dominated:
                        step_size *= 0.85
                    else:
                        step_size *= 0.95
                
                if step_size * (hi - lo).max() < 1e-4:
                    break
            
            return results, used_here
        
        # Main NSGA-II loop
        while used < self.budget:
            if time.time() > deadline - 10:
                break
            
            # Check if we should switch to refinement mode
            # Reserve last 30% of budget for refinement
            refine_threshold = int(self.budget * 0.7)
            
            if used >= refine_threshold:
                # Switch to refinement mode
                break
            
            # Select parents
            parents = select_parents(F, 2)
            p1, p2 = parents
            
            # Crossover and mutation
            if rng.random() < 0.85:
                c1, c2 = crossover(X[p1], X[p2])
            else:
                c1 = X[p1].copy()
                c2 = X[p2].copy()
            
            # Adaptive mutation: use larger eta early, smaller later for finer search
            progress = used / self.budget
            eta_mut = 20.0 - 10.0 * progress  # From 20 to 10
            
            c1 = mutate(c1, eta_mut)
            c2 = mutate(c2, eta_mut)
            
            # Evaluate children
            f1 = problem(c1)
            used += 1
            f2 = problem(c2)
            used += 1
            
            # Combine and select
            X_comb = np.vstack([X, [c1, c2]])
            F_comb = np.vstack([F, [f1, f2]])
            
            fronts = non_dominated_sort(F_comb)
            
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    remaining_slots = pop_size - len(X_new)
                    cd = crowding_distance(F_comb, front)
                    order = np.argsort(-cd)
                    for i in order[:remaining_slots]:
                        idx = front[i]
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
        
        # Final refinement phase: use remaining budget to polish the Pareto front
        remaining_budget = self.budget - used
        
        if remaining_budget > 0 and time.time() < deadline - 5:
            # Get current non-dominated points
            fronts = non_dominated_sort(F)
            if fronts:
                nd_indices = fronts[0]
                
                # Sort non-dominated points by their crowding distance (most isolated first)
                cd = crowding_distance(F, nd_indices)
                nd_sorted = sorted(nd_indices, key=lambda i: -cd[nd_indices.index(i)])
                
                # Allocate refinement budget proportionally
                n_nd = len(nd_sorted)
                base_budget_per_point = max(2, remaining_budget // max(1, n_nd))
                
                # Refine each non-dominated point
                for idx in nd_sorted:
                    if used >= self.budget or time.time() > deadline - 3:
                        break
                    
                    # Give more budget to points with higher crowding distance
                    weight = cd[nd_indices.index(idx)] / (np.sum(cd) + 1e-10)
                    budget_for_this = max(2, int(base_budget_per_point * (0.5 + weight * n_nd)))
                    budget_for_this = min(budget_for_this, self.budget - used)
                    
                    if budget_for_this < 1:
                        continue
                    
                    results, used_here = refine_around(X[idx], F[idx], budget_for_this)
                    used += used_here
                    
                    # Add new points to population
                    for x_new, f_new in results:
                        X = np.vstack([X, [x_new]])
                        F = np.vstack([F, [f_new]])
                
                # Final selection to maintain diversity
                if len(X) > pop_size:
                    fronts = non_dominated_sort(F)
                    X_new = []
                    F_new = []
                    for front in fronts:
                        if len(X_new) + len(front) <= pop_size:
                            for idx in front:
                                X_new.append(X[idx])
                                F_new.append(F[idx])
                        else:
                            remaining_slots = pop_size - len(X_new)
                            cd = crowding_distance(F, front)
                            order = np.argsort(-cd)
                            for i in order[:remaining_slots]:
                                idx = front[i]
                                X_new.append(X[idx])
                                F_new.append(F[idx])
                            break
                    
                    X = np.array(X_new)
                    F = np.array(F_new)
        
        return X
# EVOLVE-BLOCK-END
