# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using a hybrid approach:
    1. Initial random sampling
    2. NSGA-II-style evolutionary search with crowding distance
    3. Local refinement using Nelder-Mead on promising solutions
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_eval = 0

        # Helper to evaluate and count
        def eval_point(x):
            nonlocal n_eval
            x = np.clip(x, lo, hi)
            f = problem(x)
            n_eval += 1
            return x, f

        # Phase 1: Initial random population
        pop_size = min(30, self.budget // 5)
        X = []
        F = []
        for _ in range(pop_size):
            x = lo + (hi - lo) * rng.random(self.dim)
            x, f = eval_point(x)
            X.append(x)
            F.append(f)
        
        X = np.array(X)
        F = np.array(F)

        # Phase 2: NSGA-II style evolution
        # We'll use a smaller population for efficiency
        evolve_pop_size = min(20, self.budget // 10)
        
        def dominates(a, b):
            """Check if a dominates b (minimization)"""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """NSGA-II non-dominated sorting"""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
            n_dominators = np.zeros(n, dtype=int)  # number of solutions that dominate i
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dominators[i] += 1
            
            # First front
            front = [i for i in range(n) if n_dominators[i] == 0]
            if not front:
                return [list(range(n))]
            fronts.append(front)
            
            while front:
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dominators[j] -= 1
                        if n_dominators[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                front = next_front
            
            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices"""
            m = len(indices)
            if m <= 2:
                return np.full(m, np.inf)
            
            dist = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(self.n_obj):
                # Sort by objective k
                order = np.argsort(F_sub[:, k])
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                min_val = F_sub[order[0], k]
                max_val = F_sub[order[-1], k]
                if max_val - min_val < 1e-12:
                    continue
                for i in range(1, m - 1):
                    dist[order[i]] += (F_sub[order[i + 1], k] - F_sub[order[i - 1], k]) / (max_val - min_val)
            
            return dist

        # Evolution loop
        while n_eval < self.budget:
            # Get non-dominated fronts
            fronts = non_dominated_sort(F)
            
            # Select parents from first few fronts
            parent_indices = []
            for front in fronts[:2]:
                parent_indices.extend(front)
            
            if not parent_indices:
                parent_indices = list(range(len(F)))
            
            # Create children
            n_children = min(evolve_pop_size, self.budget - n_eval)
            if n_children <= 0:
                break
            
            child_X = []
            child_F = []
            
            for _ in range(n_children):
                # Pick two parents from parent pool
                p1_idx = parent_indices[rng.integers(len(parent_indices))]
                p2_idx = parent_indices[rng.integers(len(parent_indices))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # Crossover: blend
                alpha = rng.random()
                child = alpha * p1 + (1 - alpha) * p2
                
                # Mutation: Gaussian with adaptive sigma
                sigma = 0.15 * (hi - lo)
                child = child + sigma * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                x, f = eval_point(child)
                child_X.append(x)
                child_F.append(f)
            
            if not child_X:
                break
                
            child_X = np.array(child_X)
            child_F = np.array(child_F)
            
            # Combine parents and children
            all_X = np.vstack([X, child_X])
            all_F = np.vstack([F, child_F])
            
            # Select next generation using non-dominated sorting + crowding
            fronts = non_dominated_sort(all_F)
            next_indices = []
            
            for front in fronts:
                if len(next_indices) + len(front) <= evolve_pop_size:
                    next_indices.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    need = evolve_pop_size - len(next_indices)
                    cd = crowding_distance(all_F, front)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    selected = [front[i] for i in order[:need]]
                    next_indices.extend(selected)
                    break
            
            X = all_X[next_indices]
            F = all_F[next_indices]
            
            if n_eval >= self.budget:
                break

        # Phase 3: Local refinement using Nelder-Mead on best solutions
        if n_eval < self.budget:
            # Get non-dominated solutions
            fronts = non_dominated_sort(F)
            nd_indices = fronts[0] if fronts else list(range(len(F)))
            
            # Pick a few promising solutions for local optimization
            n_local = min(3, len(nd_indices), (self.budget - n_eval) // 15)
            
            for idx in nd_indices[:n_local]:
                if n_eval >= self.budget:
                    break
                
                x0 = X[idx].copy()
                f0 = F[idx].copy()
                
                # Simple local search: try small perturbations in each dimension
                step_size = 0.02 * (hi - lo)
                for dim_i in range(self.dim):
                    if n_eval >= self.budget:
                        break
                    
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        if n_eval >= self.budget:
                            break
                        
                        x_trial = x0.copy()
                        x_trial[dim_i] += direction * step_size
                        x_trial = np.clip(x_trial, lo, hi)
                        
                        x_trial, f_trial = eval_point(x_trial)
                        
                        # Accept if it's better in at least one objective and not worse in others
                        # Or if it's non-dominated by current
                        if np.any(f_trial < f0) and np.all(f_trial <= f0 + 1e-8):
                            # Check if it dominates any existing solution
                            dominated_any = False
                            for j in range(len(F)):
                                if dominates(f_trial, F[j]):
                                    dominated_any = True
                                    break
                            
                            if dominated_any:
                                x0 = x_trial
                                f0 = f_trial
                                X[idx] = x0
                                F[idx] = f0

        return X
# EVOLVE-BLOCK-END
