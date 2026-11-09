import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0 # 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Population size
        pop_size = 20
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Adaptive mutation step
        # Start large, end small
        max_step = 0.5
        min_step = 0.01
        total_iterations = (self.budget - pop_size) // 2
        iter_count = 0
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Identify non-dominated set
            # Simple O(N^2) domination check for small N
            nd_indices = []
            for i in range(len(F)):
                is_dominated = False
                for j in range(len(F)):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_indices.append(i)
            
            if not nd_indices:
                nd_indices = list(range(len(F)))
                
            # Select parents from non-dominated set
            # If ND set is small, pick randomly from it.
            # To maintain diversity, we can pick randomly.
            p1_idx = rng.choice(nd_indices)
            p2_idx = rng.choice(nd_indices)
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            
            # SBX Crossover
            # eta = 15 is standard
            children = self._sbx_crossover(p1, p2, lo, hi, rng, eta=15)
            
            # Mutation
            # Linear decay of step size
            if total_iterations > 0:
                progress = iter_count / total_iterations
            else:
                progress = 1.0
            step = max_step - (max_step - min_step) * progress
            
            for i in range(len(children)):
                # Gaussian mutation
                noise = rng.standard_normal(self.dim) * step * (hi - lo)
                children[i] = np.clip(children[i] + noise, lo, hi)
            
            # Evaluate children
            F1 = problem(children[0])
            used += 1
            F2 = problem(children[1])
            used += 1
            
            # Update population
            # Simple replacement: replace the two worst points
            # Or better: maintain a pool and select best pop_size
            # Given small budget, let's replace the two points with the worst "sum" or dominated status
            # Actually, let's just append and prune to keep pop_size
            
            # Combine
            X_new = np.vstack([X, children])
            F_new = np.vstack([F, np.array([F1, F2])])
            
            # Prune to pop_size
            # 1. Get non-dominated set of X_new
            nd_new_indices = []
            for i in range(len(F_new)):
                is_dominated = False
                for j in range(len(F_new)):
                    if i == j: continue
                    if np.all(F_new[j] <= F_new[i]) and np.any(F_new[j] < F_new[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_new_indices.append(i)
            
            if len(nd_new_indices) >= pop_size:
                # Select pop_size from ND set based on diversity (distance)
                # Simple: pick random from ND, or use crowding
                # For simplicity and speed, let's just pick random from ND
                # But this might lose good points.
                # Better: Sort ND by distance to centroid?
                # Let's just keep the first pop_size ND indices found? No, random is safer for diversity.
                selected_nd = rng.choice(nd_new_indices, size=pop_size, replace=False)
                X = X_new[selected_nd]
                F = F_new[selected_nd]
            else:
                # Fill with best dominated points
                dominated_indices = [i for i in range(len(F_new)) if i not in nd_new_indices]
                # Sort dominated by sum of objectives (lower is better)
                dominated_indices.sort(key=lambda i: np.sum(F_new[i]))
                # Take first (pop_size - len(nd_new_indices))
                fill_count = pop_size - len(nd_new_indices)
                selected_dominated = dominated_indices[:fill_count]
                
                selected_indices = list(nd_new_indices) + selected_dominated
                X = X_new[selected_indices]
                F = F_new[selected_indices]
            
            iter_count += 1

        return X

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=15):
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        for i in range(len(p1)):
            if rng.random() < 0.5:
                # Do crossover
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1 = p1[i]
                    x2 = p2[i]
                    if x1 > x2:
                        x1, x2 = x2, x1
                    
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    c1[i] = 0.5 * ((x1 + x2) - beta * (x2 - x1))
                    c2[i] = 0.5 * ((x1 + x2) + beta * (x2 - x1))
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return [c1, c2]
