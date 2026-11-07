import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random

        # Parameters
        N = 40  # Population size - larger for better coverage
        F_start = 0.9
        F_end = 0.4
        CR = 0.85
        used = 0

        # Helper functions
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def get_non_dominated_indices(F_set):
            K = len(F_set)
            if K == 0:
                return []
            nd_indices = []
            for i in range(K):
                is_dom = False
                for j in range(K):
                    if i != j and dominates(F_set[j], F_set[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nd_indices.append(i)
            return nd_indices

        def hypervolume_2d(points, ref):
            if not points:
                return 0.0
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_f2 = ref[1]
            for p in sorted_pts:
                f1, f2 = p
                if f1 < ref[0] and f2 < ref[1]:
                    hv += (ref[0] - f1) * (prev_f2 - f2)
                    prev_f2 = f2
            return hv

        def get_non_dominated_points(points):
            non_dominated = []
            for i, p in enumerate(points):
                is_dominated = False
                for j, q in enumerate(points):
                    if i != j and dominates(q, p):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(p)
            return non_dominated

        # Phase 1: Latin Hypercube initialization
        n_init = min(N, budget)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            fractions = (perm + rng.random(n_init)) / n_init
            X[:, d] = lo[d] + fractions * (hi[d] - lo[d])
        
        F_vals = np.array([problem(x) for x in X])
        used = n_init

        # Phase 2: Main DE loop with adaptive mutation
        total_generations = max(1, (budget - used) // N)
        gen = 0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            gen += 1
            # Adaptive F: start high for exploration, decrease for exploitation
            progress = min(1.0, gen / total_generations)
            F = F_start + (F_end - F_start) * progress
            
            num_offspring = min(N, budget - used)
            U = np.zeros((num_offspring, dim))
            U_vals = np.zeros((num_offspring, n_obj))
            
            for i in range(num_offspring):
                if time.time() > deadline:
                    break
                    
                # Pick 3 distinct random indices
                r1, r2, r3 = rng.choice(N, 3, replace=False)
                
                # Differential Mutation
                v = X[r1] + F * (X[r2] - X[r3])
                
                # Binomial Crossover
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() <= CR or j == j_rand:
                        U[i, j] = v[j]
                    else:
                        U[i, j] = X[i, j]
                
                # Clip to bounds
                U[i] = np.clip(U[i], lo, hi)
                
                # Evaluate
                U_vals[i] = problem(U[i])
                used += 1
                
            if used >= budget or time.time() > deadline:
                break

            # Combine and select using hypervolume-based greedy selection
            F_combined = np.vstack([F_vals, U_vals[:num_offspring]])
            X_combined = np.vstack([X, U[:num_offspring]])
            
            n_total = len(F_combined)
            non_dom_indices = get_non_dominated_indices(F_combined)
            
            # Dynamic reference point for selection
            ref_point = np.max(F_combined, axis=0) + 1.0
            
            # Greedy HV selection
            selected_indices = []
            remaining_indices = non_dom_indices[:]
            
            for _ in range(min(N, len(remaining_indices))):
                if not remaining_indices:
                    break
                if time.time() > deadline:
                    break

                best_idx = -1
                best_hv = -1.0
                current_selected_pts = [F_combined[i] for i in selected_indices]
                
                for idx in remaining_indices:
                    candidate_pts = current_selected_pts + [F_combined[idx]]
                    nd_pts = get_non_dominated_points(candidate_pts)
                    hv_val = hypervolume_2d(nd_pts, ref_point)
                    
                    if hv_val > best_hv:
                        best_hv = hv_val
                        best_idx = idx
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
            
            if len(selected_indices) < N:
                selected_set = set(selected_indices)
                for i in range(n_total):
                    if len(selected_indices) >= N:
                        break
                    if i not in selected_set:
                        selected_indices.append(i)
                        selected_set.add(i)
                while len(selected_indices) < N:
                    selected_indices.append(selected_indices[-1])

            X = X_combined[selected_indices]
            F_vals = F_combined[selected_indices]

        # Phase 3: Polish - local refinement around best non-dominated points
        if used < budget and time.time() < deadline - 10:
            nd_idx = get_non_dominated_indices(F_vals)
            nd_points = [(X[i], F_vals[i]) for i in nd_idx]
            
            remaining = budget - used
            n_polish = min(remaining, 30)
            polish_scale = 0.02 * (hi - lo)
            
            for i in range(n_polish):
                if used >= budget or time.time() > deadline:
                    break
                if not nd_points:
                    break
                
                pt_idx = rng.randint(len(nd_points))
                x_base, f_base = nd_points[pt_idx]
                
                # Try a few perturbations
                for attempt in range(3):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    x_new = x_base + polish_scale * rng.standard_normal(dim)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    
                    # If this point is non-dominated, add to population
                    if not any(dominates(f, f_new) for f in F_vals):
                        # Replace a dominated point if possible
                        for j in range(len(F_vals)):
                            if dominates(f_new, F_vals[j]):
                                X[j] = x_new
                                F_vals[j] = f_new
                                break
                        else:
                            # If no dominated point, replace worst by HV contribution
                            # For simplicity, replace random
                            j = rng.randint(len(F_vals))
                            X[j] = x_new
                            F_vals[j] = f_new
                        break

        return X
