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
        N = 30  # Population size - matched with inspiration for more generations
        F = 0.7 # Fixed F, balanced
        CR = 0.9 # High crossover rate
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
            # Filter points strictly better than ref
            valid_points = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if not valid_points:
                return 0.0
            sorted_pts = sorted(valid_points, key=lambda p: p[0])
            hv = 0.0
            prev_f2 = ref[1]
            for p in sorted_pts:
                f1, f2 = p
                # Since sorted by f1 asc, f1 < ref[0] is guaranteed for valid points
                # But we must ensure f2 < prev_f2 for the rectangle to have positive height
                if f2 < prev_f2:
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

        # Phase 2: Main DE loop
        total_generations = max(1, (budget - used) // N)
        gen = 0
        
        while used < budget:
            if time.time() > deadline:
                break
            
            gen += 1
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
            # Use a point slightly worse than the worst objective value in the current non-dominated set
            # to ensure all points are "inside" the reference box for HV calculation
            if len(non_dom_indices) > 0:
                ref_point = np.max(F_combined[non_dom_indices], axis=0) + 1.0
            else:
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

        # Phase 3: Polish - Local refinement around best non-dominated points
        if used < budget and time.time() < deadline - 5:
            nd_idx = get_non_dominated_indices(F_vals)
            if len(nd_idx) > 0:
                # Sort non-dominated points by their individual HV contribution (approximation: distance to worst)
                # Or just take the first few
                top_k = min(5, len(nd_idx))
                polish_indices = nd_idx[:top_k]
                
                remaining = budget - used
                # Each polish step tries 5 neighbors for 5 points = 25 evals
                # We have 'remaining' evals.
                n_polish_steps = min(remaining // 25, 3) # Max 3 steps of beam search
                
                for step in range(n_polish_steps):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    new_candidates_X = []
                    new_candidates_F = []
                    
                    for idx in polish_indices:
                        x_base = X[idx]
                        f_base = F_vals[idx]
                        
                        # Generate 5 neighbors
                        for _ in range(5):
                            if used >= budget:
                                break
                            # Small Gaussian perturbation
                            scale = 0.05 * (hi - lo)
                            x_new = x_base + scale * rng.standard_normal(dim)
                            x_new = np.clip(x_new, lo, hi)
                            f_new = problem(x_new)
                            used += 1
                            new_candidates_X.append(x_new)
                            new_candidates_F.append(f_new)
                    
                    if not new_candidates_X:
                        break
                    
                    new_candidates_X = np.array(new_candidates_X)
                    new_candidates_F = np.array(new_candidates_F)
                    
                    # Combine with current population and select best N
                    F_combined = np.vstack([F_vals, new_candidates_F])
                    X_combined = np.vstack([X, new_candidates_X])
                    
                    n_total = len(F_combined)
                    non_dom_indices = get_non_dominated_indices(F_combined)
                    
                    ref_point = np.max(F_combined, axis=0) + 1.0
                    
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
                    
                    # Update polish indices for next step
                    nd_idx = get_non_dominated_indices(F_vals)
                    if len(nd_idx) > 0:
                        polish_indices = nd_idx[:top_k]

        return X
