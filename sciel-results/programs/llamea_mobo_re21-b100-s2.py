import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Parameters
        N = 20  # Population size
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        X = lo + (hi - lo) * np.random.rand(N, self.dim)
        F_vals = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Check if a dominates b
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Calculate Hypervolume contribution of a point to a set of non-dominated points
        # Since n_obj is small (2), we can do this efficiently.
        # Reference point is [3000.0, 0.06] based on task description? 
        # Wait, the task says "reference point [3000.0, 0.06]". 
        # However, for internal selection, we should use a reference point that is worse than all current points.
        # A dynamic reference point is safer: max of all points + epsilon.
        
        def hypervolume_2d(points, ref):
            """
            points: list of (f1, f2) tuples, non-dominated.
            ref: (r1, r2)
            Returns hypervolume.
            """
            if not points:
                return 0.0
            # Sort by f1 ascending
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_f2 = ref[1]
            for p in sorted_pts:
                f1, f2 = p
                if f1 < ref[0] and f2 < ref[1]:
                    hv += (ref[0] - f1) * (prev_f2 - f2)
                    prev_f2 = f2
            return hv

        def get_non_dominated(points):
            """
            points: list of indices or tuples. Here we use list of (f1, f2).
            Returns list of non-dominated points.
            """
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

        # Main Loop
        while used < self.budget:
            # Generate offspring
            U = np.zeros_like(X)
            U_vals = np.zeros_like(F_vals)
            
            # We can generate up to min(N, budget - used) offspring
            num_offspring = min(N, self.budget - used)
            
            for i in range(num_offspring):
                # Pick 3 distinct random indices
                r1, r2, r3 = np.random.choice(N, 3, replace=False)
                
                # Differential Mutation
                v = X[r1] + F * (X[r2] - X[r3])
                
                # Binomial Crossover
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() <= CR or j == j_rand:
                        U[i, j] = v[j]
                    else:
                        U[i, j] = X[i, j]
                
                # Clip to bounds
                U[i] = np.clip(U[i], lo, hi)
                
                # Evaluate
                U_vals[i] = problem(U[i])
                used += 1
                
            # If we ran out of budget, break
            if used >= self.budget:
                break

            # Combine current population and offspring
            # We need to select the best N from the combined 2*N (or N + num_offspring)
            # We use Greedy Hypervolume Selection.
            
            # Create a list of (f_vec, x_vec) for all candidates
            # Note: We only need to keep the best N for the next generation's X.
            # The "archive" for hypervolume calculation is just the non-dominated set of the candidates.
            
            # 1. Identify non-dominated points from the combined set
            # Combined F values
            F_combined = np.vstack([F_vals, U_vals])
            X_combined = np.vstack([X, U])
            
            # Get non-dominated points (indices in F_combined)
            n_total = len(F_combined)
            non_dom_indices = []
            for i in range(n_total):
                is_dom = False
                for j in range(n_total):
                    if i != j and dominates(F_combined[j], F_combined[i]):
                        is_dom = True
                        break
                if not is_dom:
                    non_dom_indices.append(i)
            
            # We want to select N points from non_dom_indices that maximize hypervolume.
            # Greedy approach:
            # 1. Start with empty selected set.
            # 2. Find the point in non_dom_indices that has the highest individual contribution to HV when added to the empty set?
            #    Actually, standard greedy HV selection:
            #    - Maintain a set of selected points.
            #    - Iteratively add the point that maximizes the HV of the non-dominated set of (selected + candidate).
            #    This is O(K^2 * N) where K is number of non-dominated points.
            
            # Optimization: 
            # Since N=20, and non-dominated set is likely small (<20), this is fast.
            
            selected_indices = []
            remaining_indices = non_dom_indices[:]
            
            # Reference point for HV calculation: 
            # Use a point slightly worse than the worst objective value in the current non-dominated set.
            # Or just use the reference point from the task [3000, 0.06] if it's safe?
            # The task ref is [3000, 0.06]. If our points are better, this works.
            # But for internal selection, a dynamic ref is better to ensure we don't bias towards the task ref if the search is far away.
            # Let's use max(F_combined) + 1 as dynamic ref.
            ref_point = np.max(F_combined, axis=0) + 1.0
            
            # Greedy Selection
            for _ in range(min(N, len(remaining_indices))):
                if not remaining_indices:
                    break
                
                best_idx = -1
                best_hv = -1.0
                
                # Current non-dominated set of selected points
                current_selected_pts = [F_combined[i] for i in selected_indices]
                
                for idx in remaining_indices:
                    # Candidate set: selected + idx
                    candidate_pts = current_selected_pts + [F_combined[idx]]
                    # Get non-dominated subset of candidate_pts
                    nd_pts = get_non_dominated(candidate_pts)
                    # Calculate HV
                    hv_val = hypervolume_2d(nd_pts, ref_point)
                    
                    if hv_val > best_hv:
                        best_hv = hv_val
                        best_idx = idx
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
            
            # If we have fewer than N non-dominated points, we need to fill the rest.
            # This case is rare in early generations but possible.
            # We can just pick random ones from the full combined set if needed, or duplicate the best.
            # For simplicity, if we don't have enough, we keep the existing X for the remaining slots?
            # Actually, the loop above selects up to N. If len(non_dom_indices) < N, we select all non-dom.
            # We need exactly N points for the next population.
            
            if len(selected_indices) < N:
                # Fill with random points from the combined set that were not selected
                # This ensures diversity if the front is very small
                selected_set = set(selected_indices)
                for i in range(n_total):
                    if len(selected_indices) >= N:
                        break
                    if i not in selected_set:
                        selected_indices.append(i)
                        selected_set.add(i)
                
                # If still not enough (should not happen as n_total >= N), duplicate last
                while len(selected_indices) < N:
                    selected_indices.append(selected_indices[-1])

            # Update X and F_vals
            X = X_combined[selected_indices]
            F_vals = F_combined[selected_indices]

        return X
