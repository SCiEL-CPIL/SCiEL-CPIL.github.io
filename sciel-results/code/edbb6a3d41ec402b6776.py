"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Hypervolume-guided Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        ref_point = np.array([3000.0, 0.06])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = np.array(problem(x), dtype=np.float64)
            except Exception:
                return None
            if not np.all(np.isfinite(f)):
                f = np.array([1e10, 1e10])
            all_X.append(x)
            all_F.append(f)
            used += 1
            return f

        # Phase 1: Initialize with a diverse set of points
        # Use a mix of random and structured points
        init_points = []
        
        # Corner points
        corners = []
        for i in range(2**dim):
            corner = np.zeros(dim)
            for j in range(dim):
                corner[j] = hi[j] if (i >> j) & 1 else lo[j]
            corners.append(corner)
        init_points.extend(corners)
        
        # Midpoints
        mid = (lo + hi) / 2
        init_points.append(mid)
        
        # Random points
        n_random = min(15, budget - len(init_points))
        for _ in range(n_random):
            x = lo + span * np.random.random(dim)
            init_points.append(x)
        
        # Evaluate initial points
        pop_X = []
        pop_F = []
        
        for x in init_points:
            f = eval_point(x)
            if f is None:
                break
            pop_X.append(x)
            pop_F.append(f)
        
        if len(pop_X) == 0:
            return all_X
        
        pop_X = np.array(pop_X)
        pop_F = np.array(pop_F)

        def get_non_dominated_indices(F):
            if len(F) == 0:
                return np.array([], dtype=int)
            n = len(F)
            le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
            lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
            dominated = np.any(le & lt, axis=0)
            return np.where(~dominated)[0]

        def compute_hv(F_nd, ref):
            """Compute hypervolume of non-dominated set F_nd with reference ref."""
            if len(F_nd) == 0:
                return 0.0
            # Sort by first objective
            idx = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[idx]
            
            hv = 0.0
            for i in range(len(F_sorted)):
                if i == len(F_sorted) - 1:
                    f1_next = ref[0]
                else:
                    f1_next = F_sorted[i + 1, 0]
                width = f1_next - F_sorted[i, 0]
                height = ref[1] - F_sorted[i, 1]
                if width > 0 and height > 0:
                    hv += width * height
            return hv

        def find_best_improvement_direction(F_nd, ref, pop_X, pop_F):
            """
            Find the direction in objective space that would give the best HV improvement.
            Returns a target point in decision space to search towards.
            """
            if len(F_nd) < 2:
                # If we have fewer than 2 points, just return a random point
                return lo + span * np.random.random(dim)
            
            # Sort by first objective
            idx = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[idx]
            
            # Find the gap that would give the best HV improvement
            # For each pair of consecutive points, the "gap" is the rectangle
            # that could be filled
            best_hv_gain = 0
            best_target = None
            best_source_idx = None
            
            for i in range(len(F_sorted)):
                if i == len(F_sorted) - 1:
                    f1_next = ref[0]
                else:
                    f1_next = F_sorted[i + 1, 0]
                
                width = f1_next - F_sorted[i, 0]
                if width <= 0:
                    continue
                
                # The ideal point in this gap would be at (F_sorted[i,0], F_sorted[i,1])
                # but we want to improve the front, so we want a point that is
                # better than F_sorted[i] in objective 2 while keeping objective 1 similar
                # Or better than the next point in objective 1 while keeping objective 2 similar
                
                # Strategy: Try to find a point that dominates the current front in the gap
                # The "target" in objective space is (F_sorted[i,0], F_sorted[i,1] - epsilon)
                # or (F_sorted[i+1,0] - epsilon, F_sorted[i,1])
                
                # For now, let's just return a point that is the average of the two neighbors
                # in objective space, mapped back to decision space via interpolation
                
                if i < len(F_sorted) - 1:
                    # Interpolate between point i and i+1
                    # This is a rough heuristic
                    t = 0.5
                    target_f = (1 - t) * F_sorted[i] + t * F_sorted[i + 1]
                else:
                    target_f = F_sorted[i].copy()
                    target_f[0] = F_sorted[i, 0]  # Keep first objective
                    target_f[1] = ref[1]  # Try to minimize second objective
                
                # Map target_f back to decision space
                # We need to find a decision vector that produces target_f
                # This is an inverse problem, which is hard
                # Instead, let's just return a random point in the region
                # that is "between" the two source points in decision space
                
                # Get the indices of the two source points
                src_i_idx = idx[i]
                if i < len(F_sorted) - 1:
                    src_j_idx = idx[i + 1]
                    # Interpolate in decision space
                    t = 0.5
                    target_x = (1 - t) * pop_X[src_i_idx] + t * pop_X[src_j_idx]
                else:
                    target_x = pop_X[src_i_idx].copy()
                    # Perturb towards the reference direction
                    # We don't know the mapping, so just perturb randomly
                    target_x = target_x + 0.1 * span * (np.random.random(dim) - 0.5)
                    target_x = np.clip(target_x, lo, hi)
                
                # Estimate HV gain
                # This is a rough estimate
                est_gain = width * (ref[1] - target_f[1]) * 0.5
                if est_gain > best_hv_gain:
                    best_hv_gain = est_gain
                    best_target = target_x
                    best_source_idx = src_i_idx
            
            if best_target is None:
                return lo + span * np.random.random(dim)
            
            return best_target

        # Phase 2: Local search guided by hypervolume improvement
        # We'll do a series of local searches around promising points
        
        # Keep track of the best non-dominated set
        current_nd = get_non_dominated_indices(pop_F)
        
        # Main loop: alternate between global exploration and local refinement
        exploration_budget = min(30, budget - used)
        for _ in range(exploration_budget):
            if used >= budget or time.time() > deadline:
                break
            
            # Generate a random point
            x = lo + span * np.random.random(dim)
            f = eval_point(x)
            if f is None:
                break
            pop_X = np.vstack([pop_X, [x]])
            pop_F = np.vstack([pop_F, [f]])
        
        # Now do local search
        # For each point in the current front, do a local search
        # to try to improve the hypervolume
        
        # Get current non-dominated set
        nd_idx = get_non_dominated_indices(pop_F)
        if len(nd_idx) == 0:
            return all_X
        
        # Sort by first objective for consistent ordering
        nd_idx_sorted = nd_idx[np.argsort(pop_F[nd_idx, 0])]
        
        # For each point in the front, do a local search
        # Use Nelder-Mead style search in decision space
        # Objective: maximize HV contribution of the new point
        
        # We'll do a simple local search: for each front point, try small perturbations
        # and keep the one that improves HV the most
        
        # First, let's define a function to compute the HV gain of adding a point
        def hv_gain_if_add(new_f, F_nd, ref):
            """Compute the HV gain if we add new_f to the non-dominated set F_nd."""
            # Create the new set
            new_set = np.vstack([F_nd, [new_f]])
            
            # Get non-dominated set
            nd_new = get_non_dominated_indices(new_set)
            
            if len(nd_new) == 0:
                return 0.0
            
            # Compute HV of new set
            hv_new = compute_hv(new_set[nd_new], ref)
            
            # Compute HV of old set
            hv_old = compute_hv(F_nd, ref)
            
            return hv_new - hv_old

        # Local search around each front point
        # Use a simple hill-climbing approach
        
        # Get the current non-dominated set
        F_nd_current = pop_F[nd_idx]
        X_nd_current = pop_X[nd_idx]
        
        # For each point, do a local search
        # We'll do this in a round-robin fashion
        
        # Define a local search function
        def local_search_around(x0, f0, F_nd, n_iterations=5):
            """
            Do a local search around x0 to try to improve HV.
            Returns the best point found.
            """
            best_x = x0.copy()
            best_f = f0.copy()
            best_hv_gain = hv_gain_if_add(f0, F_nd, ref_point)
            
            # Try small perturbations in each dimension
            step_size = 0.1 * span
            for iteration in range(n_iterations):
                if used >= budget or time.time() > deadline:
                    break
                
                improved = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        x_trial = best_x.copy()
                        x_trial[d] += direction * step_size[d] * (0.5 ** iteration)
                        x_trial = np.clip(x_trial, lo, hi)
                        
                        f_trial = eval_point(x_trial)
                        if f_trial is None:
                            continue
                        
                        # Check if this point is non-dominated
                        # and if it improves HV
                        # First, check if it dominates any point in F_nd
                        # or is non-dominated
                        
                        # Compute HV gain
                        hv_gain = hv_gain_if_add(f_trial, F_nd, ref_point)
                        
                        if hv_gain > best_hv_gain + 1e-10:
                            best_x = x_trial
                            best_f = f_trial
                            best_hv_gain = hv_gain
                            improved = True
                
                if not improved:
                    break
            
            return best_x, best_f, best_hv_gain

        # Do local search around each front point
        # We'll do this in a round-robin fashion, alternating between points
        
        # Get the current non-dominated set indices
        # We'll keep a list of "active" points to search around
        
        # For simplicity, let's just do a fixed number of local searches
        # around the most promising points
        
        # Identify the most promising points: those with high HV contribution
        # Compute HV contributions
        def compute_hv_contributions(F_nd, ref):
            if len(F_nd) == 0:
                return np.array([])
            
            idx = np.argsort(F_nd[:, 0])
            F_sorted = F_nd[idx]
            
            contributions = np.zeros(len(F_sorted))
            
            f1_next = np.empty(len(F_sorted))
            f1_next[:-1] = F_sorted[1:, 0]
            f1_next[-1] = ref[0]
            
            widths = f1_next - F_sorted[:, 0]
            heights = ref[1] - F_sorted[:, 1]
            
            contributions = widths * heights
            
            inv_idx = np.empty(len(idx), dtype=int)
            inv_idx[idx] = np.arange(len(idx))
            contrib_original = np.zeros(len(F_nd))
            contrib_original[inv_idx] = contributions
            
            return contrib_original

        # Compute contributions for current front
        contribs = compute_hv_contributions(F_nd_current, ref_point)
        
        # Sort by contribution (descending)
        order = np.argsort(contribs)[::-1]
        
        # Do local search around the top points
        n_local_searches = min(10, budget - used)
        for i in range(n_local_searches):
            if used >= budget or time.time() > deadline:
                break
            
            # Pick the point with the highest contribution
            point_idx = order[i % len(order)]
            
            x0 = X_nd_current[point_idx]
            f0 = F_nd_current[point_idx]
            
            # Do local search
            best_x, best_f, best_gain = local_search_around(x0, f0, F_nd_current, n_iterations=3)
            
            # If we found an improvement, add it to the population
            if best_gain > 1e-10:
                pop_X = np.vstack([pop_X, [best_x]])
                pop_F = np.vstack([pop_F, [best_f]])
            
            # Update the non-dominated set
            nd_idx = get_non_dominated_indices(pop_F)
            if len(nd_idx) == 0:
                break
            F_nd_current = pop_F[nd_idx]
            X_nd_current = pop_X[nd_idx]
            
            # Recompute contributions
            contribs = compute_hv_contributions(F_nd_current, ref_point)
            order = np.argsort(contribs)[::-1]

        return all_X
# EVOLVE-BLOCK-END
