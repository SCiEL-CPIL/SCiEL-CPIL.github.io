import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Define islands based on objective weights
        # We want to cover the Pareto front.
        # WFG4 is a multi-objective problem.
        # Let's use 6 islands with specific weight profiles.
        # Weights should sum to 1.
        weights_list = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4],
        ]
        
        n_islands = len(weights_list)
        # Budget is 200. 
        # If we use 6 islands, we can have small populations.
        # Let's aim for pop_size such that total_init + mutations <= budget.
        # Let's say pop_size = 5. Total init = 30. Remaining = 170 mutations.
        # This allows for decent evolution.
        pop_size = 5
        
        # Ensure we don't exceed budget
        total_init = n_islands * pop_size
        if total_init >= budget:
            # Reduce pop size
            pop_size = max(1, budget // (n_islands * 2))
            total_init = n_islands * pop_size
            if total_init >= budget:
                pop_size = 1
                total_init = n_islands

        # Initialize islands
        islands = []
        for i in range(n_islands):
            X = lo + (hi - lo) * np.random.rand(pop_size, dim)
            F = np.array([problem(x) for x in X])
            islands.append({'X': X, 'F': F, 'weights': weights_list[i]})
            
        used = total_init
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Helper to calculate 3D Hypervolume
        def calc_hv_3d(F, ref):
            """
            Calculate Hypervolume for 3D objective vectors.
            F: (N, 3) array
            ref: (3,) array
            """
            if len(F) == 0:
                return 0.0
            
            # Filter out points that are not strictly better than ref in all objectives
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            if len(F) == 0:
                return 0.0
            if len(F) == 1:
                f = F[0]
                return (ref[0]-f[0]) * (ref[1]-f[1]) * (ref[2]-f[2])
                
            # Remove dominated points
            # O(N^2)
            nd_idx = []
            n = len(F)
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j:
                        if F[j][0] <= F[i][0] and F[j][1] <= F[i][1] and F[j][2] <= F[i][2] and (F[j][0] < F[i][0] or F[j][1] < F[i][1] or F[j][2] < F[i][2]):
                            dominated = True
                            break
                if not dominated:
                    nd_idx.append(i)
                    
            F_nd = F[nd_idx]
            if len(F_nd) == 0:
                return 0.0
            if len(F_nd) == 1:
                f = F_nd[0]
                return (ref[0]-f[0]) * (ref[1]-f[1]) * (ref[2]-f[2])
                
            # Sort by f1
            F_nd = F_nd[np.argsort(F_nd[:, 0])]
            
            # Sweep line in f1
            # The HV is the sum of volumes of boxes.
            # For 3D, we can compute the area of the 2D projection (f2, f3) for each "slice" in f1.
            
            def hv_2d(points, ref_2d):
                # points: (M, 2)
                if len(points) == 0: return 0.0
                if len(points) == 1:
                    p = points[0]
                    if p[0] >= ref_2d[0] or p[1] >= ref_2d[1]: return 0.0
                    return (ref_2d[0]-p[0]) * (ref_2d[1]-p[1])
                
                # Filter dominated
                nd_2d = []
                m = len(points)
                for i in range(m):
                    dom = False
                    for j in range(m):
                        if i != j:
                            if points[j][0] <= points[i][0] and points[j][1] <= points[i][1] and (points[j][0] < points[i][0] or points[j][1] < points[i][1]):
                                dom = True
                                break
                    if not dom:
                        nd_2d.append(points[i])
                        
                if len(nd_2d) == 0: return 0.0
                if len(nd_2d) == 1:
                    p = nd_2d[0]
                    return (ref_2d[0]-p[0]) * (ref_2d[1]-p[1])
                    
                # Sort by x (first coordinate)
                nd_2d = sorted(nd_2d, key=lambda x: x[0])
                
                # Calculate area
                # Points are non-dominated, so x is increasing and y is increasing?
                # Wait, in minimization, if x increases, y must increase for non-dominated.
                # So the "staircase" goes from bottom-left to top-right.
                # The HV is the area under the staircase, bounded by ref.
                # Area = sum of rectangles.
                # Let's trace:
                # Points (x1, y1), (x2, y2) with x1 < x2, y1 < y2.
                # HV = (x2 - x1)*(ref_y - y1) + (ref_x - x2)*(ref_y - y2)
                
                x_coords = [p[0] for p in nd_2d]
                y_coords = [p[1] for p in nd_2d]
                ref_x, ref_y = ref_2d
                
                area = 0.0
                for i in range(len(nd_2d)):
                    if i < len(nd_2d) - 1:
                        dx = x_coords[i+1] - x_coords[i]
                    else:
                        dx = ref_x - x_coords[i]
                    dy = ref_y - y_coords[i]
                    area += dx * dy
                    
                return area

            total_hv = 0.0
            n_nd = len(F_nd)
            active_2d_points = []
            
            for i in range(n_nd):
                # Add point i to active set
                # The point is (f2, f3)
                active_2d_points.append((F_nd[i][1], F_nd[i][2]))
                
                # Calculate 2D HV of active_2d_points
                h_2d = hv_2d(np.array(active_2d_points), np.array([ref[1], ref[2]]))
                
                # Width in f1
                if i < n_nd - 1:
                    width = F_nd[i+1][0] - F_nd[i][0]
                else:
                    width = ref[0] - F_nd[i][0]
                    
                total_hv += width * h_2d
                
            return total_hv

        # Main Loop
        # We will iterate through islands in a round-robin fashion to ensure balanced exploration
        island_counter = 0
        
        while used < budget:
            # Pick island
            island_idx = island_counter % n_islands
            island_counter += 1
            island = islands[island_idx]
            weights = island['weights']
            
            # Select parent: Weighted sum selection
            # Calculate weighted sum for each individual in the island
            # scores = F @ weights
            # To ensure we explore the specific region, we pick the best or a random one?
            # Let's use a tournament on weighted sum.
            # Or simply pick the one with the best weighted sum (closest to target weight profile).
            # To maintain diversity within the island, we can pick randomly from the top half?
            # Let's pick the one with the minimum weighted sum (best for that weight).
            scores = np.sum(island['F'] * weights, axis=1)
            parent_idx = np.argmin(scores)
            
            parent = island['X'][parent_idx]
            
            # Mutation
            # Adaptive step size based on progress
            progress = 1.0 - (used / budget)
            # Weight-specific mutation intensity?
            # Islands with "corner" weights (0.8, 0.1, 0.1) might need different exploration than "center" ones.
            # Let's use a standard Gaussian mutation with adaptive sigma.
            step_size = 0.1 * (0.5 + 0.5 * progress)
            mutant = parent + step_size * (hi - lo) * np.random.randn(dim)
            mutant = np.clip(mutant, lo, hi)
            
            f_mutant = problem(mutant)
            used += 1
            
            # Update Island
            # New candidate set
            new_X = np.vstack([island['X'], [mutant]])
            new_F = np.vstack([island['F'], [f_mutant]])
            
            # Selection: Keep top pop_size by Hypervolume contribution
            # We want the subset of size pop_size from new_F that maximizes HV.
            # Greedy approach:
            # 1. Start with empty set.
            # 2. Add point that maximizes HV of the set.
            # 3. Repeat until size is pop_size.
            # This is O(pop_size * N * HV_cost). N is pop_size+1.
            # Since pop_size is small (5), this is fast.
            
            # However, we can optimize:
            # If the new point is dominated by an existing point, discard it.
            # If the new point dominates an existing point, remove the dominated one.
            # If neither, we have a trade-off.
            
            # Let's do a simple greedy HV maximization for the whole set to be safe and accurate.
            # It's O(k * M * C_hv) where k is target size, M is current size.
            # M is ~6. k is 5. C_hv is small.
            
            selected_idx = []
            remaining_idx = list(range(len(new_F)))
            
            for _ in range(pop_size):
                if not remaining_idx:
                    break
                best_idx = -1
                best_hv = -1.0
                for idx in remaining_idx:
                    # Calculate HV of selected + idx
                    temp_set = np.vstack([new_F[selected_idx], [new_F[idx]]]) if selected_idx else new_F[idx:idx+1]
                    hv_val = calc_hv_3d(temp_set, ref_point)
                    if hv_val > best_hv:
                        best_hv = hv_val
                        best_idx = idx
                selected_idx.append(best_idx)
                remaining_idx.remove(best_idx)
                
            island['X'] = new_X[selected_idx]
            island['F'] = new_F[selected_idx]

        return None
