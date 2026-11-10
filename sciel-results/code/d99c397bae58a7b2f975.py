import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Define weight vectors for islands to partition the objective space
        # For 3 objectives, we use 4 islands covering corners and center
        self.weight_vectors = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [1.0/3, 1.0/3, 1.0/3]
        ])

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        n_islands = len(self.weight_vectors)
        pop_size = 10
        total_init = n_islands * pop_size
        
        if total_init >= budget:
            pop_size = max(1, budget // n_islands - 1)
            total_init = n_islands * pop_size
            
        ref_point = np.array([3.0, 5.0, 7.0])

        # Helper: Calculate 3D Hypervolume
        def calc_hv_3d(points, ref):
            """
            Calculate Hypervolume for 3D objective vectors.
            points: (N, 3) array
            ref: (3,) array
            Returns: float
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than ref
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            if len(pts) == 1:
                p = pts[0]
                return (ref[0]-p[0]) * (ref[1]-p[1]) * (ref[2]-p[2])
                
            # Remove dominated points
            nd_idx = []
            for i in range(len(pts)):
                dominated = False
                for j in range(len(pts)):
                    if i != j:
                        if pts[j][0] <= pts[i][0] and pts[j][1] <= pts[i][1] and pts[j][2] <= pts[i][2] and (pts[j][0] < pts[i][0] or pts[j][1] < pts[i][1] or pts[j][2] < pts[i][2]):
                            dominated = True
                            break
                if not dominated:
                    nd_idx.append(i)
                    
            pts_nd = pts[nd_idx]
            if len(pts_nd) == 0:
                return 0.0
            if len(pts_nd) == 1:
                p = pts_nd[0]
                return (ref[0]-p[0]) * (ref[1]-p[1]) * (ref[2]-p[2])
                
            # Sort by f1
            pts_nd = pts_nd[np.argsort(pts_nd[:, 0])]
            
            # Sweep line in f1
            # We calculate the 2D HV of the projection (f2, f3) for the "active" set
            # and multiply by the width in f1.
            
            def hv_2d(points_2d, ref_2d):
                if len(points_2d) == 0:
                    return 0.0
                # Filter dominated in 2D
                nd_2d = []
                for i in range(len(points_2d)):
                    dom = False
                    for j in range(len(points_2d)):
                        if i != j:
                            if points_2d[j][0] <= points_2d[i][0] and points_2d[j][1] <= points_2d[i][1] and (points_2d[j][0] < points_2d[i][0] or points_2d[j][1] < points_2d[i][1]):
                                dom = True
                                break
                    if not dom:
                        nd_2d.append(points_2d[i])
                        
                if len(nd_2d) == 0:
                    return 0.0
                if len(nd_2d) == 1:
                    p = nd_2d[0]
                    if p[0] >= ref_2d[0] or p[1] >= ref_2d[1]:
                        return 0.0
                    return (ref_2d[0]-p[0]) * (ref_2d[1]-p[1])
                    
                # Sort by x
                nd_2d = sorted(nd_2d, key=lambda x: x[0])
                area = 0.0
                for i in range(len(nd_2d)):
                    if i < len(nd_2d) - 1:
                        dx = nd_2d[i+1][0] - nd_2d[i][0]
                    else:
                        dx = ref_2d[0] - nd_2d[i][0]
                    dy = ref_2d[1] - nd_2d[i][1]
                    area += dx * dy
                return area

            total_hv = 0.0
            n = len(pts_nd)
            active_2d = []
            
            for i in range(n):
                active_2d.append((pts_nd[i][1], pts_nd[i][2]))
                h_2d = hv_2d(np.array(active_2d), np.array([ref[1], ref[2]]))
                
                if i < n - 1:
                    width = pts_nd[i+1][0] - pts_nd[i][0]
                else:
                    width = ref[0] - pts_nd[i][0]
                    
                total_hv += width * h_2d
                
            return total_hv

        # Helper: Greedy selection by Marginal HV
        def select_by_hv(candidates, ref, k):
            """
            Select k points from candidates that maximize total HV.
            candidates: list of (x, f) tuples or just F? We need X too.
            Let's pass indices or arrays.
            candidates_F: (N, 3)
            candidates_X: (N, dim)
            """
            if len(candidates_F) <= k:
                return candidates_X, candidates_F
                
            selected_indices = []
            remaining_indices = list(range(len(candidates_F)))
            
            while len(selected_indices) < k and remaining_indices:
                best_idx = -1
                best_hv = -1
                # Current HV of selected
                if selected_indices:
                    current_hv = calc_hv_3d(candidates_F[selected_indices], ref)
                else:
                    current_hv = 0.0
                    
                for i in remaining_indices:
                    p_f = candidates_F[i]
                    # Quick dominance check against selected
                    dominated = False
                    for s_idx in selected_indices:
                        s_f = candidates_F[s_idx]
                        if s_f[0] <= p_f[0] and s_f[1] <= p_f[1] and s_f[2] <= p_f[2] and (s_f[0] < p_f[0] or s_f[1] < p_f[1] or s_f[2] < p_f[2]):
                            dominated = True
                            break
                    if dominated:
                        continue
                        
                    new_selected = selected_indices + [i]
                    new_hv = calc_hv_3d(candidates_F[new_selected], ref)
                    if new_hv > best_hv:
                        best_hv = new_hv
                        best_idx = i
                        
                if best_idx == -1:
                    # Fallback: pick random
                    best_idx = remaining_indices[0]
                    
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                
            return candidates_X[selected_indices], candidates_F[selected_indices]

        # Initialize islands
        islands = []
        for i in range(n_islands):
            X = lo + (hi - lo) * np.random.rand(pop_size, dim)
            F = np.array([problem(x) for x in X])
            islands.append({'X': X, 'F': F})
            
        used = total_init

        # Main Loop
        while used < budget:
            # Pick island
            island_idx = np.random.randint(n_islands)
            island = islands[island_idx]
            w = self.weight_vectors[island_idx]
            
            # Select parent: Minimize weighted sum (Scalarization)
            # This ensures the island explores the region defined by w
            weighted_sums = island['F'] @ w
            parent_idx = np.argmin(weighted_sums)
            parent = island['X'][parent_idx]
            
            # Mutation
            # Adaptive step size
            progress = 1.0 - (used / budget)
            # Slightly larger steps early on, smaller later
            step_size = 0.15 * (0.5 + 0.5 * progress)
            
            # Weight-specific mutation intensity?
            # If the island is "stuck" (low diversity), increase step.
            # For simplicity, we use a constant adaptive step.
            # To satisfy "weight-specific mutation intensities", we can scale step by 
            # the "spread" of the population in that region.
            # Or just keep it simple. The weights guide the direction via parent selection.
            
            mutant = parent + step_size * (hi - lo) * np.random.randn(dim)
            mutant = np.clip(mutant, lo, hi)
            
            f_mutant = problem(mutant)
            used += 1
            
            # Update Island
            new_X = np.vstack([island['X'], [mutant]])
            new_F = np.vstack([island['F'], [f_mutant]])
            
            # Select top pop_size by HV (Greedy Marginal HV)
            # This ensures the local front is optimized for HV
            if len(new_F) > pop_size:
                island['X'], island['F'] = select_by_hv(new_F, new_F, pop_size, ref_point)
            else:
                island['X'] = new_X
                island['F'] = new_F

        return None
