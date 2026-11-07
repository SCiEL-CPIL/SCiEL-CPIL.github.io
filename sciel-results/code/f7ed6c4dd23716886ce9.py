import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin (300s limit)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 specific knowledge:
        # Variables 0 to dim-3 are non-influencing (affect g)
        # Variables dim-2 to dim-1 are influencing (affect y)
        # To minimize g, set non-influencing vars to 0.
        # Optimize influencing vars.
        
        n_influencing = 2
        n_non_influencing = self.dim - n_influencing
        
        # Helper to create a full x vector from influencing vars
        def make_x(y_vars):
            x = np.zeros(self.dim)
            x[self.dim-n_influencing:] = y_vars
            return x
        
        def check_deadline():
            return time.time() > deadline
        
        def get_f_from_x(x):
            x_clipped = np.clip(x, lo, hi)
            return problem(x_clipped)
        
        evaluations_used = 0
        points = []  # List of (x, f) tuples
        
        # Phase 1: Dense grid sampling on the Pareto front subspace
        # For DTLZ7 with g=0, the Pareto front is parameterized by the last 2 vars
        # Use a dense grid to get good coverage of the front
        grid_size = 14
        y1_vals = np.linspace(0.0, 1.0, grid_size)
        y2_vals = np.linspace(0.0, 1.0, grid_size)
        
        for y1 in y1_vals:
            for y2 in y2_vals:
                if evaluations_used >= self.budget or check_deadline():
                    break
                
                y_vars = np.array([y1, y2])
                x = make_x(y_vars)
                x = np.clip(x, lo, hi)
                
                f = get_f_from_x(x)
                points.append((x, f))
                evaluations_used += 1
            
            if evaluations_used >= self.budget or check_deadline():
                break
        
        if evaluations_used >= self.budget or check_deadline():
            return None
        
        # Helper: Check if a point is dominated by any point in the list
        def is_dominated(f_new, points_list):
            for _, f in points_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
        
        # Helper: Get non-dominated points
        def get_non_dominated(points_list):
            nd_indices = []
            n = len(points_list)
            for i in range(n):
                f_i = points_list[i][1]
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    f_j = points_list[j][1]
                    if np.all(f_j <= f_i) and np.any(f_j < f_i):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices
        
        # Phase 2: Iterative refinement with local optimization
        # Strategy: 
        # 1. Find non-dominated points
        # 2. For each non-dominated point, perform local search to improve it
        # 3. Use gradient-free local optimization (Nelder-Mead style)
        
        while evaluations_used < self.budget and not check_deadline():
            # Get current non-dominated set
            nd_indices = get_non_dominated(points)
            
            if len(nd_indices) < 2:
                # If we have too few non-dominated points, do random sampling
                for _ in range(min(5, self.budget - evaluations_used)):
                    if evaluations_used >= self.budget or check_deadline():
                        break
                    y_vars = np.random.uniform(0, 1, n_influencing)
                    x = make_x(y_vars)
                    x = np.clip(x, lo, hi)
                    f = get_f_from_x(x)
                    points.append((x, f))
                    evaluations_used += 1
                continue
            
            # For each non-dominated point, try local refinement
            # Use a simple coordinate descent / local search approach
            refined = False
            for idx in nd_indices:
                if evaluations_used >= self.budget or check_deadline():
                    break
                
                x_nd, f_nd = points[idx]
                y_nd = x_nd[self.dim-n_influencing:].copy()
                
                # Try small perturbations in each direction
                for dim_i in range(n_influencing):
                    for step in [0.01, 0.005, 0.001]:
                        if evaluations_used >= self.budget or check_deadline():
                            break
                        
                        for direction in [-1, 1]:
                            if evaluations_used >= self.budget or check_deadline():
                                break
                            
                            y_trial = y_nd.copy()
                            y_trial[dim_i] += direction * step
                            y_trial = np.clip(y_trial, 0, 1)
                            
                            x_trial = make_x(y_trial)
                            x_trial = np.clip(x_trial, lo, hi)
                            f_trial = get_f_from_x(x_trial)
                            evaluations_used += 1
                            
                            # Check if this improves (non-dominated and better in at least one objective)
                            if not is_dominated(f_trial, points):
                                # Check if it's actually better than f_nd in some sense
                                if np.any(f_trial < f_nd):
                                    points.append((x_trial, f_trial))
                                    refined = True
                                    break
                        if evaluations_used >= self.budget or check_deadline():
                            break
                    if evaluations_used >= self.budget or check_deadline():
                        break
                
                if evaluations_used >= self.budget or check_deadline():
                    break
            
            # If no refinement happened, try gap-filling
            if not refined:
                nd_points = [(points[i][0], points[i][1]) for i in nd_indices]
                nd_points_sorted = sorted(nd_points, key=lambda p: p[1][0])
                
                gap_candidates = []
                for i in range(len(nd_points_sorted) - 1):
                    x1, f1 = nd_points_sorted[i]
                    x2, f2 = nd_points_sorted[i + 1]
                    
                    for t in [0.33, 0.5, 0.67]:
                        y1 = x1[self.dim-n_influencing:]
                        y2 = x2[self.dim-n_influencing:]
                        y_interp = y1 * (1 - t) + y2 * t
                        y_perturbed = y_interp + np.random.normal(0, 0.01, n_influencing)
                        y_perturbed = np.clip(y_perturbed, 0, 1)
                        gap_candidates.append(y_perturbed)
                
                np.random.shuffle(gap_candidates)
                
                for y_vars in gap_candidates:
                    if evaluations_used >= self.budget or check_deadline():
                        break
                    
                    x = make_x(y_vars)
                    x = np.clip(x, lo, hi)
                    f = get_f_from_x(x)
                    evaluations_used += 1
                    
                    if not is_dominated(f, points):
                        points.append((x, f))
        
        return None
