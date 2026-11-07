"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using hypervolume-based greedy search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])

        # Hypervolume contribution of a point to a set of non-dominated points
        def hv_contribution(point, points):
            """Calculate hypervolume contribution of a single point to a set."""
            if len(points) == 0:
                # Hypervolume of a single point w.r.t. reference
                if np.all(point < ref_point):
                    return np.prod(ref_point - point)
                return 0.0
            
            # Check if point is dominated by any existing point
            for p in points:
                if np.all(p <= point) and np.any(p < point):
                    return 0.0
            
            # Calculate hypervolume of points + new point minus hypervolume of points
            def hv(points_list):
                if len(points_list) == 0:
                    return 0.0
                if len(points_list) == 1:
                    if np.all(points_list[0] < ref_point):
                        return np.prod(ref_point - points_list[0])
                    return 0.0
                # Simple recursive HV for 3D
                # Sort by first objective
                sorted_pts = sorted(points_list, key=lambda x: x[0])
                total_hv = 0.0
                for i in range(len(sorted_pts)):
                    p = sorted_pts[i]
                    if np.all(p < ref_point):
                        # Width in first dimension
                        if i == 0:
                            w0 = ref_point[0] - p[0]
                        else:
                            w0 = sorted_pts[i-1][0] - p[0]
                        # 2D hypervolume of remaining dimensions
                        remaining = [pt[1:] for pt in sorted_pts[i:] if np.all(pt[1:] < ref_point[1:])]
                        if len(remaining) > 0:
                            # 2D HV
                            remaining_sorted = sorted(remaining, key=lambda x: x[0])
                            hv2d = 0.0
                            for j in range(len(remaining_sorted)):
                                rp = remaining_sorted[j]
                                if j == 0:
                                    w1 = ref_point[1] - rp[0]
                                else:
                                    w1 = remaining_sorted[j-1][0] - rp[0]
                                w2 = ref_point[2] - rp[1]
                                hv2d += w1 * w2
                            total_hv += w0 * hv2d
                return total_hv
            
            hv_with = hv(points + [point])
            hv_without = hv(points)
            return hv_with - hv_without

        # Better: use a simpler approach - maintain archive of non-dominated points
        # and use greedy search with local refinement
        
        # Initial sampling
        n_init = min(30, self.budget // 4)
        if n_init < 5:
            n_init = 5
        
        X_archive = []
        F_archive = []
        used = 0
        
        # Generate initial points using stratified random
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            f = problem(x)
            used += 1
            # Check if non-dominated
            dominated = False
            for existing_f in F_archive:
                if np.all(existing_f <= f) and np.any(existing_f < f):
                    dominated = True
                    break
            if not dominated:
                # Remove dominated points from archive
                new_F_archive = []
                for ef in F_archive:
                    if not (np.all(f <= ef) and np.any(f < ef)):
                        new_F_archive.append(ef)
                F_archive = new_F_archive
                X_archive.append(x)
                F_archive.append(f)
        
        if len(F_archive) == 0:
            # Fallback: just use one point
            x = lo + (hi - lo) * 0.5
            f = problem(x)
            used += 1
            X_archive.append(x)
            F_archive.append(f)
        
        # Main loop: greedy local search from archive points
        best_hv = 0.0
        no_improve_count = 0
        max_no_improve = 20
        
        while used < self.budget and time.time() < deadline:
            # Pick a random point from archive to refine
            idx = rng.integers(0, len(X_archive))
            base_x = X_archive[idx].copy()
            
            # Local search: try small perturbations
            found_improvement = False
            n_attempts = min(10, self.budget - used)
            
            for attempt in range(n_attempts):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Adaptive step size
                progress = used / self.budget
                sigma = (hi - lo) * (0.15 * (1 - progress) + 0.02)
                
                # Gaussian perturbation
                x_new = base_x + rng.standard_normal(self.dim) * sigma
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Check if non-dominated
                dominated = False
                for existing_f in F_archive:
                    if np.all(existing_f <= f_new) and np.any(existing_f < f_new):
                        dominated = True
                        break
                
                if not dominated:
                    # Remove dominated points
                    new_X_archive = []
                    new_F_archive = []
                    for i in range(len(F_archive)):
                        if not (np.all(f_new <= F_archive[i]) and np.any(f_new < F_archive[i])):
                            new_X_archive.append(X_archive[i])
                            new_F_archive.append(F_archive[i])
                    
                    X_archive = new_X_archive
                    F_archive = new_F_archive
                    X_archive.append(x_new)
                    F_archive.append(f_new)
                    found_improvement = True
                    no_improve_count = 0
                    break
                else:
                    no_improve_count += 1
            
            if not found_improvement:
                no_improve_count += 1
            
            # If too many failures, try a completely random point
            if no_improve_count > max_no_improve and used < self.budget:
                if time.time() < deadline:
                    x_rand = lo + (hi - lo) * rng.random(self.dim)
                    f_rand = problem(x_rand)
                    used += 1
                    
                    dominated = False
                    for existing_f in F_archive:
                        if np.all(existing_f <= f_rand) and np.any(existing_f < f_rand):
                            dominated = True
                            break
                    
                    if not dominated:
                        new_X_archive = []
                        new_F_archive = []
                        for i in range(len(F_archive)):
                            if not (np.all(f_rand <= F_archive[i]) and np.any(f_rand < F_archive[i])):
                                new_X_archive.append(X_archive[i])
                                new_F_archive.append(F_archive[i])
                        
                        X_archive = new_X_archive
                        F_archive = new_F_archive
                        X_archive.append(x_rand)
                        F_archive.append(f_rand)
                        no_improve_count = 0
        
        return X_archive if X_archive else []
# EVOLVE-BLOCK-END
