"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hypervolume-guided adaptive differential evolution for bi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Deterministic Grid Initialization
        # Create corners (2^dim) and midpoints to ensure coverage
        # For dim=4, 2^4=16 points.
        n_corners = 2 ** self.dim
        corners = []
        for i in range(n_corners):
            point = []
            for j in range(self.dim):
                # Bit j of i determines low or high
                if (i >> j) & 1:
                    point.append(hi[j])
                else:
                    point.append(lo[j])
            corners.append(point)
        
        # Add midpoints for better initial spread
        midpoints = []
        for j in range(self.dim):
            point = (lo + hi) / 2.0
            # Perturb slightly or just use center? 
            # Let's use the center point once, and maybe some axis-aligned midpoints
            # Actually, just adding the single center point is good.
            midpoints.append((lo + hi) / 2.0)
            
        # Combine initial candidates
        init_points = np.array(corners + midpoints)
        
        # Ensure we don't exceed budget for init
        if len(init_points) > self.budget:
            init_points = init_points[:self.budget]
            
        # Evaluate initial points
        F = np.array([problem(x) for x in init_points])
        X = init_points
        used = len(X)
        
        # Filter to non-dominated for the active population
        # We will maintain an archive of non-dominated points
        archive_X = []
        archive_F = []
        
        def is_dominated(f, archive_F):
            # Check if f is dominated by any point in archive_F
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        # Populate archive
        for i in range(len(X)):
            if not is_dominated(F[i], archive_F):
                # Remove points dominated by F[i]
                new_archive_F = []
                new_archive_X = []
                for j in range(len(archive_F)):
                    if not (np.all(F[i] <= archive_F[j]) and np.any(F[i] < archive_F[j])):
                        new_archive_F.append(archive_F[j])
                        new_archive_X.append(archive_X[j])
                archive_F.append(F[i])
                archive_X.append(X[i])
                archive_F = new_archive_F
                archive_X = new_archive_X
        
        # If archive is empty (shouldn't happen), fallback
        if not archive_F:
             archive_F = [F[0]]
             archive_X = [X[0]]
        
        # 2. Adaptive Differential Evolution Loop
        # We use a population size based on budget
        # Let's keep the archive size manageable, e.g., max 10-20 points
        max_archive_size = 15
        
        # Mutation factor F_de starts high and decays
        f_de = 0.8
        cr = 0.5 # Crossover rate (not used in standard DE for 1 child, but we can use it for mixing)
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select parents from archive
            n_arch = len(archive_X)
            if n_arch < 3:
                # If archive is too small, use random points from initial set or just random search
                # Let's just do random search from bounds
                parent_idx = 0
                x1 = archive_X[0]
                x2 = archive_X[0]
                x3 = archive_X[0]
            else:
                # Select distinct parents
                idx1 = np.random.randint(n_arch)
                idx2 = np.random.randint(n_arch)
                idx3 = np.random.randint(n_arch)
                # Ensure distinctness if possible
                if idx1 == idx2: idx2 = (idx2 + 1) % n_arch
                if idx1 == idx3: idx3 = (idx3 + 1) % n_arch
                if idx2 == idx3: idx3 = (idx3 + 1) % n_arch
                
                x1 = np.array(archive_X[idx1])
                x2 = np.array(archive_X[idx2])
                x3 = np.array(archive_X[idx3])
            
            # DE Mutation: v = x1 + F * (x2 - x3)
            v = x1 + f_de * (x2 - x3)
            
            # Clip to bounds
            v = np.clip(v, lo, hi)
            
            # Evaluate
            fv = problem(v)
            used += 1
            
            # Update Archive
            # Check if v is dominated by archive
            if not is_dominated(fv, archive_F):
                # Remove dominated points from archive
                new_archive_F = []
                new_archive_X = []
                for i in range(len(archive_F)):
                    if not (np.all(fv <= archive_F[i]) and np.any(fv < archive_F[i])):
                        new_archive_F.append(archive_F[i])
                        new_archive_X.append(archive_X[i])
                
                # Add v
                new_archive_F.append(fv)
                new_archive_X.append(v)
                
                # If archive is too large, prune by hypervolume contribution
                if len(new_archive_F) > max_archive_size:
                    # Calculate hypervolume contribution of each point
                    # For 2 objectives, HV contribution of point i is:
                    # (max_x - x_i) * (max_y - y_i) ... roughly, but depends on neighbors.
                    # Simple heuristic: remove the point with the largest sum of objectives (worst)
                    # Or calculate marginal HV.
                    # Let's use a simple marginal HV approximation:
                    # Sort by obj1.
                    # For 2D, marginal HV of point i is (ref1 - f1_i) * (f2_i - f2_next_worse) ...
                    # Simpler: just remove the point that is dominated by the convex hull of others?
                    # Easiest robust method: Remove the point with the lowest "quality" score.
                    # Score = - (f1 + f2) is bad because scales differ.
                    # Let's use the hypervolume contribution calculation.
                    
                    # Function to calculate marginal HV contribution for 2D
                    def calc_hv_contrib(points_f, ref):
                        # points_f: list of [f1, f2]
                        # ref: [ref1, ref2]
                        # Sort by f1 ascending
                        pts = sorted(points_f, key=lambda x: x[0])
                        contribs = np.zeros(len(pts))
                        prev_f2 = ref[1]
                        # Iterate from worst f1 to best f1?
                        # HV is sum of rectangles.
                        # Let's sort by f1 ascending (best to worst).
                        # f1: small -> large
                        # f2: small -> large
                        # The contribution of a point is the area it adds.
                        # If we sort by f1 ascending, the "next" point in f1 has a larger f1.
                        # The vertical span is determined by the next point with a smaller f2?
                        
                        # Standard 2D HV algorithm:
                        # Sort by f1 ascending.
                        # Keep track of min f2 seen so far (from the right/best side?).
                        # Actually, easier:
                        # Sort by f1 ascending.
                        # Iterate from last (worst f1) to first (best f1).
                        # Maintain min_f2_so_far.
                        # Contribution of point i is (f1_{i+1} - f1_i) * (min_f2_so_far - f2_i) ?? No.
                        
                        # Let's use the property:
                        # HV = sum over non-dominated points of (ref1 - f1_i) * (f2_i - f2_{i+1}) if sorted by f1 asc and f2 desc?
                        # No, non-dominated points sorted by f1 asc have f2 desc.
                        
                        pts_sorted = sorted(points_f, key=lambda x: x[0])
                        # pts_sorted[0] has smallest f1, largest f2 (among non-dom)
                        # pts_sorted[-1] has largest f1, smallest f2
                        
                        # Marginal HV of point i:
                        # It is the area of the rectangle defined by (f1_i, f2_i) and the next "worse" point in the front?
                        # Actually, the marginal contribution is the area that would be lost if the point were removed.
                        # For 2D, if points are sorted by f1 asc (and thus f2 desc):
                        # Point i contributes: (f1_{i+1} - f1_i) * (f2_i - f2_{i+1}) ? No.
                        # The rectangle added by point i is bounded by f1_i, f2_i, and the reference/next points.
                        
                        # Let's just use a simple proxy: distance to reference point.
                        # Points closer to (0,0) are better.
                        # dist = sqrt((f1/ref1)^2 + (f2/ref2)^2)
                        # Remove point with largest dist.
                        
                        scores = []
                        for p in new_archive_F:
                            # Normalize by ref to handle scale
                            d = np.sqrt((p[0]/ref[0])**2 + (p[1]/ref[1])**2)
                            scores.append(d)
                        
                        # Find index of worst score
                        worst_idx = np.argmax(scores)
                        
                        # Remove it
                        del new_archive_F[worst_idx]
                        del new_archive_X[worst_idx]
                
                archive_F = new_archive_F
                archive_X = new_archive_X
            
            # Decay mutation factor
            f_de *= 0.99
            
            # Optional: if no improvement in last 10 steps, increase exploration
            # (Not implemented for simplicity, but DE with decaying F is usually fine)
            
        # Return the best result found (ignored by evaluator, but required)
        if archive_X:
            return np.array(archive_X)
        else:
            return X

# EVOLVE-BLOCK-END
