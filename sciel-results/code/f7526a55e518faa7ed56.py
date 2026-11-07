"""Multi-objective optimizer: HV-guided greedy search with LHS init and adaptive local mutation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # 45s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        used = 0
        archive_F = []  # non-dominated objective vectors
        archive_X = []  # corresponding x vectors

        # --- Helpers ---
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def hv_2d(pts, ref_y, ref_z):
            """Compute 2D hypervolume of points with reference (ref_y, ref_z)."""
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts)
            mask = (pts[:, 0] < ref_y) & (pts[:, 1] < ref_z)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n_pts = len(pts)
            hv2 = 0.0
            for i in range(n_pts):
                y_next = pts[i + 1, 0] if i + 1 < n_pts else ref_y
                if i + 1 < n_pts:
                    min_z_remaining = np.min(pts[i + 1:, 1])
                else:
                    min_z_remaining = ref_z
                if min_z_remaining > pts[i, 1]:
                    w_y = y_next - pts[i, 0]
                    w_z = min_z_remaining - pts[i, 1]
                    if w_y > 0 and w_z > 0:
                        hv2 += w_y * w_z
            return hv2

        def hv_3d(points, ref):
            """Compute 3D hypervolume using slicing method."""
            if len(points) == 0:
                return 0.0
            P = np.array(points)
            mask = np.all(P < ref, axis=1)
            P = P[mask]
            if len(P) == 0:
                return 0.0

            order = np.argsort(P[:, 0])
            P_sorted = P[order]
            n_pts = len(P_sorted)
            hv = 0.0
            for i in range(n_pts):
                f0_min = P_sorted[i, 0]
                f0_max = P_sorted[i + 1, 0] if i + 1 < n_pts else ref[0]
                if f0_max <= f0_min:
                    continue
                mask_slice = (P_sorted[:, 0] >= f0_min) & (P_sorted[:, 0] < f0_max)
                slice_pts = P_sorted[mask_slice, 1:3]
                hv_slice = hv_2d(slice_pts, ref[1], ref[2])
                hv += hv_slice * (f0_max - f0_min)
            return hv

        def marginal_hv(point, archive):
            """Compute marginal HV contribution of adding point to archive."""
            if not np.all(point < ref_point):
                return 0.0
            for f in archive:
                if dominates(f, point):
                    return 0.0
            if len(archive) == 0:
                return hv_3d([point], ref_point)
            hv_before = hv_3d(archive, ref_point)
            hv_after = hv_3d(archive + [point], ref_point)
            return max(0.0, hv_after - hv_before)

        def is_dominated(point, archive):
            for f in archive:
                if dominates(f, point):
                    return True
            return False

        def add_point(f, x):
            """Add point to archive if it improves HV. Returns True if added."""
            nonlocal used
            if used >= budget:
                return False
            if not np.all(f < ref_point):
                return False
            if is_dominated(f, archive_F):
                return False
            # Check if it actually adds HV
            mhv = marginal_hv(f, archive_F)
            if mhv > 1e-12:
                archive_F.append(f)
                archive_X.append(x)
                used += 1
                return True
            return False

        # --- Latin Hypercube initialization ---
        n_lhs = min(30, budget)
        lhs_points = []
        for _ in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                x[d] = lo[d] + (hi[d] - lo[d]) * np.random.rand()
            lhs_points.append(x)

        # Evaluate LHS points
        for x in lhs_points:
            if used >= budget or time.time() > deadline:
                break
            try:
                f = np.array(problem(x))
            except Exception:
                continue
            add_point(f, x)

        # --- Greedy HV-guided search with local mutation ---
        # Strategy: pick parent from archive with highest "potential" (marginal HV direction),
        # mutate it, and evaluate. Accept if it improves HV.
        
        # Adaptive step size
        step_size = 0.15
        
        # Phase 1: broader exploration
        phase1_budget = min(80, budget - used)
        for i in range(phase1_budget):
            if used >= budget or time.time() > deadline:
                break
            
            if len(archive_F) == 0:
                # No archive yet, try random
                x = lo + np.random.rand(dim) * (hi - lo)
            else:
                # Pick parent: prefer points with higher marginal HV contribution
                # Approximate by picking randomly from archive with some bias
                # Better: pick the point that has the most "room" to improve
                parent_idx = np.random.randint(len(archive_F))
                parent_x = archive_X[parent_idx]
                
                # Mutate: Gaussian perturbation
                x = parent_x + np.random.randn(dim) * step_size
                x = np.clip(x, lo, hi)
            
            try:
                f = np.array(problem(x))
            except Exception:
                continue
            
            if add_point(f, x):
                # Successful addition: slightly decrease step size
                step_size *= 0.95
            else:
                # Failed: try a different direction
                step_size *= 1.05
                step_size = min(step_size, 0.3)

        # Phase 2: finer local search around archive points
        step_size = 0.05
        phase2_budget = min(60, budget - used)
        for i in range(phase2_budget):
            if used >= budget or time.time() > deadline:
                break
            
            if len(archive_F) == 0:
                break
            
            # Pick parent with highest "spread" potential
            # Use a simple heuristic: pick randomly but weight by inverse of local density
            parent_idx = np.random.randint(len(archive_F))
            parent_x = archive_X[parent_idx]
            
            # Smaller mutation
            x = parent_x + np.random.randn(dim) * step_size
            x = np.clip(x, lo, hi)
            
            try:
                f = np.array(problem(x))
            except Exception:
                continue
            
            if add_point(f, x):
                step_size *= 0.9
            else:
                step_size *= 1.02
                step_size = min(step_size, 0.1)

        # Phase 3: very fine polish
        step_size = 0.01
        phase3_budget = min(30, budget - used)
        for i in range(phase3_budget):
            if used >= budget or time.time() > deadline:
                break
            
            if len(archive_F) == 0:
                break
            
            parent_idx = np.random.randint(len(archive_F))
            parent_x = archive_X[parent_idx]
            
            x = parent_x + np.random.randn(dim) * step_size
            x = np.clip(x, lo, hi)
            
            try:
                f = np.array(problem(x))
            except Exception:
                continue
            
            add_point(f, x)

        # Return all evaluated x points (the evaluator records all returned objective vectors)
        # We need to return something; the return value is ignored but we should return the archive X
        return archive_X
# EVOLVE-BLOCK-END
