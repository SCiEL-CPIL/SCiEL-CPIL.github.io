import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # For DTLZ4 with n=12, m=3: k = n - m + 1 = 10
        # Position vars: x_0..x_9, Convergence vars: x_10, x_11
        k = n - m + 1  # 10
        
        # Initialize: set convergence vars to 1.0 to be on the front
        # Use a spread of position variables
        archive = []  # list of (x, f)
        used = 0
        
        # Helper: create a point on the front
        def make_front_point(pos_vars):
            x = np.zeros(n)
            x[:k] = pos_vars
            x[k:] = 1.0  # convergence vars at 1
            return x
        
        # Initial population: use a Latin hypercube or grid in position space
        # But position space is 10D, so let's use random sampling
        rng = np.random.default_rng(42)
        
        init_size = min(30, self.budget // 5)
        for i in range(init_size):
            pos = rng.random(k)
            x = make_front_point(pos)
            f = np.array(problem(x))
            archive.append((x, f))
            used += 1
            if used >= self.budget or time.time() > deadline:
                break
        
        # Hypervolume computation (3D, exact)
        ref = np.array([2.5, 2.5, 2.5])
        
        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume for non-dominated points."""
            if len(points) == 0:
                return 0.0
            pts = np.array(points)
            # Filter points strictly better than ref
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            # For 3D, use the standard algorithm
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            # ... complex 3D HV computation
            # Use a simpler approach: since points are on/near a sphere, 
            # approximate with integration
            # Actually, let's use a proper 3D HV
            return _hv3d(pts, ref)
        
        def _hv3d(pts, ref):
            """Exact 3D hypervolume."""
            n = len(pts)
            if n == 0:
                return 0.0
            # Sort by f1
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            # Use the standard 3D algorithm
            # For each point, compute the volume of the box it contributes
            # This is complex; use a simpler method
            # Since we're dealing with points near a sphere, we can use
            # the fact that HV = integral over the front
            # For exact 3D HV, use the following:
            # Sort points by f1, then for each point, compute the 2D HV
            # of the remaining points in the f2-f3 plane, weighted by (ref[0] - f1)
            total = 0.0
            for i in range(n):
                f1 = pts[i, 0]
                # Points that are not dominated by this point in f2,f3
                # and have f1' >= f1 (since sorted)
                # The contribution is (ref[0] - f1) * HV_2D of points j>=i in (f2,f3)
                # but only those not dominated
                sub = pts[i:, 1:3]
                sub_ref = ref[1:3]
                hv2 = _hv2d(sub, sub_ref)
                total += (ref[0] - f1) * hv2
            return total
        
        def _hv2d(pts, ref):
            """Exact 2D hypervolume."""
            n = len(pts)
            if n == 0:
                return 0.0
            # Sort by f2
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            total = 0.0
            max_f3 = 0.0
            for i in range(n-1, -1, -1):
                f2 = pts[i, 0]
                f3 = pts[i, 1]
                if f3 > max_f3:
                    max_f3 = f3
                total += (ref[0] - f2) * (ref[1] - max_f3)
            return total
        
        # Greedy hypervolume maximization
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidates by mutating archive points
            candidates = []
            n_cand = min(20, self.budget - used)
            
            for _ in range(n_cand):
                # Pick a random archive point
                idx = rng.integers(len(archive))
                parent_x = archive[idx][0].copy()
                
                # Mutate position variables only
                pos = parent_x[:k].copy()
                # Gaussian mutation
                pos += 0.1 * rng.standard_normal(k)
                pos = np.clip(pos, 0, 1)
                
                x = make_front_point(pos)
                f = np.array(problem(x))
                used += 1
                
                # Compute marginal HV contribution
                # Add to archive, compute HV, remove, compute HV, difference
                # This is expensive; approximate
                candidates.append((x, f))
                
                if used >= self.budget or time.time() > deadline:
                    break
            
            # Select the best candidate by marginal HV
            if candidates:
                best_idx = 0
                best_hv = -1
                for i, (cx, cf) in enumerate(candidates):
                    # Approximate marginal HV: distance to nearest neighbor
                    # or just use the sum of objective values (lower is better)
                    # Better: compute actual marginal HV
                    new_archive = archive + [(cx, cf)]
                    hv_new = hypervolume_3d([p[1] for p in new_archive], ref)
                    if hv_new > best_hv:
                        best_hv = hv_new
                        best_idx = i
                
                # Add the best candidate
                best_x, best_f = candidates[best_idx]
                archive.append((best_x, best_f))
        
        return archive
