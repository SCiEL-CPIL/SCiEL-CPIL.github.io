# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """Compute 3D hypervolume contribution of a set of non-dominated points."""
        if len(points) == 0:
            return 0.0
        # Sort by first objective
        points = sorted(points, key=lambda p: p[0])
        total = 0.0
        prev_y = ref[1]
        prev_z = ref[2]
        for p in points:
            if p[1] < ref[1] and p[2] < ref[2]:
                # The contribution of this point to hypervolume
                # For 3D, we need to be more careful
                pass
        
        # Better approach: sort by x, then compute areas in y-z plane
        # Filter points that are within reference point
        valid = [p for p in points if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
        if len(valid) == 0:
            return 0.0
        
        # Sort by first objective
        valid = sorted(valid, key=lambda p: p[0])
        
        total = 0.0
        prev_x = valid[0][0]
        
        # For each point, compute the hypervolume it uniquely contributes
        # This is complex for 3D. Let's use a simpler approach:
        # Sort by x, then for each point, the contribution is (x_next - x_i) * area_in_yz
        
        for i in range(len(valid)):
            x_i = valid[i][0]
            y_i = valid[i][1]
            z_i = valid[i][2]
            
            if i < len(valid) - 1:
                x_next = valid[i + 1][0]
            else:
                x_next = ref[0]
            
            if x_next <= x_i:
                continue
                
            # The width in x direction
            dx = x_next - x_i
            
            # Now compute the area in y-z plane for points up to this one
            # that have y < y_i and z < z_i... this is getting complex
            
            # Simpler: just sum up the contributions using a grid-like approach
            # Actually, for a small number of points, we can compute hypervolume
            # by iterating through all non-dominated points and computing their
            # unique contribution.
            
            # For point i, its unique contribution is the box:
            # [x_i, x_next) x [y_i, ?) x [z_i, ?)
            # where ? depends on other points
            
            # Let's use a different, simpler but correct method for small sets:
            # The hypervolume is the union of boxes from each point to ref.
            # For 3D with small n, we can compute this by:
            # 1. Sort by x
            # 2. For each consecutive pair in x, compute the 2D hypervolume
            #    of the y-z projections of points with x <= current x
            #    minus the 2D hypervolume of points with x <= previous x
            
            pass
        
        # Let me use a cleaner implementation
        return self._hv_3d_impl(valid, ref)

    def _hv_3d_impl(self, points, ref):
        """Compute 3D hypervolume for a list of non-dominated points."""
        if len(points) == 0:
            return 0.0
        
        # Sort by first objective
        points = sorted(points, key=lambda p: p[0])
        
        total = 0.0
        n = len(points)
        
        for i in range(n):
            x_i = points[i][0]
            y_i = points[i][1]
            z_i = points[i][2]
            
            x_next = points[i + 1][0] if i < n - 1 else ref[0]
            if x_next <= x_i:
                continue
            
            dx = x_next - x_i
            
            # Compute 2D hypervolume in y-z plane for points[0..i]
            # projected onto y-z, with reference (ref[1], ref[2])
            # But we only care about points that have x <= x_i, which is all of points[0..i]
            # The 2D hypervolume of the set {(y_j, z_j) for j=0..i} with ref (ref[1], ref[2])
            
            # 2D hypervolume: sort by y, then sum (y_next - y_i) * (ref_z - z_i) for non-dominated
            # But we need to handle non-dominance in 2D
            
            # Collect y-z pairs for points 0..i
            yz_points = [(points[j][1], points[j][2]) for j in range(i + 1)
                        if points[j][1] < ref[1] and points[j][2] < ref[2]]
            
            if not yz_points:
                continue
            
            # 2D hypervolume
            hv_2d = self._hv_2d(yz_points, ref[1], ref[2])
            
            # But this overcounts. We need the incremental contribution.
            # Actually, the correct formula is:
            # HV = sum over i of dx_i * HV_2D({(y_j,z_j) : j <= i, y_j < ref_y, z_j < ref_z})
            # This is correct because the x-interval [x_i, x_{i+1}) is dominated only
            # by points with index <= i.
            
            total += dx * hv_2d
        
        return total

    def _hv_2d(self, points, ref_y, ref_z):
        """Compute 2D hypervolume for a list of (y, z) points."""
        if len(points) == 0:
            return 0.0
        
        # Filter valid points
        points = [(y, z) for y, z in points if y < ref_y and z < ref_z]
        if len(points) == 0:
            return 0.0
        
        # Sort by y
        points = sorted(points, key=lambda p: p[0])
        
        total = 0.0
        n = len(points)
        
        for i in range(n):
            y_i = points[i][0]
            z_i = points[i][1]
            
            y_next = points[i + 1][0] if i < n - 1 else ref_y
            if y_next <= y_i:
                continue
            
            dy = y_next - y_i
            
            # The z-height at this y-interval is the minimum z among points 0..i
            # that have y <= y_i... actually, for 2D hypervolume:
            # The contribution is dy * (ref_z - min_z) where min_z is the minimum
            # z value among points with y <= y_i (which is all points 0..i)
            
            min_z = min(z for _, z in points[:i + 1])
            if min_z < ref_z:
                total += dy * (ref_z - min_z)
        
        return total

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        # Reference point for internal hypervolume calculations
        ref_point = np.array([3.0, 5.0, 7.0])

        # Phase 1: Initial space-filling sampling
        # Use a Halton-like sequence for good coverage
        n_initial = min(40, budget)
        X = []
        F = []
        evals_used = 0

        # Generate space-filling points using a simple lattice
        # For 12 dimensions, use a grid-like approach with prime bases
        def halton_sequence(n, dim):
            """Generate n points in [0,1)^dim using Halton sequence."""
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            points = np.zeros((n, dim))
            for d in range(dim):
                base = primes[d]
                for i in range(n):
                    f = 1.0
                    num = i + 1
                    while num > 0:
                        f /= base
                        num = (num - 1) // base
                    points[i, d] = f
            return points

        halton_pts = halton_sequence(n_initial, dim)
        for i in range(n_initial):
            x = lo + (hi - lo) * halton_pts[i]
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            evals_used += 1
            if evals_used >= budget:
                break

        X = np.array(X)
        F = np.array(F)

        # Phase 2: Iterative refinement
        # Keep the best non-dominated points and refine them
        while evals_used < budget:
            if time.time() > deadline:
                break

            # Identify non-dominated points
            n = len(F)
            dominated = [False] * n
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        dominated[j] = True

            # Get non-dominated indices
            nondom_idx = [i for i in range(n) if not dominated[i]]

            if len(nondom_idx) == 0:
                # All dominated, pick the best single point
                best_idx = np.argmin(np.sum(F, axis=1))
                nondom_idx = [best_idx]

            # Select the top k non-dominated points to refine
            # Use hypervolume contribution as a guide
            k_refine = min(5, len(nondom_idx), max(1, (budget - evals_used) // 10))

            # Sort non-dominated points by their hypervolume contribution
            # Approximate: points closer to the "ideal" point are better
            # Ideal point is the minimum of each objective
            ideal = np.min(F, axis=0)
            scores = np.sum((F - ideal) ** 2, axis=1)
            nondom_sorted = sorted(nondom_idx, key=lambda i: scores[i])

            # Refine the top k points
            for idx in nondom_sorted[:k_refine]:
                if evals_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Local refinement around this point
                x_base = X[idx].copy()
                f_base = F[idx].copy()

                # Try a few random perturbations
                n_local = min(8, budget - evals_used)
                for _ in range(n_local):
                    if evals_used >= budget:
                        break
                    if time.time() > deadline:
                        break

                    # Random perturbation with decreasing step size
                    step = 0.1 * (hi - lo)
                    x_new = x_base + rng.standard_normal(dim) * step
                    x_new = np.clip(x_new, lo, hi)

                    f_new = problem(x_new)
                    evals_used += 1

                    # Check if f_new is non-dominated w.r.t. current population
                    is_dominated = False
                    for j in range(len(F)):
                        if np.all(F[j] <= f_new) and np.any(F[j] < f_new):
                            is_dominated = True
                            break

                    if not is_dominated:
                        # Add to population
                        X = np.vstack([X, x_new])
                        F = np.vstack([F, f_new])

                        # Remove any points dominated by f_new
                        new_dominated = []
                        for j in range(len(F) - 1):
                            if np.all(f_new <= F[j]) and np.any(f_new < F[j]):
                                new_dominated.append(j)

                        if new_dominated:
                            # Keep only non-dominated indices
                            keep = [j for j in range(len(F) - 1) if j not in new_dominated]
                            keep.append(len(F) - 1)  # the new point
                            X = X[keep]
                            F = F[keep]

                        # Update base if improved
                        if np.all(f_new <= f_base):
                            x_base = x_new
                            f_base = f_new

                        # Limit population size
                        if len(F) > 50:
                            # Keep only non-dominated points
                            n_pop = len(F)
                            dom = [False] * n_pop
                            for i in range(n_pop):
                                if dom[i]:
                                    continue
                                for j in range(n_pop):
                                    if i == j:
                                        continue
                                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                                        dom[j] = True
                            keep = [i for i in range(n_pop) if not dom[i]]
                            if len(keep) > 50:
                                # If still too many, keep the best 50 by sum of objectives
                                keep_scores = np.sum(F[keep], axis=1)
                                keep_sorted = sorted(keep, key=lambda i: keep_scores[i])
                                keep = keep_sorted[:50]
                            X = X[keep]
                            F = F[keep]

        return X
# EVOLVE-BLOCK-END
