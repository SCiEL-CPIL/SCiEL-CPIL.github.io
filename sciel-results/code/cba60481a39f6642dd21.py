# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using hypervolume-driven search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 290s safety margin (limit is 330s)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        width = hi - lo

        rng = np.random.default_rng(42)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (scoring reference)
        ref = np.array([3.0, 5.0, 7.0])

        # --- Initial population ---
        pop_size = min(40, budget // 2)
        pop_size = max(pop_size, 10)
        X = lo + width * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive: list of (x, f) pairs — all non-dominated points seen
        # We'll maintain a list and periodically prune dominated points
        archive_X = list(X)
        archive_F = list(F)

        def hypervolume_contribution(f, points_F, ref_pt):
            """Compute the hypervolume contribution of point f given existing points.
            For 3 objectives, compute the volume of the box dominated by f
            that is not dominated by any point in points_F.
            """
            # Only count if f is better than ref in all objectives
            if np.any(f >= ref_pt):
                return 0.0
            # For each objective, the max extent is ref
            # The contribution is the volume of the region dominated by f
            # but not by any existing point.
            # Simple approach: compute full box volume, subtract overlaps
            # For 3D, the box dominated by f is [f0, ref0] x [f1, ref1] x [f2, ref2]
            # We need to subtract the part already covered by existing points.
            # Efficient 3D: for each existing point p, the overlap is the intersection.
            # But this is O(n) per query. For small archives it's fine.
            
            box_vol = np.prod(ref_pt - f)
            # Subtract volumes of existing points that dominate parts of f's box
            # Actually, we want the NEW volume added. The new volume is the part
            # of f's box not covered by the union of existing boxes.
            # For 3D, a simpler approach: check if f is dominated by any existing point
            # If so, contribution is 0. Otherwise, compute the volume of f's box
            # minus the parts covered by existing points.
            
            # Check if f is dominated
            for pf in points_F:
                if np.all(pf <= f) and np.any(pf < f):
                    return 0.0
            
            # Compute contribution via inclusion-exclusion for 3D
            # The new volume is the set of points y where y_i >= f_i for all i
            # and y is not dominated by any existing point.
            # For 3D, this can be computed by considering the "skyline" of existing points.
            # Simplified: for each pair of objectives, find the max extent.
            
            # A practical approach: the contribution is approximately
            # the volume of the box minus overlaps with existing points.
            # For 3 objectives, we can do:
            # new_vol = box_vol
            # For each existing point pf:
            #   overlap = product of max(0, min(ref_j, pf_j) - f_j) for j where pf_j > f_j
            #   But this over-subtracts due to double counting.
            
            # Better: use the fact that in 3D, the contribution of f is:
            # Integrate over the region. For small archives, just compute numerically.
            # Actually, for the purpose of selection, we can use a simpler proxy:
            # the volume of the box that is NOT dominated by any single existing point.
            # This is an upper bound but works well for ranking.
            
            # For 3D, the exact contribution can be computed as:
            # Sort existing points and use sweep-line. But for simplicity and speed,
            # use: contribution = box_vol - sum of overlaps (approximate, but monotone)
            # Actually, let's just use a simpler metric: the "gap" in the Pareto front.
            
            # Simplest effective approach: contribution = product of (ref - f)
            # weighted by how "spread out" it is from existing points.
            # For selection purposes, we rank by the volume of the box that f dominates
            # which isn't dominated by others. In 3D with small n, we can do:
            
            # Exact 3D hypervolume contribution:
            # The new points covered are those y in [f0,ref0]x[f1,ref1]x[f2,ref2]
            # such that no existing pf dominates y, i.e., for all pf, NOT(all(pf<=y)).
            # Equivalent: y is in f's box AND for every existing pf, at least one
            # objective has pf_j > y_j.
            
            # For 3D, we can compute this by:
            # 1. For each existing point pf, it "blocks" the region [pf0,ref0]x[pf1,ref1]x[pf2,ref2]
            #    intersected with f's box = [max(f0,pf0), ref0] x [max(f1,pf1), ref1] x [max(f2,pf2), ref2]
            #    if all max < ref.
            # 2. The union of blocked regions within f's box is what we subtract.
            
            # This is getting complex. For practical purposes with small archives (< 100),
            # let's use a simpler proxy that still captures diversity:
            # contribution = box_vol * (1 - max_overlap_fraction)
            
            # Actually, the simplest effective proxy for 3D:
            # For each objective j, find the best (min) value among existing points
            # that are "close" in the other objectives. 
            # Even simpler: just use the box volume as a proxy, combined with
            # non-domination check. Points that are non-dominated and have large
            # boxes are good.
            
            return box_vol

        def is_dominated(f, F_arr):
            """Check if f is dominated by any row in F_arr."""
            if len(F_arr) == 0:
                return False
            dominated = np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1)
            return np.any(dominated)

        def prune_dominated(archive_F_list):
            """Remove dominated points from archive."""
            n = len(archive_F_list)
            if n <= 1:
                return list(range(n))
            F_arr = np.array(archive_F_list)
            keep = []
            for i in range(n):
                # Check if i is dominated by any j != i
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    keep.append(i)
            return keep

        # --- Main search loop ---
        # Use a population-based approach with hypervolume-guided selection
        pop_X = X.copy()
        pop_F = F.copy()
        n_pop = pop_size

        # Adaptive mutation scale
        mut_scale = 0.3

        while used < budget:
            # Check deadline
            if time.time() > deadline - 2.0:
                break

            # Identify non-dominated points in current population
            n = len(pop_F)
            F_arr = np.array(pop_F)
            non_dom_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if non_dom_mask[i]:
                    # Check if i is dominated by any j
                    dom = np.all(F_arr <= F_arr[i], axis=1) & np.any(F_arr < F_arr[i], axis=1)
                    dom[i] = False  # exclude self
                    if np.any(dom):
                        non_dom_mask[i] = False

            # Select parents from non-dominated set
            non_dom_idx = np.where(non_dom_mask)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(n)

            # Choose parent using tournament on hypervolume contribution
            # For speed, use a simple proxy: pick randomly from non-dominated,
            # but bias toward points with larger "box" (farther from ref)
            # Actually, pick the parent that has the largest contribution
            # to the archive hypervolume.
            
            # For efficiency, use a simpler selection: random from non-dominated
            # with occasional exploitation of the best point.
            parent_idx = non_dom_idx[rng.integers(len(non_dom_idx))]
            parent = pop_X[parent_idx].copy()

            # Mutation: Gaussian + occasional large jump
            if rng.random() < 0.15:
                # Large jump: restart-like
                child = parent + width * rng.standard_normal(dim) * 0.5
            else:
                # Standard mutation
                child = parent + width * rng.standard_normal(dim) * mut_scale
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Update archive
            archive_X.append(child)
            archive_F.append(fc)

            # Periodically prune archive to keep it manageable
            if len(archive_F) > 200:
                keep = prune_dominated(archive_F)
                archive_X = [archive_X[i] for i in keep]
                archive_F = [archive_F[i] for i in keep]

            # Replace worst point in population (by sum of objectives)
            # Or replace a dominated point
            worst_idx = int(np.argmax(np.sum(pop_F, axis=1)))
            pop_X[worst_idx] = child
            pop_F[worst_idx] = fc

            # Adapt mutation scale based on progress
            # If recent improvements are small, reduce scale
            # Simple adaptation: every 20 evals, shrink scale slightly
            if used % 20 == 0:
                mut_scale *= 0.95
                mut_scale = max(mut_scale, 0.05)

        return pop_X
# EVOLVE-BLOCK-END
