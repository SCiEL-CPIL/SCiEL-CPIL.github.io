"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Hypervolume-Guided Search (HGS)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, aim for 320s to be safe.
        deadline = start_time + 320.0
        
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1705.0, 12.0, 0.3])
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Initialize archive with a diverse set of points (Latin Hypercube or similar)
        # Use a fixed seed for reproducibility of the initial spread, 
        # though the evaluator seeds np.random, we use a local rng for our internal logic 
        # to ensure the *structure* of the search is deterministic relative to the problem,
        # but we must respect the evaluator's seed for any stochasticity we expose?
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)...".
        # It does not explicitly say we must use np.random for our internal randomness,
        # but it's good practice to be deterministic if possible, or use the global seed.
        # However, the previous program used np.random.default_rng(0).
        # Let's use a local rng seeded by the global state to be safe, or just default_rng(0) 
        # if we want the same initial points every time regardless of seed?
        # Actually, if we use default_rng(0), the initial points are the same for seed 0 and 1.
        # The difference between seed 0 and 1 in the evaluator only matters if we use np.random.
        # Let's use np.random to respect the "seed" aspect if it implies stochastic variation is desired/allowed.
        # But for HGS, determinism is often better. Let's use a local rng but seed it from np.random to mix.
        rng = np.random.default_rng(np.random.randint(1e9))

        n_init = 10
        # Generate initial points
        X = np.zeros((n_init, dim))
        F = np.zeros((n_init, n_obj))
        for i in range(n_init):
            x = lo + span * rng.random(dim)
            X[i] = x
            F[i] = problem(x)
        
        used = n_init
        archive_X = X.copy()
        archive_F = F.copy()

        # Helper to calculate Hypervolume
        # Since dim=3, we can use an exact O(N^2) or O(N log N) algorithm.
        # For N<=200, a simple recursive or iterative exact HV calc is fast enough.
        def calc_hv(points, ref):
            """Calculate exact hypervolume for 3D points."""
            # Filter points that are dominated by ref (not strictly better)
            # A point contributes if it is strictly better than ref in all objectives.
            # Since we are minimizing, p < ref is the condition.
            valid = np.all(points < ref, axis=1)
            if not np.any(valid):
                return 0.0
            pts = points[valid]
            
            # Sort by first objective (f0)
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            
            # For 3D, we can project to 2D for each layer.
            # HV = sum over layers of (width in f0) * HV_2D(layer)
            # Layer i contains points where f0 is in (pts[i-1,0], pts[i,0]]?
            # Actually, standard algorithm:
            # Sort by f0 ascending.
            # Keep track of the "skyline" in the remaining dimensions.
            
            # Simpler for small N:
            # HV = sum of volumes of boxes.
            # For each point p, the volume it contributes is the volume of the box
            # [p[0], ref[0]] x [p[1], ref[1]] x [p[2], ref[2]]
            # MINUS the volume of boxes that dominate it in the other dimensions?
            # No, that's complex.
            
            # Let's use the standard recursive inclusion-exclusion or the layering method.
            # Layering method for 3D:
            # Sort by f0.
            # The HV is the sum over i of (f0[i] - f0[i-1]) * HV_2D(points with f0 <= f0[i])
            # Wait, if we sort ascending f0, the "width" in f0 for the i-th point (0-indexed)
            # is f0[i] - f0[i-1] (with f0[-1] = -inf or 0? No, ref is the upper bound).
            # Actually, the box for point i extends from f0[i] to ref[0].
            # The volume is (ref[0] - f0[i]) * HV_2D( {p_j : f0[j] <= f0[i]} )?
            # No, that counts overlapping regions.
            
            # Correct Layering:
            # Sort points by f0 ascending: p_1, ..., p_n.
            # Let S_k be the set of points {p_1, ..., p_k}.
            # The hypervolume is sum_{k=1}^n (f0[k] - f0[k-1]) * HV_2D(S_k)
            # where f0[0] = 0 (or -inf, but since we are minimizing and ref is finite, we can use 0 if all f0>0? No, use the min f0? No.)
            # Actually, the interval in f0 for the k-th layer is (f0[k-1], f0[k]].
            # The volume added in this layer is the width (f0[k] - f0[k-1]) times the 2D HV of the projection of S_k onto f1-f2 plane.
            # Base case: HV_2D(empty) = 0.
            # We need HV_2D of the set of points projected to (f1, f2).
            # For 2D, HV is sum of rectangles.
            
            # Let's implement HV_2D for a set of points (f1, f2) with ref (ref1, ref2).
            def hv_2d(pts_2d, ref_2d):
                # pts_2d: (N, 2)
                # ref_2d: (2,)
                # Filter valid
                valid = np.all(pts_2d < ref_2d, axis=1)
                if not np.any(valid):
                    return 0.0
                pts = pts_2d[valid]
                # Sort by f1 ascending
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
                
                # The 2D HV is the area of the union of rectangles [p[0], ref[0]] x [p[1], ref[1]].
                # We can sweep from left to right (increasing f1).
                # Maintain a list of active rectangles' right boundaries? No, we need the union area.
                # For 2D, we can just compute the union of rectangles.
                # Sort by x1 (f1).
                # Keep track of the maximum y (f2) seen so far? No.
                # The area is sum of (x_next - x_curr) * (ref_y - min_y_so_far)? No.
                # Let's use a simple O(N^2) for 2D:
                # For each point i, the area it contributes is (ref[0] - pts[i,0]) * (ref[1] - pts[i,1])
                # minus the area covered by points that dominate it in the other dimension?
                # Actually, for 2D, if we sort by x, the contribution of point i is:
                # width = ref[0] - pts[i,0]
                # height = ref[1] - max(pts[j,1] for j < i where pts[j,0] < pts[i,0]?)
                # No, it's easier to just compute the union of rectangles.
                # Since N is small, let's do:
                # Area = 0
                # For each point i:
                #   The rectangle is [x_i, ref_x] x [y_i, ref_y].
                #   We subtract the area of this rectangle that is already covered by previous points.
                #   This is complex.
                
                # Alternative 2D HV:
                # Sort by x ascending.
                # prev_y = ref[1]
                # area = 0
                # for i in range(len(pts)):
                #   x_i, y_i = pts[i]
                #   if y_i < prev_y:
                #     area += (ref[0] - x_i) * (prev_y - y_i)
                #     prev_y = y_i
                #   # else, this point is dominated in y by a previous point with smaller x?
                #   # Wait, if y_i >= prev_y, it doesn't add area?
                #   # Yes, because the rectangle [x_i, ref_x] x [y_i, ref_y] is a subset of [x_prev, ref_x] x [y_prev, ref_y] if x_prev < x_i and y_prev < y_i.
                #   # But we sorted by x, so x_prev <= x_i.
                #   # If y_i >= prev_y, then the new rectangle is "above" the current skyline, so it adds no area.
                #   # If y_i < prev_y, it adds a strip of height (prev_y - y_i) from x_i to ref_x.
                #   # Wait, is it always from x_i to ref_x?
                #   # Yes, because there are no points with x between x_i and ref_x that have y < y_i?
                #   # Not necessarily. But since we process in increasing x, any point with x > x_i is processed later.
                #   # So the "skyline" y is non-increasing as x increases?
                #   # No, the skyline y (max y for a given x) is non-increasing?
                #   # Actually, the set of non-dominated points in 2D, sorted by x, has strictly decreasing y.
                #   # So the algorithm holds.
                
                # Let's verify:
                # Points: (1, 10), (2, 5), (3, 2). Ref: (10, 10).
                # Sort: (1,10), (2,5), (3,2).
                # prev_y = 10.
                # i=0: y=10. y < 10? No (equal). Area += 0. prev_y = 10.
                # i=1: y=5. 5 < 10. Area += (10-2)*(10-5) = 8*5=40. prev_y=5.
                # i=2: y=2. 2 < 5. Area += (10-3)*(5-2) = 7*3=21. prev_y=2.
                # Total = 61.
                # Manual check:
                # Rect 1: [1,10]x[10,10] -> area 0.
                # Rect 2: [2,10]x[5,10] -> area 40.
                # Rect 3: [3,10]x[2,10] -> area 49.
                # Union:
                # x in [1,2]: y in [10,10] -> 0
                # x in [2,3]: y in [5,10] -> 5*1=5
                # x in [3,10]: y in [2,10] -> 8*7=56
                # Total 61. Correct.
                
                area = 0.0
                prev_y = ref_2d[1]
                for i in range(len(pts)):
                    y_i = pts[i, 1]
                    if y_i < prev_y:
                        area += (ref_2d[0] - pts[i, 0]) * (prev_y - y_i)
                        prev_y = y_i
                return area

            # Back to 3D
            # Sort by f0
            idx = np.argsort(pts[:, 0])
            pts = pts[idx]
            
            # We need HV_2D of the set of points seen so far.
            # We can maintain the list of non-dominated points in 2D (f1, f2) as we add points?
            # Or just recompute HV_2D for the subset of points with f0 <= current f0.
            # Since N is small (<=200), O(N^3) might be okay? 200^3 = 8e6. In Python, this is too slow.
            # We need O(N^2) or O(N log N).
            
            # Let's maintain the 2D skyline.
            # As we iterate f0 ascending, we add points to the 2D set.
            # We need the HV_2D of the current 2D set.
            # We can update the 2D HV incrementally?
            # Or just recompute HV_2D on the growing list.
            # If we recompute, it's O(N^2) per step, total O(N^3).
            # 200^3 = 8,000,000 operations. In Python, ~1s? Maybe too slow if called 200 times (800s).
            # We need a faster HV calc.
            
            # Let's optimize HV_2D to O(N log N) or O(N).
            # The sweep line above is O(N) if the points are sorted by x.
            # So if we maintain the 2D points sorted by f1, we can compute HV_2D in O(N).
            # But adding a point to a sorted list is O(N).
            # So total O(N^2) for the whole 3D HV calculation.
            # 200^2 = 40,000. This is fast.
            
            # Algorithm:
            # 1. Sort 3D points by f0.
            # 2. Initialize an empty list `active_2d` for (f1, f2) points.
            # 3. Initialize `prev_f0 = pts[0, 0]`? No, we need the width.
            #    The width for the first point is `ref[0] - pts[0, 0]`?
            #    No, the layering formula:
            #    HV = sum_{i=0}^{N-1} (f0[i] - f0[i-1]) * HV_2D(S_i)
            #    where S_i is the set of points {p_0, ..., p_i}.
            #    And f0[-1] is... what?
            #    Actually, the volume is the union of boxes.
            #    Box i is [f0[i], ref[0]] x [f1[i], ref[1]] x [f2[i], ref[2]].
            #    The "new" volume added by point i (assuming f0 sorted asc) is:
            #    (ref[0] - f0[i]) * HV_2D( {p_j : f0[j] <= f0[i]} ) - (ref[0] - f0[i-1]) * HV_2D( {p_j : f0[j] <= f0[i-1]} )?
            #    No, that's not right.
            
            # Correct Formula:
            # Sort by f0 ascending: p_1, ..., p_n.
            # Let S_k = {p_1, ..., p_k}.
            # HV = sum_{k=1}^n (f0[k] - f0[k-1]) * HV_2D(S_k)
            # where f0[0] = 0 (or -inf, but effectively the lower bound of the domain, or we can just say the first term is f0[1] * HV_2D(S_1) if we assume f0[0]=0? No.)
            # Actually, the interval for the k-th point is (f0[k-1], f0[k]].
            # The volume in this slab is (f0[k] - f0[k-1]) * HV_2D(S_k).
            # For k=1, the slab is (0, f0[1]]? No, (-inf, f0[1]]?
            # Since we are minimizing, the box extends to ref.
            # The "width" in f0 for the first point is ref[0] - f0[1]? No.
            
            # Let's stick to the definition:
            # HV is the volume of the union of boxes B_i = [p_i, ref].
            # Sort by p_i[0] ascending.
            # The union can be decomposed into slabs perpendicular to f0 axis.
            # Slab k is defined by f0 in (p_{k-1}[0], p_k[0]].
            # In this slab, the cross-section is the 2D HV of the set of points that have f0 <= p_k[0].
            # Wait, if a point has f0 > p_k[0], its box does not extend into this slab.
            # So the cross-section is indeed the 2D HV of {p_j : p_j[0] <= p_k[0]}.
            # The width of slab k is p_k[0] - p_{k-1}[0].
            # For the last slab (k=n), the width is ref[0] - p_n[0].
            # So:
            # HV = sum_{k=1}^{n-1} (p_k[0] - p_{k-1}[0]) * HV_2D(S_k) + (ref[0] - p_n[0]) * HV_2D(S_n)
            # where S_k = {p_1, ..., p_k}.
            
            # Let's implement this.
            # We need to maintain S_k and compute HV_2D(S_k) efficiently.
            # We can just recompute HV_2D on the list of (f1, f2) for S_k.
            # Since we add one point at a time, and HV_2D is O(N log N) or O(N) if sorted,
            # and we do this N times, it's O(N^2 log N) or O(N^2).
            # N=200 -> 40,000 ops. Very fast.
            
            active_2d = [] # List of [f1, f2]
            hv_total = 0.0
            prev_f0 = pts[0, 0]
            
            # We need to handle the first point carefully.
            # The formula above assumes p_0 is the first.
            # Let's iterate k from 0 to N-1.
            # For each point p_k, we add it to active_2d.
            # Then we compute HV_2D(active_2d).
            # The width of the slab ending at p_k[0] is p_k[0] - p_{k-1}[0].
            # For k=0, the slab is (-inf, p_0[0]]? No, the box starts at p_0[0].
            # The volume to the left of p_0[0] is 0 because no box starts before p_0[0]?
            # Actually, the box for p_0 starts at p_0[0].
            # So the slab (p_{-1}, p_0] doesn't exist?
            # Let's use the standard recursive definition or the slab method correctly.
            
            # Slab method:
            # Sort by f0.
            # Let x_0 = -inf, x_{N+1} = ref[0].
            # This is getting complicated.
            
            # Let's just use the O(N^2) pairwise check for HV contribution?
            # No, let's use the slab method with a slight adjustment.
            # HV = sum_{i=0}^{N-1} (ref[0] - p_i[0]) * HV_2D({p_j : p_j[0] <= p_i[0] and p_j is not dominated in 2D by any p_k with p_k[0] < p_i[0]})?
            # No, HV_2D already handles domination.
            
            # Let's try:
            # HV = sum_{i=0}^{N-1} (p_{i+1}[0] - p_i[0]) * HV_2D(S_i)
            # where p_N[0] = ref[0].
            # S_i = {p_0, ..., p_i}.
            # This works if we define p_N = ref.
            
            # Implementation:
            # 1. Sort pts by f0.
            # 2. Create a list of f0 values: [p_0[0], p_1[0], ..., p_{N-1}[0], ref[0]].
            # 3. Iterate i from 0 to N-1.
            #    Add p_i[1:] to active_2d.
            #    Compute HV_2D(active_2d).
            #    width = f0_vals[i+1] - f0_vals[i].
            #    hv_total += width * hv_2d_val.
            
            # This is O(N^2) because HV_2D is O(N) (if we maintain sorted order) or O(N log N).
            # Let's just recompute HV_2D from scratch each time for simplicity, since N is small.
            
            pts_sorted = pts[np.argsort(pts[:, 0])]
            f0_vals = np.append(pts_sorted[:, 0], ref[0])
            
            active_2d = []
            for i in range(len(pts_sorted)):
                # Add current point's f1, f2
                active_2d.append(pts_sorted[i, 1:3])
                active_2d_arr = np.array(active_2d)
                hv_2d_val = hv_2d(active_2d_arr, ref[1:3])
                width = f0_vals[i+1] - f0_vals[i]
                hv_total += width * hv_2d_val
                
            return hv_total

        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Select parent: Point with highest marginal HV contribution
            # We approximate marginal contribution by calculating the HV of the archive
            # with and without each point? That's O(N^2 * HV_cost). Too slow.
            # Instead, we use a heuristic: The point that is "most isolated" in the non-dominated set.
            # Or, we can calculate the exact HV contribution for a few candidates.
            # Given N is small, let's try to calculate the exact HV of the archive.
            # Then, for each point in the non-dominated set, remove it and calculate HV.
            # The one with the largest drop in HV is the most important.
            # But we want to *extend* the front, not just preserve it.
            # Usually, we pick a point from the non-dominated set to mutate.
            # Let's pick the non-dominated point with the largest "empty space" around it.
            
            # Calculate non-dominated mask
            dominated = np.zeros(len(archive_F), dtype=bool)
            for i in range(len(archive_F)):
                for j in range(len(archive_F)):
                    if i != j and np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        dominated[i] = True
                        break
            
            if np.any(~dominated):
                non_dom_idx = np.where(~dominated)[0]
                # Pick the one with largest average distance to other non-dominated points
                # This encourages filling gaps.
                if len(non_dom_idx) > 1:
                    F_nd = archive_F[non_dom_idx]
                    # Normalize
                    F_min = F_nd.min(axis=0)
                    F_max = F_nd.max(axis=0)
                    F_range = F_max - F_min
                    F_range[F_range == 0] = 1
                    F_norm = (F_nd - F_min) / F_range
                    avg_dists = np.zeros(len(non_dom_idx))
                    for k in range(len(non_dom_idx)):
                        avg_dists[k] = np.mean(np.linalg.norm(F_norm - F_norm[k], axis=1))
                    best_idx = non_dom_idx[np.argmax(avg_dists)]
                else:
                    best_idx = non_dom_idx[0]
            else:
                best_idx = np.argmin(np.sum(archive_F, axis=1))

            # 2. Generate children around the parent
            # Use a deterministic set of mutations to explore the neighborhood.
            # Coordinate descent steps + random perturbations.
            parent = archive_X[best_idx]
            
            # Define a set of candidate children
            candidates = []
            
            # Small random steps
            for _ in range(3):
                step = 0.05 * span
                child = parent + step * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                candidates.append(child)
                
            # Coordinate steps
            for d in range(dim):
                for direction in [-1, 1]:
                    child = parent.copy()
                    child[d] += direction * 0.05 * span[d]
                    child = np.clip(child, lo, hi)
                    candidates.append(child)
                    
            # Large exploration step
            child = parent + 0.2 * span * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            candidates.append(child)

            # Evaluate candidates and keep the best one (based on HV contribution or just non-domination)
            # To save budget, we evaluate all candidates (up to 8) and keep the best.
            # But we have a budget of 200. 10 init + 190 * 8 = 1530 evaluations. Too many.
            # We must be selective.
            # Let's evaluate only 2-3 candidates per iteration.
            # Or, we can do a local search: evaluate a candidate, if it improves HV, keep it and continue from there.
            
            # Strategy: Evaluate 3 candidates, pick the one that is non-dominated or has best sum.
            # Actually, let's just evaluate 1 candidate per iteration to maximize the number of distinct regions explored?
            # No, HGS usually evaluates a batch and selects the best.
            # Let's evaluate 2 candidates per iteration.
            
            best_child = None
            best_child_f = None
            best_score = -np.inf
            
            # We will evaluate 2 random mutations
            for _ in range(2):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Mix of small and medium steps
                step_size = rng.choice([0.02, 0.05, 0.1])
                child = parent + step_size * span * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                
                # Check if it's non-dominated in the archive
                is_dom = False
                for j in range(len(archive_F)):
                    if np.all(archive_F[j] <= fc) and np.any(archive_F[j] < fc):
                        is_dom = True
                        break
                        
                # Score: prefer non-dominated, then lower sum
                if not is_dom:
                    score = -np.sum(fc)
                else:
                    score = -np.sum(fc) - 1000 # Penalize dominated
                
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_child_f = fc
                    
            if best_child is not None:
                archive_X = np.vstack([archive_X, best_child.reshape(1, -1)])
                archive_F = np.vstack([archive_F, best_child_f.reshape(1, -1)])
                
        return archive_X
# EVOLVE-BLOCK-END
