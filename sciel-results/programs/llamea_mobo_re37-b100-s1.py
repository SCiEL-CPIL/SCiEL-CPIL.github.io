"""Multi-objective optimizer using LHS initialization and Hypervolume-Indicative Evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using:
    1. Latin Hypercube Sampling (LHS) for initial population to ensure spread.
    2. Hypervolume-Indicative Evolutionary Algorithm (HVEA) logic:
       - Selects parents based on their marginal contribution to the hypervolume.
       - Uses DE/rand/1/bin mutation but biases towards high-contribution individuals.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # 15s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume calculation
        # Task says ref point is [1.1, 1.3, 1.1]
        # We use this for internal selection to align with scoring
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Helper: Compute hypervolume contribution of a single point to a set of points
        # This is O(N) for 3D if we sort, but we can do a simpler O(N^2) or O(N) check
        # For 3 objectives, HV contribution of point p to set S (where S is non-dominated)
        # is the volume of the box [p, ref] minus the volume covered by other points in S.
        # A simpler approximation often used in HVEA is to just pick the point that
        # is non-dominated and has the largest "distance" to the ref point in a normalized sense,
        # or simply use the exact HV contribution.
        
        def hv_contribution_single(point, points_list, ref):
            """
            Calculate the hypervolume contribution of a single point to a set of non-dominated points.
            For 3D, we can compute this exactly by checking which sub-boxes are unique to this point.
            However, for speed in a loop, a common heuristic is:
            Contribution = Volume(point, ref) - Volume(union of boxes of other points intersecting with point's box)
            
            Given the small budget (200) and dim (4), we can afford a slightly more expensive
            but accurate check for selection.
            
            Actually, for HVEA selection, we often just rank by the HV of the set if we added the point
            vs if we didn't, but that's expensive.
            
            A faster proxy: 
            1. Check if point is dominated by any in points_list. If yes, contribution 0.
            2. If not, compute the volume of the box defined by point and ref.
            3. Subtract volumes of boxes of other points that are "inside" this box? 
               No, that's not correct for union.
            
            Let's use a robust simple metric: 
            The "niche" value. 
            Or, simply: Select the point that maximizes the HV of the set {point} U {subset of neighbors}.
            
            Given time constraints, let's use a simpler but effective strategy:
            Select the parent that has the largest "space" around it relative to the current front.
            We can approximate this by:
            1. For each candidate parent i in the non-dominated set:
               a. Calculate the volume of the box [f_i, ref].
               b. For each other point j in the non-dominated set:
                  if f_j dominates f_i, contribution is 0 (but we assume non-dominated set).
                  if f_i and f_j overlap, the unique contribution is reduced.
            
            Let's implement a fast exact HV contribution for 3D.
            """
            # If point is worse than ref in any dim, contribution is 0
            if np.any(point >= ref):
                return 0.0
            
            # Volume of the box for this point
            # Box is [point, ref]
            # V = (ref[0]-p[0]) * (ref[1]-p[1]) * (ref[2]-p[2])
            
            # We need to subtract the volume covered by other points in the set.
            # This is the standard "incremental hypervolume" problem.
            # For 3D, we can do this by sorting points by one coordinate and sweeping.
            # But doing this for EVERY candidate parent in EVERY generation is O(N^2 log N).
            # N (archive size) is small (< 100). Budget is 200.
            # So 200 * 100 * 100 * log(100) is ~ 200 * 10000 * 7 = 14M ops. 
            # Python might be slow. Let's optimize.
            
            # Alternative: Pre-compute HV contributions for the current non-dominated set once per generation,
            # then select the best parent from that list.
            
            pass # Placeholder, will implement in main loop logic differently

        def compute_hv_contributions(F_nd, ref):
            """
            Compute the HV contribution of each point in the non-dominated set F_nd.
            F_nd: array of shape (N, 3)
            Returns: array of shape (N,) with contributions.
            """
            N = len(F_nd)
            if N == 0:
                return np.array([])
            
            # For 3D, we can use the following algorithm:
            # Sort points by f0.
            # For each point i, its contribution is the volume of the box [f_i, ref]
            # minus the volume of the union of boxes [f_j, ref] for all j != i that overlap.
            
            # A simpler exact method for 3D:
            # The HV of the set is sum of contributions.
            # We can calculate the total HV of the set, and then the HV of the set without point i.
            # Contribution(i) = HV(S) - HV(S \ {i})
            
            def calc_hv(points, ref):
                """Calculate HV of a set of 3D points."""
                if len(points) == 0:
                    return 0.0
                # Sort by f0
                pts = sorted(points, key=lambda x: x[0])
                # We can use a recursive or sweep line approach.
                # For small N, O(N^2) is fine.
                # Actually, for 3D, we can project to 2D for each "layer".
                
                # Let's use a simple O(N^3) or O(N^2) method suitable for small N.
                # Or even simpler: since N is small, we can just compute the union volume.
                # Union volume of boxes in 3D can be computed by inclusion-exclusion (2^N terms, too slow)
                # or by sweeping.
                
                # Sweep line on x-axis (f0):
                # Events: start of box, end of box.
                # But boxes are all anchored at ref? No, they are [f_i, ref].
                # So they all share the same "top-right" corner.
                # This simplifies things!
                # The union of boxes [f_i, ref] is equivalent to the set of points y such that y <= ref and y >= min(f_i) in some sense? No.
                # A point y is in the union if there exists i such that y >= f_i (element-wise) and y <= ref.
                # This is the same as: y is in the union if y is not dominated by any f_i? No.
                # y is covered if min(f_i[0] <= y[0], f_i[1] <= y[1], f_i[2] <= y[2]) is true for some i.
                
                # Actually, since all boxes share the corner ref, the union volume is simply:
                # Volume of [min(f_i), ref] minus the holes?
                # No, the union of boxes sharing a common corner is NOT just the box of the minima.
                # Example: (0,1,1) and (1,0,1) with ref (2,2,2).
                # Box 1: [0,2]x[1,2]x[1,2]
                # Box 2: [1,2]x[0,2]x[1,2]
                # Union is not [0,2]x[0,2]x[1,2] because the corner (0.5, 0.5, 1.5) is not in either.
                
                # So we must compute the union volume properly.
                # Given N is small (< 50), we can use a recursive decomposition or just a grid? No grid.
                # Let's use the standard 3D union volume algorithm.
                
                # Sort by x.
                # For each interval between x-coordinates, the cross-section in y-z plane is a union of rectangles.
                
                xs = sorted(set([p[0] for p in points] + [ref[0]]))
                total_vol = 0.0
                
                for k in range(len(xs) - 1):
                    x1, x2 = xs[k], xs[k+1]
                    if x1 >= x2: continue
                    dx = x2 - x1
                    # Which points cover this x-slice?
                    # A point p covers this slice if p[0] <= x1 (since box starts at p[0])
                    # Wait, box is [p[0], ref[0]]. So it covers x if p[0] <= x <= ref[0].
                    # Since x is in [x1, x2] and x2 <= ref[0], we just need p[0] <= x1.
                    
                    active_pts = [p for p in points if p[0] <= x1]
                    if not active_pts:
                        continue
                    
                    # Now compute area of union of rectangles [p[1], ref[1]] x [p[2], ref[2]]
                    # in the y-z plane.
                    area = union_area_2d(active_pts, ref)
                    total_vol += dx * area
                    
                return total_vol

            def union_area_2d(points, ref):
                """Union area of rectangles [p[1], ref[1]] x [p[2], ref[2]]."""
                if not points:
                    return 0.0
                
                ys = sorted(set([p[1] for p in points] + [ref[1]]))
                total_area = 0.0
                
                for k in range(len(ys) - 1):
                    y1, y2 = ys[k], ys[k+1]
                    if y1 >= y2: continue
                    dy = y2 - y1
                    # Active points for this y-slice: p[1] <= y1
                    active_pts_y = [p for p in points if p[1] <= y1]
                    if not active_pts_y:
                        continue
                    
                    # Now compute length of union of intervals [p[2], ref[2]]
                    # This is 1D union length.
                    intervals = [(p[2], ref[2]) for p in active_pts_y]
                    # Sort and merge
                    intervals.sort()
                    if not intervals:
                        continue
                    merged = [intervals[0]]
                    for i in range(1, len(intervals)):
                        last = merged[-1]
                        curr = intervals[i]
                        if curr[0] <= last[1]:
                            merged[-1] = (last[0], max(last[1], curr[1]))
                        else:
                            merged.append(curr)
                    
                    len_sum = sum(m[1] - m[0] for m in merged)
                    total_area += dy * len_sum
                    
                return total_area

            # Calculate HV for full set
            hv_full = calc_hv(F_nd, ref)
            
            contributions = np.zeros(N)
            for i in range(N):
                # Remove point i
                S_i = [F_nd[j] for j in range(N) if j != i]
                hv_i = calc_hv(S_i, ref)
                contributions[i] = hv_full - hv_i
                
            return contributions

        # 1. Initialization with LHS
        # We want a population that covers the space well.
        # Let's use a population size of 50.
        pop_size = 50
        if pop_size > budget:
            pop_size = budget
            
        # LHS generation
        # For each dimension, split [0,1] into pop_size equal intervals.
        # Assign one sample from each interval to each dimension, then shuffle.
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            # Generate LHS samples for this dimension
            # np.random.permutation(pop_size) gives a permutation of 0..pop_size-1
            # (i + rand) / pop_size gives uniform in [i/pop_size, (i+1)/pop_size]
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.random(pop_size)) / pop_size
        
        # Scale to bounds
        X = lo + X * (hi - lo)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Maintain a list of all evaluated points (x, f)
        # We will keep the non-dominated set in a list for selection
        all_X = X.copy()
        all_F = F.copy()
        
        # Find initial non-dominated set
        def get_nondominated_idx(F_list):
            N = len(F_list)
            nd_idx = []
            for i in range(N):
                is_dom = False
                for j in range(N):
                    if i == j: continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nd_idx.append(i)
            return nd_idx

        nd_indices = get_nondominated_idx(all_F)
        archive_X = [all_X[i] for i in nd_indices]
        archive_F = [all_F[i] for i in nd_indices]
        
        # 2. Evolution Loop
        # We will use the archive to select parents.
        # Strategy:
        # 1. Calculate HV contributions for the current archive.
        # 2. Select 3 parents: 
        #    - One from the top 30% of HV contributions (exploit good spots)
        #    - Two from the bottom 70% (explore new spots) or random from archive?
        #    Standard HVEA picks the one with highest contribution as the "best" parent.
        #    Let's pick:
        #    Parent 1: Highest HV contribution
        #    Parent 2: Random from archive
        #    Parent 3: Random from archive
        #    Or:
        #    Parent 1: Highest HV contribution
        #    Parent 2: Second highest
        #    Parent 3: Third highest
        #    This might be too greedy. Let's mix.
        #    P1: Top contributor.
        #    P2, P3: Random from archive.
        
        CR = 0.9
        F_base = 0.8
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # If archive is empty (shouldn't happen), restart
            if len(archive_F) == 0:
                # Just random sample
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                used += 1
                all_X = np.vstack([all_X, x])
                all_F = np.vstack([all_F, f])
                continue
            
            # Calculate HV contributions for archive
            # This is the expensive part. 
            # If archive is large, this slows down. 
            # Limit archive size to 50 for selection efficiency?
            # If archive > 50, we can prune it to the best 50 based on HV contribution.
            
            if len(archive_F) > 60:
                # Prune archive to keep only the most contributing 50 points
                # We need contributions to prune, so we calculate them.
                # To save time, we can just keep the first 50 or random 50?
                # Better: calculate contributions, sort, keep top 50.
                contribs = compute_hv_contributions(archive_F, ref_point)
                top_idx = np.argsort(contribs)[-50:]
                archive_X = [archive_X[i] for i in top_idx]
                archive_F = [archive_F[i] for i in top_idx]
                # Note: We don't update all_X/all_F here, but that's fine for selection.
                # The "history" is all_X/all_F, but we only select parents from archive.
            
            if len(archive_F) > 0:
                contribs = compute_hv_contributions(archive_F, ref_point)
                
                # Select parents
                # P1: Index of max contribution
                idx_best = np.argmax(contribs)
                p1 = archive_X[idx_best]
                
                # P2, P3: Random from archive
                if len(archive_X) > 2:
                    other_indices = np.setdiff1d(np.arange(len(archive_X)), idx_best)
                    idx_p2 = other_indices[np.random.randint(len(other_indices))]
                    idx_p3 = other_indices[np.random.randint(len(other_indices))]
                    # Ensure distinct
                    if idx_p2 == idx_p3:
                         idx_p3 = (idx_p3 + 1) % len(archive_X)
                         if idx_p3 == idx_best:
                             idx_p3 = (idx_p3 + 1) % len(archive_X)
                else:
                    # Fallback if archive is tiny
                    idx_p2 = np.random.randint(len(archive_X))
                    idx_p3 = np.random.randint(len(archive_X))
                    while idx_p3 == idx_p2:
                        idx_p3 = np.random.randint(len(archive_X))
                
                p2 = archive_X[idx_p2]
                p3 = archive_X[idx_p3]
                
                # Mutation
                F_curr = F_base + 0.2 * np.random.random()
                v = p1 + F_curr * (p2 - p3)
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = p1[j]
                
                trial = np.clip(trial, lo, hi)
                
                f_trial = problem(trial)
                used += 1
                
                # Update all history
                all_X = np.vstack([all_X, trial])
                all_F = np.vstack([all_F, f_trial])
                
                # Update Archive
                # Check if trial is dominated by any in archive
                is_dominated = False
                dominated_indices = []
                
                for i, f_arch in enumerate(archive_F):
                    if np.all(f_arch <= f_trial) and np.any(f_arch < f_trial):
                        is_dominated = True
                        break
                    elif np.all(f_trial <= f_arch) and np.any(f_trial < f_arch):
                        dominated_indices.append(i)
                
                if not is_dominated:
                    # Remove dominated
                    for i in sorted(dominated_indices, reverse=True):
                        archive_X.pop(i)
                        archive_F.pop(i)
                    
                    # Add trial
                    archive_X.append(trial)
                    archive_F.append(f_trial)
                    
                    # If archive is getting too big, we might want to prune more aggressively
                    # But the loop above handles > 60.
        
        return all_X
# EVOLVE-BLOCK-END
