"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer using Sobol sampling and hypervolume-based local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. 
        # We aim to finish well before that to ensure validity.
        deadline = start_time + 290 
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Phase 1: Sobol Quasi-Random Sampling for uniform coverage
        # We want 150 points. 
        # Sobol sequence requires power of 2 for best properties, but works for others.
        # 128 is a good power of 2 close to 150.
        n_sobol = 128
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Generate points in [0, 1]^dim
        sobol_samples = sampler.random(n_sobol)
        
        # Map to bounds
        sobol_samples_mapped = lo + sobol_samples * span
        
        # Evaluate all Sobol points
        # We need to keep track of all evaluated points and their objective values
        # to calculate hypervolume later.
        # The evaluator records every call to problem(x).
        
        # We will store our own internal representation to make decisions for Phase 2.
        # internal_x: list of x vectors
        # internal_f: list of f vectors
        
        internal_x = []
        internal_f = []
        
        def add_point(x, f):
            internal_x.append(x)
            internal_f.append(f)

        # Evaluate Sobol points
        for x in sobol_samples_mapped:
            if time.time() > deadline:
                break
            f = problem(x)
            add_point(x, f)
            
        # If we haven't used all 150, fill with random to hit the target count?
        # The directive says "first 150 evaluations". 
        # Let's add random points to reach 150 if we have budget.
        target_phase1 = 150
        current_evals = len(internal_x)
        
        # Use a fixed seed for reproducibility of the random part if needed, 
        # but np.random.seed is set by evaluator. We should probably use np.random directly 
        # or a local RNG. The prompt says evaluator calls np.random.seed(seed).
        # So using np.random.random is consistent with the environment state.
        
        while current_evals < target_phase1 and time.time() < deadline:
            x = lo + np.random.random(self.dim) * span
            f = problem(x)
            add_point(x, f)
            current_evals += 1
            
        # Phase 2: Hypervolume-based Local Search
        # Remaining budget
        remaining_budget = self.budget - current_evals
        
        # Reference point for hypervolume calculation
        # Task says: reference point [1705.0, 12.0, 0.3]
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # We need a function to calculate the hypervolume contribution of a point
        # given a set of other points.
        # For 3 objectives, we can compute the volume of the box dominated by a point
        # that is not dominated by any other point in the set.
        
        def dominated_by(p, others):
            # p is a point (3 obj), others is a list of points (N, 3)
            # Returns True if any point in others dominates p
            # Dominance: a <= b for all objectives, and a < b for at least one
            if len(others) == 0:
                return False
            # Check if any point in others is <= p and < p
            # others[i] <= p and others[i] != p
            leq = np.all(others <= p, axis=1)
            lt = np.any(others < p, axis=1)
            return np.any(leq & lt)
            
        def hypervolume_contribution(p, others, ref):
            # Calculate the volume of the box defined by p and ref, 
            # minus the volume covered by other points that also dominate this region?
            # Actually, the marginal contribution of p to the hypervolume of S ∪ {p}
            # is the volume of the region dominated by p but not by any point in S.
            # For 3D, this is complex to compute exactly for arbitrary sets.
            # However, a simpler heuristic often used:
            # 1. Calculate the volume of the box (p, ref).
            # 2. Subtract volumes of boxes of other points that overlap? 
            # This is an approximation. 
            # A better approximation for small sets:
            # The contribution is the volume of the union of boxes of S ∪ {p} minus union of S.
            # Computing union of boxes in 3D is non-trivial but doable for small N.
            # However, with N up to 150, O(N^2) or O(N^3) might be too slow if done repeatedly.
            
            # Let's use a simpler proxy: 
            # The "potential" contribution is the volume of the box (p, ref) 
            # weighted by how much of it is "uncovered" by others.
            # Actually, let's just use the volume of the box (p, ref) as a base,
            # but prioritize points that are non-dominated.
            # If p is dominated by S, its marginal contribution is 0.
            # If p is non-dominated, its marginal contribution is positive.
            
            if dominated_by(p, np.array(others) if others else []):
                return 0.0
                
            # Box volume for p
            box_vol = 0.0
            for i in range(self.n_obj):
                if ref[i] > p[i]:
                    box_vol *= (ref[i] - p[i])
                else:
                    box_vol = 0.0
                    break
            
            # To get a better estimate of marginal contribution, we can subtract
            # the volume of the intersection of (p, ref) with the union of (q, ref) for q in S.
            # This is expensive. 
            # Alternative: Use the fact that if p is non-dominated, it adds some volume.
            # We can approximate by checking how many points in S are "close" to p in a way that they cover its volume.
            # For the purpose of selecting the point to replace, we can just use the box volume
            # as a heuristic for "value" and rely on the non-domination check.
            # But we want to find the point with the LOWEST marginal contribution to remove/replace.
            # The points with the lowest marginal contribution are usually those that are
            # "redundant" or close to others.
            
            # Let's refine: The marginal contribution of p is the volume of R_p \ ∪_{q in S} R_q,
            # where R_p is the box dominated by p and bounded by ref.
            # For 3D, we can compute this exactly for a small set of "relevant" neighbors.
            # But with 150 points, it's heavy.
            
            # Heuristic: 
            # Contribution is high if p is far from other points in the objective space.
            # Let's use the box volume as a proxy for "potential" and penalize if it's close to others.
            # Actually, if p is non-dominated, it always adds *some* volume.
            # The points that add the *least* volume are those that are "tightly packed".
            
            # Let's compute the exact marginal contribution for the top few candidates?
            # No, we need to rank all 150 to find the min.
            
            # Let's try a simpler approach for ranking:
            # 1. If dominated, contribution = 0.
            # 2. If non-dominated, contribution ≈ Box Volume * (1 / (1 + num_close_neighbors))
            # This is a rough heuristic.
            
            # A better heuristic for "lowest marginal contribution":
            # Points that are very close to other non-dominated points have low marginal contribution.
            # Let's calculate the distance to the nearest other non-dominated point.
            
            return box_vol

        # We want to identify the point with the lowest marginal hypervolume contribution.
        # To do this efficiently, we first filter for non-dominated points.
        # Only non-dominated points can have positive marginal contribution.
        # Among non-dominated points, those with smaller "box volume" or those that are
        # "covered" by others (in a spatial sense) are less valuable.
        
        # Let's identify the Pareto front of the current set.
        f_array = np.array(internal_f)
        n_points = len(f_array)
        
        # Find non-dominated points
        is_nondominated = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            dominated = False
            for j in range(n_points):
                if i == j: continue
                # Check if j dominates i
                if np.all(f_array[j] <= f_array[i]) and np.any(f_array[j] < f_array[i]):
                    dominated = True
                    break
            if not dominated:
                is_nondominated[i] = True
                
        # If there are no non-dominated points (shouldn't happen), pick random.
        if not np.any(is_nondominated):
            target_idx = np.random.randint(n_points)
        else:
            # Among non-dominated points, we want to find the one with the lowest marginal contribution.
            # Approximation: The one with the smallest "box volume" is often the one that
            # is closest to the reference point (worst in terms of volume added) or 
            # actually, points closer to the origin (better objectives) have LARGER boxes.
            # Wait. The box is from p to ref.
            # If p is very good (small values), the box is large.
            # If p is bad (large values, close to ref), the box is small.
            # So points close to the reference point have small box volumes.
            # But are they low contribution?
            # If p is close to ref, it dominates a small volume.
            # If that small volume is already covered by another point q which is better than p,
            # then p's marginal contribution is 0 (it's dominated).
            # But we already filtered for non-dominated.
            # So for non-dominated points, p is not dominated.
            # Therefore, p must dominate some volume not dominated by others.
            # The marginal contribution is the volume of that unique region.
            # 
            # Heuristic: 
            # Points that are "extreme" (vertices of the convex hull of the front) usually have high contribution.
            # Points on the "edges" or "interior" of the front might have lower contribution.
            # 
            # Let's use the box volume as a proxy for the *maximum possible* contribution.
            # Points with small box volumes are candidates for low contribution.
            # However, a point with a large box volume could still have low marginal contribution
            # if most of its box is covered by others.
            
            # Let's calculate the actual marginal contribution for the non-dominated points.
            # Since the set of non-dominated points is likely small (10-30), we can afford O(K^2) or O(K^3).
            
            nondom_indices = np.where(is_nondominated)[0]
            nondom_f = f_array[nondom_indices]
            K = len(nondom_f)
            
            contributions = np.zeros(K)
            
            # For each non-dominated point i, calculate its marginal contribution
            # to the hypervolume of the set of non-dominated points.
            # Note: Points that are dominated by the *entire* set are already removed.
            # But a point can be non-dominated yet have its volume largely covered by others.
            
            # Algorithm for 3D hypervolume contribution of point i in set S:
            # 1. Define box B_i = [f_i, ref].
            # 2. The marginal contribution is Vol(B_i) - Vol(B_i ∩ (∪_{j != i} B_j)).
            # 3. Since all points are non-dominated, B_i is not fully contained in any B_j.
            # 4. We can compute the union of boxes of other points intersected with B_i.
            
            # This is still complex. 
            # Let's use a simpler iterative removal heuristic:
            # The point with the lowest "regret" is the one whose removal causes the least drop in hypervolume.
            # We can estimate this by looking at the "density" of points around it.
            
            # Let's just use the box volume as a primary sort key for non-dominated points,
            # but invert it? No, we want to remove the LEAST valuable one.
            # Least valuable = Smallest unique volume.
            # 
            # Let's compute the exact hypervolume of the set S, and then the hypervolume of S \ {i} for each i?
            # That's K * HV(S\{i}). HV in 3D is O(N log^2 N) or similar.
            # K is small. This is feasible.
            
            def calc_hypervolume_3d(points, ref):
                # points: (N, 3) array of objective values
                # ref: (3,) array
                # Returns the hypervolume dominated by the set of points, bounded by ref.
                # Only points strictly better than ref in all objectives count.
                # Filter points
                valid = np.all(points < ref, axis=1)
                points = points[valid]
                if len(points) == 0:
                    return 0.0
                
                # Sort by first objective
                order = np.argsort(points[:, 0])
                points = points[order]
                
                # We can use a sweep line or recursive decomposition.
                # For small N, a simple recursive approach works.
                # HV(S) = sum of contributions.
                # Contribution of point i (sorted by f0 ascending) is:
                # (f0[i+1] - f0[i]) * HV_2D(S_i, ref_2)
                # where S_i is the set of points in S with f0 <= f0[i] (including i)
                # and HV_2D is the 2D hypervolume of the projection.
                
                # Let's implement a simple recursive HV for 3D.
                def hv_2d(pts, ref_2d):
                    # pts: (M, 2)
                    # ref_2d: (2,)
                    if len(pts) == 0:
                        return 0.0
                    order = np.argsort(pts[:, 0])
                    pts = pts[order]
                    area = 0.0
                    max_y = -np.inf
                    for p in pts:
                        if p[1] < ref_2d[1]:
                            # Width is difference in x
                            # We need the union of rectangles.
                            # Sweep line:
                            pass
                    # Simpler 2D HV:
                    # Sort by x. 
                    # Area = sum_{i} (x[i+1]-x[i]) * max(y[0..i])
                    # Careful with boundaries.
                    sorted_pts = pts
                    current_max_y = -np.inf
                    total_area = 0.0
                    for i in range(len(sorted_pts)):
                        x_i = sorted_pts[i][0]
                        y_i = sorted_pts[i][1]
                        if y_i > current_max_y:
                            current_max_y = y_i
                        # Next x
                        if i < len(sorted_pts) - 1:
                            x_next = sorted_pts[i+1][0]
                        else:
                            x_next = ref_2d[0]
                        
                        if current_max_y > -np.inf and x_next > x_i:
                            # Height is min(current_max_y, ref_2d[1]) - 0? 
                            # No, the box is [0, ref] or [pts, ref]?
                            # The standard HV definition: Volume of points dominated by S and bounded by ref.
                            # If objectives are to be minimized, the dominated region is [0, ref] intersected with union of boxes [0, p].
                            # Wait, the points are the objectives. The box dominated by p is [0, p] if we assume 0 is the ideal.
                            # But the problem doesn't specify 0 is the ideal. It specifies a reference point.
                            # The hypervolume is usually defined relative to the reference point.
                            # Vol = measure of { y | y <= p for some p in S, y <= ref }
                            # This is the union of boxes [0, p] intersected with [0, ref].
                            # So yes, we are calculating the volume of the union of boxes from origin to p, clipped by ref.
                            
                            height = min(current_max_y, ref_2d[1])
                            if height > 0:
                                total_area += (x_next - x_i) * height
                    return total_area

                # 3D HV using sweep line on first dimension
                total_vol = 0.0
                current_max_yz_area = 0.0 # This is not enough, we need the shape of the 2D cross section.
                
                # Actually, the sweep line method for 3D:
                # Sort points by f0.
                # Maintain a data structure for the 2D hypervolume of the active set (points with f0 <= current f0).
                # When we pass a point, we add it to the active set.
                # The volume contribution of the slice between f0[i] and f0[i+1] is (f0[i+1]-f0[i]) * HV_2D(active_set, ref[1:]).
                
                active_set = []
                # We need to update HV_2D efficiently or just recompute it since N is small.
                # N is at most 150. K is at most 150.
                # Recomputing HV_2D for each step is O(N^2). Total O(N^3). 150^3 = 3.3M. Acceptable.
                
                for i in range(len(points)):
                    # Add point i to active set
                    active_set.append(points[i])
                    
                    # Calculate HV_2D of active_set projected on f1, f2, bounded by ref[1], ref[2]
                    # Filter active_set for points where f1 < ref[1] and f2 < ref[2]
                    proj_pts = []
                    for p in active_set:
                        if p[1] < ref[1] and p[2] < ref[2]:
                            proj_pts.append([p[1], p[2]])
                    if proj_pts:
                        proj_arr = np.array(proj_pts)
                        area_2d = hv_2d(proj_arr, ref[1:])
                    else:
                        area_2d = 0.0
                        
                    # Determine width in f0
                    if i < len(points) - 1:
                        next_f0 = points[i+1][0]
                    else:
                        next_f0 = ref[0]
                    
                    width = next_f0 - points[i][0]
                    if width > 0:
                        total_vol += width * area_2d
                        
                return total_vol
            
            # Calculate HV for the full non-dominated set
            hv_full = calc_hypervolume_3d(nondom_f, ref_point)
            
            # Calculate HV for each set excluding one point
            min_hv_drop = np.inf
            target_idx_in_nondom = -1
            
            for i in range(K):
                # Create set without i
                mask = np.ones(K, dtype=bool)
                mask[i] = False
                reduced_set = nondom_f[mask]
                hv_reduced = calc_hypervolume_3d(reduced_set, ref_point)
                drop = hv_full - hv_reduced
                if drop < min_hv_drop:
                    min_hv_drop = drop
                    target_idx_in_nondom = i
                    
            # Map back to original index
            if target_idx_in_nondom != -1:
                target_idx = nondom_indices[target_idx_in_nondom]
            else:
                target_idx = nondom_indices[0] if K > 0 else np.random.randint(n_points)
                
        # Now we have the point with the lowest marginal contribution.
        # We will perform local search around this point? 
        # No, the strategy says: "replacing the point with the lowest marginal hypervolume contribution via Gaussian perturbations".
        # This implies we generate a new point near this one, evaluate it, and if it's better (higher marginal contribution or just better in some sense), we keep it.
        # But we don't "replace" in the sense of removing the old one from the recorded set (we can't un-record).
        # We just use the budget to find a *new* point that is likely to have high marginal contribution.
        # So we should perturb the *best* points? Or perturb the *worst* points to see if we can find something better nearby?
        # Actually, if a point has low marginal contribution, it's likely redundant.
        # Perturbing it might lead to a point with higher contribution if we move it to a less crowded area.
        # Or, we should perturb the *front* to find new non-dominated points.
        
        # Let's reinterpret: "switch to a hypervolume-contribution-based local search ... to refine the front".
        # This usually means: 
        # 1. Identify a point on the front.
        # 2. Perturb it.
        # 3. If the new point is non-dominated and has good contribution, keep it.
        
        # Which point to perturb?
        # If we perturb the point with the LOWEST contribution, we are trying to "fix" a redundant point.
        # If we perturb a point with HIGH contribution, we are trying to extend the front.
        
        # Let's try perturbing the point with the LOWEST contribution, hoping to find a neighbor that is less redundant or extends the front.
        # Actually, a better strategy is to perturb points that are on the "edges" of the front to extend it, or points in the "middle" to fill gaps.
        
        # Let's stick to the directive: "replacing the point with the lowest marginal hypervolume contribution".
        # This suggests we treat the set as a population and we want to improve the worst member.
        
        x_to_perturb = internal_x[target_idx]
        
        # Perform local search
        # We have `remaining_budget` evaluations left.
        # We will do a simple random walk / local search around x_to_perturb.
        # We will generate perturbations and evaluate them.
        # If a new point is non-dominated (with respect to the current internal set), we add it to our internal set.
        # We do NOT remove the old one, but we effectively "replace" its role in our mental model of the front.
        
        # To make it a "local search", we can use a decreasing step size.
        
        n_local_steps = remaining_budget
        # We can distribute these steps among a few starting points?
        # The directive says "replacing THE point". Singular.
        # So we focus on this one point.
        
        # But if we only perturb one point, we might miss other parts of the front.
        # However, Phase 1 covered the space. Phase 2 refines.
        
        # Let's use a simple Gaussian perturbation with adaptive step size.
        step_size = 0.1 * span # Initial step size
        best_f = internal_f[target_idx]
        best_x = x_to_perturb
        
        # We will try to find a point that dominates the current one or is non-dominated and has better contribution.
        
        for _ in range(n_local_steps):
            if time.time() > deadline:
                break
                
            # Generate perturbation
            # Use Gaussian noise
            noise = np.random.normal(0, step_size, self.dim)
            x_new = best_x + noise
            # Clip to bounds
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            
            # Check if f_new is non-dominated with respect to the current internal set
            # (excluding the point we are perturbing? Or including?)
            # If we include it, and f_new is worse than best_f in all objectives, it's dominated.
            # We want to find points that are BETTER or NON-DOMINATED.
            
            # Check dominance against all current internal points
            dominated = False
            for i in range(len(internal_f)):
                if i == target_idx:
                    # Compare against the point we are improving
                    # If new point is worse than old, it's not an improvement
                    if np.all(f_new >= internal_f[i]) and np.any(f_new > internal_f[i]):
                        dominated = True
                        break
                else:
                    if np.all(internal_f[i] <= f_new) and np.any(internal_f[i] < f_new):
                        dominated = True
                        break
            
            if not dominated:
                # It's non-dominated. Add it to our internal set.
                # We don't remove the old one, but we update our "best" for this local search.
                # Should we update best_x/best_f?
                # If f_new is better than best_f in some objectives, yes.
                # Let's just add it to the set.
                internal_x.append(x_new)
                internal_f.append(f_new)
                
                # Update best for this local search trajectory?
                # Let's keep the best non-dominated point found so far in this phase as the anchor.
                # If f_new dominates best_f, update best.
                if np.all(f_new <= best_f) and np.any(f_new < best_f):
                    best_x = x_new
                    best_f = f_new
                    # Decrease step size?
                    step_size *= 0.9
                else:
                    # It's non-dominated but doesn't dominate best_f.
                    # It's a different part of the front.
                    # We can keep it as an anchor for future perturbations?
                    # For simplicity, let's just continue perturbing from the original target_idx point
                    # or the most recent non-dominated point.
                    # Let's perturb from the most recent non-dominated point to explore locally.
                    best_x = x_new
                    best_f = f_new
                    step_size *= 0.95 # Slow decrease
            
            else:
                # Dominated. Increase step size to escape?
                step_size *= 1.1
                # Cap step size
                step_size = min(step_size, 0.5 * span)
                
        return []

# EVOLVE-BLOCK-END
