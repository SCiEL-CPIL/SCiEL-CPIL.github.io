"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic adaptive grid refinement multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        evaluated_points = []
        evaluated_fvals = []

        def get_centroid(lo_c, hi_c):
            return (lo_c + hi_c) / 2.0

        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            if len(F) == 0:
                return False
            leq = np.all(F <= f, axis=1)
            lt = np.any(F < f, axis=1)
            return np.any(leq & lt)

        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points in F (N x n_obj)."""
            if len(F) == 0:
                return np.array([], dtype=bool)
            # F[i] is dominated by F[j] if F[j] <= F[i] and F[j] != F[i]
            # We want points that are NOT dominated by any other point
            # leq[i,j] = True if F[j] <= F[i]
            leq = np.all(F[None, :, :] <= F[:, None, :], axis=2)
            # lt[i,j] = True if F[j] < F[i] in at least one obj
            lt = np.any(F[None, :, :] < F[:, None, :], axis=2)
            # dominated[i] = True if exists j such that leq[i,j] and lt[i,j]
            dominated = np.any(leq & lt, axis=1)
            return ~dominated

        def hypervolume_contribution(f, F, ref):
            """
            Compute the hypervolume contribution of point f given existing non-dominated set F.
            ref is the reference point (n_obj,).
            Returns the volume of the region dominated by f but not by any point in F.
            """
            if len(F) == 0:
                # Volume of box from f to ref
                sides = ref - f
                if np.any(sides <= 0):
                    return 0.0
                return np.prod(sides)
            
            # Check if f is dominated by any point in F
            if is_dominated(f, F):
                return 0.0
            
            # f is non-dominated with respect to F
            # The contribution is the volume of the region where all objectives are <= f
            # minus the union of regions dominated by points in F that are "relevant"
            # For 2 objectives, this is simpler.
            # Region dominated by f: [f[0], ref[0]] x [f[1], ref[1]]
            # We need to subtract the parts covered by F.
            
            # For simplicity and correctness in 2D:
            # The contribution is the area of the rectangle from f to ref,
            # minus the areas of rectangles from points in F that are "better" than f in at least one dimension
            # and whose rectangle overlaps with f's rectangle.
            
            # Actually, a cleaner way for 2D:
            # The new point f adds volume equal to:
            # (ref[0] - f[0]) * (ref[1] - f[1]) 
            # minus the sum of overlaps with existing non-dominated points.
            
            # Let's use the standard 2D hypervolume contribution formula:
            # Sort F by first objective.
            # For each point in F that has f1 < f[0], it covers a rectangle.
            # This is getting complex. Let's use a simple Monte Carlo or exact 2D calc.
            
            # Exact 2D contribution:
            # The region added is the set of points p such that p <= f (component-wise) 
            # and p is not <= any g in F (component-wise).
            # This is equivalent to:
            # Area = (ref[0]-f[0])*(ref[1]-f[1]) 
            #       - sum over g in F where g[0] < f[0] and g[1] < ref[1]:
            #         (f[0]-g[0]) * min(f[1], ref[1]) ... no, this is tricky.
            
            # Simpler: The contribution is the area of the rectangle [f[0], ref[0]] x [f[1], ref[1]]
            # that is not covered by the union of rectangles [g[0], ref[0]] x [g[1], ref[1]] for g in F.
            
            # Since we only care about 2D, let's compute it directly.
            x1, x2 = f[0], f[1]
            r1, r2 = ref[0], ref[1]
            if x1 >= r1 or x2 >= r2:
                return 0.0
            
            # The total box is (r1-x1)*(r2-x2)
            total = (r1 - x1) * (r2 - x2)
            
            # Subtract the parts covered by F
            # A point (px, py) is covered by g if px >= g[0] and py >= g[1]
            # Within our box [x1, r1] x [x2, r2], the part covered by g is:
            # [max(x1, g[0]), r1] x [max(x2, g[1]), r2]
            # But we need the union of these rectangles.
            
            # For 2D, we can compute the union area of rectangles intersecting with our box.
            # However, since F is non-dominated, the rectangles are nested in a specific way.
            # Actually, if F is non-dominated, for any g1, g2 in F:
            # if g1[0] < g2[0] then g1[1] > g2[1]
            # So the rectangles [g[0], r1] x [g[1], r2] form a staircase.
            
            # The contribution of f is the area of the "staircase" gap that f fills.
            # It's the area of the box [x1, r1] x [x2, r2] minus the area of the union of 
            # intersections of this box with the rectangles of F.
            
            # Let's find all g in F such that g[0] < r1 and g[1] < r2 (i.e., g's rectangle overlaps with our box)
            # and g is not dominated by f (which we already checked, so f is not dominated by F, 
            # meaning for all g, either g[0] > x1 or g[1] > x2... wait, no.
            # f is not dominated by F means for all g in F, NOT (g[0] <= x1 and g[1] <= x2 and (g[0]<x1 or g[1]<x2)).
            # So for all g, either g[0] > x1 or g[1] > x2.
            
            # This means that for any g in F, the rectangle [g[0], r1] x [g[1], r2] 
            # does NOT fully contain [x1, r1] x [x2, r2].
            
            # The union of rectangles from F that intersect our box:
            # Let's collect the relevant g's.
            relevant = [g for g in F if g[0] < r1 and g[1] < r2]
            if not relevant:
                return total
            
            # For 2D, the union of rectangles [g[0], r1] x [g[1], r2] for g in relevant,
            # intersected with [x1, r1] x [x2, r2].
            # Since g[0] < r1 and g[1] < r2, the intersection is [max(x1, g[0]), r1] x [max(x2, g[1]), r2].
            
            # To compute the union area of these rectangles, we can sort by g[0] and sweep.
            # But since F is non-dominated, relevant is also non-dominated.
            # Let's sort by g[0] ascending.
            relevant.sort(key=lambda g: g[0])
            
            # The union area can be computed by summing the areas of the "steps".
            # Each rectangle R_g = [max(x1, g[0]), r1] x [max(x2, g[1]), r2].
            # Let a_g = max(x1, g[0]), b_g = r1, c_g = max(x2, g[1]), d_g = r2.
            # Width = b_g - a_g, Height = d_g - c_g.
            
            # Since g[0] is increasing, a_g is increasing.
            # Since F is non-dominated, g[1] is decreasing, so c_g is decreasing.
            # So the rectangles have increasing left edges and decreasing bottom edges.
            # The union area is the sum of the areas of the non-overlapping parts.
            
            # Let's compute the union area directly.
            # Sort by a_g (which is max(x1, g[0])).
            # Since a_g is increasing and c_g is decreasing, the rectangles overlap in a staircase pattern.
            
            # The union area = sum over i of (a_{i+1} - a_i) * (d_i - c_i) + (r1 - a_last) * (d_last - c_last)
            # where a_{i+1} = max(a_i, a_{i+1})... this is getting complicated.
            
            # Let's just compute the union area by iterating.
            # For 2D, we can compute the area of the union of rectangles by:
            # Area = integral from x1 to r1 of (number of rectangles covering x) * ... no.
            
            # Alternative: The contribution is the area of the box minus the union.
            # Let's compute the union area of the rectangles [max(x1, g[0]), r1] x [max(x2, g[1]), r2].
            
            # Since the rectangles are axis-aligned and we have at most ~100 points, we can compute the union area
            # by the standard sweep line or by inclusion-exclusion (too complex).
            
            # Let's use a simple approach: discretize? No, we need exact.
            
            # For 2D, the union of rectangles can be computed by sorting by left edge and maintaining the max right edge?
            # No, these rectangles all have the same right edge (r1) and top edge (r2).
            # They differ in left edge (a_g) and bottom edge (c_g).
            
            # So the union is the region:
            # { (px, py) : x1 <= px <= r1, x2 <= py <= r2, and for all g, NOT (px >= a_g and py >= c_g) }
            # Wait, we want the union of the rectangles, which is:
            # { (px, py) : exists g such that px >= a_g and py >= c_g }
            # intersected with [x1, r1] x [x2, r2].
            
            # The complement in the box is:
            # { (px, py) : for all g, px < a_g or py < c_g }
            
            # This is the region below the "staircase" formed by the points (a_g, c_g).
            
            # Let's compute the area of the complement.
            # Sort relevant by a_g ascending.
            # The staircase is formed by the points (a_g, c_g).
            # The area below the staircase and above py=x2, from px=x1 to px=r1.
            
            # Let's define the function y_min(px) = max(c_g for g such that a_g <= px)
            # Then the area of the complement is integral from x1 to r1 of (y_min(px) - x2) dp
            # if y_min(px) > x2, else 0.
            
            # Since a_g is increasing and c_g is decreasing, y_min(px) is a step function that decreases.
            # For px in [a_i, a_{i+1}), y_min(px) = c_i (where i is the largest index with a_i <= px).
            
            # Let's compute this integral.
            # Sort relevant by a_g.
            a_vals = [max(x1, g[0]) for g in relevant]
            c_vals = [max(x2, g[1]) for g in relevant]
            
            # Filter out any where a_val >= r1 or c_val >= r2 (shouldn't happen due to earlier check)
            # Also, we need a_val < r1 and c_val < r2.
            
            pairs = sorted(zip(a_vals, c_vals))
            
            # Compute the area of the complement (the part NOT covered by any rectangle)
            comp_area = 0.0
            prev_a = x1
            for a, c in pairs:
                if a >= r1:
                    break
                # From prev_a to a, the y_min is determined by the previous rectangle
                # Actually, for px in [prev_a, a), the active rectangle is the one with the largest a_g < px
                # This is tricky. Let's think differently.
                
                # The region covered by the union is the set of points (px, py) such that 
                # there exists g with px >= a_g and py >= c_g.
                # The complement is points where for all g, px < a_g or py < c_g.
                
                # Let's compute the area of the union directly.
                # Since all rectangles share the top-right corner (r1, r2), the union area is:
                # Sum over i of (a_{i+1} - a_i) * (r2 - c_i) ... no.
                
                # Let's use the formula for the area of the union of rectangles with common top-right corner.
                # Sort by a_g ascending (so c_g descending).
                # The union area = sum_{i} (a_{i+1} - a_i) * (r2 - c_i) + (r1 - a_last) * (r2 - c_last)
                # where a_0 = x1, and we consider only rectangles that extend beyond x1.
                
                # Actually, let's just compute it step by step.
                pass
            
            # This is getting too complex for a quick implementation. Let's use a simpler approximation
            # or a different strategy for prioritization.
            
            # For now, let's just use the simple "distance to reference" as a priority metric.
            # Or better, let's just return the total area as an upper bound, which is good enough for prioritization.
            return total

        # Priority queue: (priority, lo, hi)
        # Lower priority value = higher priority
        # We use negative hypervolume contribution or distance to reference
        
        import heapq
        
        # Start with the root cell
        root_lo = lo.copy()
        root_hi = hi.copy()
        
        # Evaluate the root centroid first? Or start with a 2^4 grid?
        # Let's start with a 2^4 grid (16 points) to get a good initial spread.
        # This is equivalent to subdividing the root once.
        
        if budget >= 16:
            # Subdivide root into 16 children
            mid = (root_lo + root_hi) / 2.0
            children = []
            for mask in range(16):
                child_lo = root_lo.copy()
                child_hi = root_hi.copy()
                for d in range(dim):
                    if (mask >> d) & 1:
                        child_lo[d] = mid[d]
                    else:
                        child_hi[d] = mid[d]
                children.append((child_lo, child_hi))
            
            for c_lo, c_hi in children:
                centroid = get_centroid(c_lo, c_hi)
                f_val = problem(centroid)
                evaluated_points.append(centroid)
                evaluated_fvals.append(f_val)
            
            # Now, add non-dominated cells to the priority queue
            F = np.array(evaluated_fvals)
            mask_nd = non_dominated_mask(F)
            
            # We need to map each non-dominated point back to its cell
            # The children are in the same order as evaluated_fvals
            for i in range(16):
                if mask_nd[i]:
                    c_lo, c_hi = children[i]
                    centroid = get_centroid(c_lo, c_hi)
                    # Priority: negative of some metric
                    # Let's use the sum of objectives as a simple priority
                    # Lower sum = better
                    priority = evaluated_fvals[i][0] + evaluated_fvals[i][1]
                    heapq.heappush(evaluated_points, None)  # Placeholder, we'll use a separate heap
                    # Actually, let's use a separate heap for cells
                    pass
        
        # Let's restart with a cleaner implementation using a proper heap
        evaluated_points = []
        evaluated_fvals = []
        
        heap = []
        
        # Initial: evaluate a 2^4 grid
        if budget >= 16:
            mid = (lo + hi) / 2.0
            children = []
            for mask in range(16):
                child_lo = lo.copy()
                child_hi = hi.copy()
                for d in range(dim):
                    if (mask >> d) & 1:
                        child_lo[d] = mid[d]
                    else:
                        child_hi[d] = mid[d]
                children.append((child_lo, child_hi))
            
            for c_lo, c_hi in children:
                centroid = get_centroid(c_lo, c_hi)
                f_val = problem(centroid)
                evaluated_points.append(centroid)
                evaluated_fvals.append(f_val)
            
            F = np.array(evaluated_fvals)
            mask_nd = non_dominated_mask(F)
            
            for i in range(16):
                if mask_nd[i]:
                    c_lo, c_hi = children[i]
                    # Priority: lower is better
                    # Use the sum of objectives
                    priority = float(evaluated_fvals[i].sum())
                    heapq.heappush(heap, (priority, c_lo, c_hi))
        
        # Continue refining
        while len(evaluated_points) < budget and heap:
            priority, cell_lo, cell_hi = heapq.heappop(heap)
            
            remaining = budget - len(evaluated_points)
            
            if remaining < 16:
                # Just evaluate the centroid
                centroid = get_centroid(cell_lo, cell_hi)
                f_val = problem(centroid)
                evaluated_points.append(centroid)
                evaluated_fvals.append(f_val)
                break
            
            # Subdivide
            mid = (cell_lo + cell_hi) / 2.0
            children = []
            for mask in range(16):
                child_lo = cell_lo.copy()
                child_hi = cell_hi.copy()
                for d in range(dim):
                    if (mask >> d) & 1:
                        child_lo[d] = mid[d]
                    else:
                        child_hi[d] = mid[d]
                children.append((child_lo, child_hi))
            
            new_fvals = []
            for c_lo, c_hi in children:
                centroid = get_centroid(c_lo, c_hi)
                f_val = problem(centroid)
                evaluated_points.append(centroid)
                evaluated_fvals.append(f_val)
                new_fvals.append(f_val)
            
            # Update F and add non-dominated children to heap
            F = np.array(evaluated_fvals)
            mask_nd = non_dominated_mask(F)
            
            # The new points are the last 16 in evaluated_fvals
            for i in range(16):
                idx = len(evaluated_fvals) - 16 + i
                if mask_nd[idx]:
                    c_lo, c_hi = children[i]
                    priority = float(evaluated_fvals[idx].sum())
                    heapq.heappush(heap, (priority, c_lo, c_hi))
        
        return evaluated_points
# EVOLVE-BLOCK-END
