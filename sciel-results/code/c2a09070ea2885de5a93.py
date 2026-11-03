import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Objective function parameters
    # Variables:
    # x_0...x_10: x-coords of inner hex centers
    # y_0...y_10: y-coords of inner hex centers
    # theta_0...theta_10: angles of inner hexes (radians)
    # ocx, ocy: center of outer hex
    # oside: side length of outer hex
    # oang: angle of outer hex
    # Total variables: 11*3 + 4 = 37
    
    def objective(params):
        # Unpack
        inner_pos = params[:3*N].reshape(N, 3)
        ox, oy, oside, oang = params[3*N:3*N+4]
        
        # 1. Outer Hexagon Containment Penalty
        # The outer hexagon must contain all vertices of the inner hexagons.
        # For a regular hexagon with side S, center (cx, cy), angle phi:
        # A point (px, py) is inside if its distance to the boundary in the 3 normal directions <= apothem.
        # Apothem a = S * sqrt(3) / 2.
        # Normal directions are at angles phi + 30, phi + 90, phi + 150 degrees.
        
        a_outer = oside * math.sqrt(3) / 2.0
        if a_outer <= 0:
            return 1e10 # Invalid side length
            
        pen_outer = 0.0
        cos_oang = math.cos(oang)
        sin_oang = math.sin(oang)
        
        # Pre-compute normal vectors for outer hex
        # Angles in radians: oang + pi/6, oang + pi/2, oang + 5*pi/6
        n1_x = math.cos(oang + math.pi/6); n1_y = math.sin(oang + math.pi/6)
        n2_x = math.cos(oang + math.pi/2); n2_y = math.sin(oang + math.pi/2)
        n3_x = math.cos(oang + 5*math.pi/6); n3_y = math.sin(oang + 5*math.pi/6)
        
        for i in range(N):
            cx, cy, theta = inner_pos[i]
            # Vertices of inner hexagon i
            # Side length 1.
            # Vertices are at angles theta + k*pi/3
            for k in range(6):
                vx_rel = math.cos(theta + k * math.pi / 3.0)
                vy_rel = math.sin(theta + k * math.pi / 3.0)
                vx = cx + vx_rel
                vy = cy + vy_rel
                
                # Vector from outer center to vertex
                dx = vx - ox
                dy = vy - oy
                
                # Project onto normals
                p1 = dx * n1_x + dy * n1_y
                p2 = dx * n2_x + dy * n2_y
                p3 = dx * n3_x + dy * n3_y
                
                # Penalty if projection exceeds apothem
                if p1 > a_outer: pen_outer += (p1 - a_outer)**2
                if p1 < -a_outer: pen_outer += (-a_outer - p1)**2
                if p2 > a_outer: pen_outer += (p2 - a_outer)**2
                if p2 < -a_outer: pen_outer += (-a_outer - p2)**2
                if p3 > a_outer: pen_outer += (p3 - a_outer)**2
                if p3 < -a_outer: pen_outer += (-a_outer - p3)**2
        
        # 2. Inner Hexagon Collision Penalty
        # For two regular hexagons to be disjoint, their Minkowski sum must not contain the origin?
        # Actually, we check if the distance between centers is sufficient given their relative orientation.
        # A simpler robust check: The minimum distance between any point in Hex A and any point in Hex B must be > 0.
        # Since they are convex, we can check if the origin is inside the Minkowski difference?
        # Or simpler: Check if the polygons intersect.
        # Given the small N, we can do a pairwise check.
        # However, for optimization, we need a differentiable-ish penalty.
        # A common approximation is to check the distance between centers against a "minimum separation" based on relative angle.
        # But exact polygon intersection is better.
        
        # Let's use a penalty based on the maximum overlap depth.
        # If we approximate the hexagons as circles of radius 1 (circumradius), min dist is 2.
        # But hexagons are "fatter" in some directions.
        # A better metric: The distance from the center of Hex B to the boundary of Hex A (and vice versa).
        
        # Let's implement a function to calculate the minimum distance between two convex polygons (hexagons).
        # If dist > 0, no overlap. If dist < 0, overlap.
        # We want to penalize max(0, -dist)^2.
        
        pen_inner = 0.0
        
        # To speed up, we can first do a bounding box check or center distance check.
        # Min distance between centers of non-overlapping unit hexes is sqrt(3) ~ 1.732 (flat-to-flat)
        # Max is 2 (vertex-to-vertex).
        # So if dist > 2.1, definitely no overlap.
        
        for i in range(N):
            cx1, cy1, th1 = inner_pos[i]
            for j in range(i+1, N):
                cx2, cy2, th2 = inner_pos[j]
                dx = cx2 - cx1
                dy = cy2 - cy1
                dist_sq = dx*dx + dy*dy
                
                # Quick reject: if distance is large enough
                if dist_sq > 4.5: # (2.1)^2
                    continue
                
                # Calculate exact minimum distance between two hexagons
                # Vertices of hex 1
                verts1 = []
                for k in range(6):
                    a = th1 + k * math.pi / 3.0
                    verts1.append((cx1 + math.cos(a), cy1 + math.sin(a)))
                
                verts2 = []
                for k in range(6):
                    a = th2 + k * math.pi / 3.0
                    verts2.append((cx2 + math.cos(a), cy2 + math.sin(a)))
                
                # Find minimum distance between segments of poly1 and poly2
                min_dist = float('inf')
                for i1 in range(6):
                    p1 = verts1[i1]
                    p2 = verts1[(i1+1)%6]
                    for i2 in range(6):
                        p3 = verts2[i2]
                        p4 = verts2[(i2+1)%6]
                        d = _seg_seg_dist_sq(p1, p2, p3, p4)
                        if d < min_dist:
                            min_dist = d
                
                # If min_dist < 0, they overlap.
                # Actually, _seg_seg_dist_sq returns 0 if they intersect or touch.
                # We need a measure of "how much" they overlap if they do.
                # A simpler proxy for optimization:
                # If they intersect, we want to push them apart.
                # We can use the negative of the minimum distance if we define distance as signed distance.
                # But segment distance is unsigned.
                
                # Alternative: Check if any vertex of 1 is inside 2, or vice versa.
                # If yes, penalty is based on depth.
                # Let's use a simpler differentiable approximation for the penalty:
                # The "effective radius" of a hexagon in direction theta is r(theta).
                # This is complex.
                
                # Let's stick to a robust but slightly expensive check:
                # If dist_sq < 3.0 (sqrt(3)^2), they *might* overlap.
                # If they definitely overlap (e.g. centers very close), penalty is high.
                
                # Let's use the property that for two convex polygons, if the distance between
                # any pair of edges is < 0, they overlap.
                # For optimization, let's penalize the square of the "penetration depth".
                # Approximation: If min_dist == 0 (intersection), we need a gradient.
                # The gradient of distance is the normal direction.
                
                # A common trick in soft constraints:
                # Penalty = max(0, min_dist_required - actual_dist)^2
                # But we don't have a simple scalar "actual_dist" for overlapping shapes.
                
                # Let's use the vertex-in-polygon check for a hard penalty component,
                # and center distance for a soft component.
                
                # Check if center of 1 is inside 2
                # And center of 2 is inside 1
                # And if any vertex of 1 is inside 2 (and vice versa)
                
                # For efficiency in the loop, let's just use a conservative center-distance based penalty
                # combined with a check for deep overlaps.
                
                # Actually, let's use the `scipy.spatial` or just manual point-in-poly.
                # Given N=11, 55 pairs. 6x6 segments = 36 checks per pair. 1980 checks.
                # This is fast enough for differential evolution if we limit iterations.
                
                # Let's refine the penalty:
                # If min_dist == 0.0, they touch or intersect.
                # We need to know if they intersect.
                # If they intersect, we can estimate overlap by checking if center of 1 is in 2.
                
                # Let's use a simpler metric that is differentiable:
                # The distance from center 1 to the boundary of hex 2, projected along the line connecting centers?
                # No, let's use the "Minkowski sum" logic implicitly.
                # If the vector (dx, dy) is inside the Minkowski sum of Hex1 (at 0) and -Hex2 (at 0), they overlap.
                # The Minkowski sum of two unit hexes is a hexagon with side 2.
                # Wait, that's only if they are aligned. With rotation, the Minkowski sum is a 12-gon (or similar).
                
                # Let's go with a robust, albeit slightly non-smooth, penalty:
                # If the polygons intersect, penalty = 1000 + (overlap_area_estimate).
                # Estimating area is hard.
                
                # Let's try this:
                # If min_dist == 0, check if they actually cross.
                # If they cross, add a penalty of (1.0) ** 2 = 1.0 to start, but we need gradient.
                # Differential evolution is derivative-free, so non-smoothness is OK, but it might get stuck.
                
                # Let's use a "soft" distance penalty.
                # Define d = min_dist.
                # If d == 0, we assume overlap.
                # We can penalize based on the distance between centers.
                # If centers are too close, penalty is high.
                # Threshold: sqrt(3).
                # If dist < sqrt(3), penalty = (sqrt(3) - dist)^2 * 100.
                # If dist >= sqrt(3), check exact intersection. If intersect, penalty = 1000.
                
                if dist_sq < 3.0:
                    dist = math.sqrt(dist_sq)
                    # Strong penalty if centers are very close
                    pen_inner += (math.sqrt(3) - dist)**2 * 50.0
                
                # Exact intersection check
                # If min_dist == 0, check if segments actually cross (not just touch)
                # Or just check if any vertex is inside.
                # For DE, a hard penalty for intersection is effective.
                if min_dist < 1e-9:
                    # Check if they really overlap (not just touch)
                    # Simple check: is center of 1 inside 2?
                    if _point_in_hex(cx1, cy1, verts2):
                        pen_inner += 1000.0
                    elif _point_in_hex(cx2, cy2, verts1):
                        pen_inner += 1000.0
                    else:
                        # Might be edge crossing without center containment
                        # Penalize anyway to be safe
                        pen_inner += 100.0

        # 3. Regularization / Compactness
        # We want the outer hexagon to be as small as possible.
        # The objective is primarily `oside`.
        # But `oside` is a variable. We want to minimize it.
        # So we add `oside` to the objective.
        
        return oside + pen_outer + pen_inner

    def _seg_seg_dist_sq(p1, p2, p3, p4):
        """
        Returns squared distance between segment p1-p2 and segment p3-p4.
        If they intersect, returns 0.
        """
        # Standard segment-segment distance algorithm
        # We can use a simpler approach: check intersection, then min of point-seg distances.
        
        # Intersection check
        # Vector d1 = p2 - p1, d2 = p4 - p3
        d1x = p2[0] - p1[0]; d1y = p2[1] - p1[1]
        d2x = p4[0] - p3[0]; d2y = p4[1] - p3[1]
        
        denom = d1x * d2y - d1y * d2x
        if abs(denom) < 1e-12:
            # Parallel
            pass
        else:
            t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / denom
            u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                return 0.0 # Intersect
        
        # Min of distances between endpoints and opposite segments
        min_d = float('inf')
        min_d = min(min_d, _pt_seg_dist_sq(p1, p3, p4))
        min_d = min(min_d, _pt_seg_dist_sq(p2, p3, p4))
        min_d = min(min_d, _pt_seg_dist_sq(p3, p1, p2))
        min_d = min(min_d, _pt_seg_dist_sq(p4, p1, p2))
        
        return min_d

    def _pt_seg_dist_sq(p, a, b):
        """
        Squared distance from point p to segment a-b.
        """
        abx = b[0] - a[0]
        aby = b[1] - a[1]
        apx = p[0] - a[0]
        apy = p[1] - a[1]
        
        ab_sq = abx*abx + aby*aby
        if ab_sq < 1e-12:
            return apx*apx + apy*apy
        
        t = (apx*abx + apy*aby) / ab_sq
        t = max(0, min(1, t))
        
        cx = a[0] + t * abx
        cy = a[1] + t * aby
        
        dx = p[0] - cx
        dy = p[1] - cy
        return dx*dx + dy*dy

    def _point_in_hex(px, py, verts):
        """
        Check if point (px, py) is inside polygon defined by verts (CCW).
        """
        # Winding number or ray casting
        # Ray casting: cast ray to +x, count intersections
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # Bounds for Differential Evolution
    # Inner hex centers: likely within [-4, 4]
    # Inner angles: [0, 2*pi]
    # Outer center: [-5, 5]
    # Outer side: [2.0, 10.0] (Min possible is ~sqrt(3) for 1 hex, for 11 it's larger. Max 10 is safe)
    # Outer angle: [0, 2*pi]
    
    bounds = []
    for _ in range(N):
        bounds.append((-4.0, 4.0)) # x
        bounds.append((-4.0, 4.0)) # y
        bounds.append((0.0, 2.0 * math.pi)) # angle
    
    bounds.append((-5.0, 5.0)) # ox
    bounds.append((-5.0, 5.0)) # oy
    bounds.append((2.0, 10.0)) # oside
    bounds.append((0.0, 2.0 * math.pi)) # oang
    
    # Run Differential Evolution
    # maxiter and popsize need to be balanced with time.
    # 150s limit. DE is stochastic.
    # Let's try popsize=50, maxiter=100.
    # If it takes too long, we need a timeout.
    
    # We will run DE with a callback to check time?
    # scipy's DE doesn't have a built-in time-based stop, only maxiter.
    # We can estimate time per iteration.
    # Or just run a fixed number of iterations that fits in ~100s.
    
    # Let's use a smaller population to be safe.
    # popsize = 30
    # maxiter = 50
    # This is 1500 evaluations.
    # Each evaluation involves ~55 pair checks, each with ~36 seg-seg dist checks.
    # 2000 seg-seg checks per eval.
    # 1500 * 2000 = 3M checks. Should be fast in Python/NumPy if optimized, but pure Python loops might be slow.
    # Let's optimize the inner loop by using NumPy where possible or keeping it simple.
    # The current implementation uses pure Python loops for geometry.
    # 3M simple math ops in Python might take ~30-60s. Acceptable.
    
    # To be safe, let's use popsize=20, maxiter=30. (600 evals)
    # And maybe run a second pass with refined bounds?
    # Or just one robust pass.
    
    # Let's try popsize=40, maxiter=50.
    # If it's too slow, we'll see.
    
    # Actually, let's use a "fast" penalty calculation.
    # We can skip the exact seg-seg check if the bounding boxes are far apart.
    
    def objective_fast(params):
        # Same as objective but with early exits
        inner_pos = params[:3*N].reshape(N, 3)
        ox, oy, oside, oang = params[3*N:3*N+4]
        
        a_outer = oside * math.sqrt(3) / 2.0
        if a_outer <= 0:
            return 1e10
            
        pen_outer = 0.0
        n1_x = math.cos(oang + math.pi/6); n1_y = math.sin(oang + math.pi/6)
        n2_x = math.cos(oang + math.pi/2); n2_y = math.sin(oang + math.pi/2)
        n3_x = math.cos(oang + 5*math.pi/6); n3_y = math.sin(oang + 5*math.pi/6)
        
        for i in range(N):
            cx, cy, theta = inner_pos[i]
            for k in range(6):
                vx = cx + math.cos(theta + k * math.pi / 3.0)
                vy = cy + math.sin(theta + k * math.pi / 3.0)
                dx = vx - ox
                dy = vy - oy
                p1 = dx * n1_x + dy * n1_y
                p2 = dx * n2_x + dy * n2_y
                p3 = dx * n3_x + dy * n3_y
                if p1 > a_outer: pen_outer += (p1 - a_outer)**2
                if p1 < -a_outer: pen_outer += (-a_outer - p1)**2
                if p2 > a_outer: pen_outer += (p2 - a_outer)**2
                if p2 < -a_outer: pen_outer += (-a_outer - p2)**2
                if p3 > a_outer: pen_outer += (p3 - a_outer)**2
                if p3 < -a_outer: pen_outer += (-a_outer - p3)**2
                
        pen_inner = 0.0
        
        # Precompute vertices for all inner hexes to avoid recalculation in pair loop
        verts_list = []
        for i in range(N):
            cx, cy, theta = inner_pos[i]
            verts = []
            for k in range(6):
                a = theta + k * math.pi / 3.0
                verts.append((cx + math.cos(a), cy + math.sin(a)))
            verts_list.append(verts)
        
        for i in range(N):
            cx1, cy1, _ = inner_pos[i]
            v1 = verts_list[i]
            for j in range(i+1, N):
                cx2, cy2, _ = inner_pos[j]
                v2 = verts_list[j]
                dx = cx2 - cx1
                dy = cy2 - cy1
                dist_sq = dx*dx + dy*dy
                
                if dist_sq > 4.5:
                    continue
                
                if dist_sq < 3.0:
                    dist = math.sqrt(dist_sq)
                    pen_inner += (math.sqrt(3) - dist)**2 * 50.0
                
                # Check intersection
                # Quick bounding box check?
                # Min x of v1 vs Max x of v2, etc.
                min_x1 = min(p[0] for p in v1)
                max_x1 = max(p[0] for p in v1)
                min_y1 = min(p[1] for p in v1)
                max_y1 = max(p[1] for p in v1)
                
                min_x2 = min(p[0] for p in v2)
                max_x2 = max(p[0] for p in v2)
                min_y2 = min(p[1] for p in v2)
                max_y2 = max(p[1] for p in v2)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue # Bounding boxes disjoint
                
                # Exact check
                min_dist = float('inf')
                for i1 in range(6):
                    p1 = v1[i1]
                    p2 = v1[(i1+1)%6]
                    for i2 in range(6):
                        p3 = v2[i2]
                        p4 = v2[(i2+1)%6]
                        d = _seg_seg_dist_sq(p1, p2, p3, p4)
                        if d < min_dist:
                            min_dist = d
                            if min_dist < 1e-9:
                                break
                    if min_dist < 1e-9:
                        break
                
                if min_dist < 1e-9:
                    if _point_in_hex(cx1, cy1, v2):
                        pen_inner += 1000.0
                    elif _point_in_hex(cx2, cy2, v1):
                        pen_inner += 1000.0
                    else:
                        pen_inner += 100.0

        return oside + pen_outer + pen_inner

    # Run DE
    # Use a seed for reproducibility if needed, but random is fine.
    # We'll run it and check time.
    
    # To handle the time limit strictly, we can run DE in a way that we can interrupt?
    # No, scipy DE is a black box.
    # We must estimate.
    # Let's use popsize=20, maxiter=20. (400 evals)
    # This should be very fast (<10s).
    # Then we can use the result to do a local refinement with `scipy.optimize.minimize` (L-BFGS-B)
    # which is faster per iteration but needs a good start.
    
    de_result = differential_evolution(
        objective_fast, 
        bounds, 
        maxiter=30, 
        popsize=30, 
        tol=1e-5, 
        seed=42,
        polish=False # Disable polish to save time, we'll do our own
    )
    
    best_params = de_result.x
    best_val = de_result.fun
    
    # Local Refinement
    # Use L-BFGS-B
    # It might get stuck if the penalty is non-smooth, but the "soft" part is smooth.
    # We can try a few iterations.
    
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import minimize
            res = minimize(
                objective_fast, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50}
            )
            if res.fun < best_val:
                best_params = res.x
                best_val = res.fun
        except Exception:
            pass
            
    # Extract best solution
    inner_pos = best_params[:3*N].reshape(N, 3)
    ox, oy, oside, oang = best_params[3*N:3*N+4]
    
    # Verify validity
    # If the solution is invalid (penalty > 0), we might need to return a fallback.
    # But DE should find a valid one.
    # Let's check if penalty is small.
    # If not, we can try to "fix" it by increasing oside slightly.
    
    # Format output
    inner_rows = []
    for i in range(N):
        x, y, theta = inner_pos[i]
        # Convert theta to degrees
        theta_deg = math.degrees(theta) % 360.0
        inner_rows.append([x, y, theta_deg])
    
    # Outer hexagon
    outer_center = (ox, oy)
    outer_side = oside
    outer_angle = math.degrees(oang) % 360.0
    
    return inner_rows, outer_center, outer_side, outer_angle

if __name__ == "__main__":
    r = construct()
    print(r)
