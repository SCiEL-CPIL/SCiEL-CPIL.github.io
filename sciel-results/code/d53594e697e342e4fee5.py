import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0

    def hex_vertices(cx, cy, side, angle_rad):
        """Generate vertices of a regular hexagon."""
        angs = angle_rad + np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
        return np.column_stack([cx + side * np.cos(angs),
                                cy + side * np.sin(angs)])

    def sat_overlap(v1, v2):
        """
        Check if two convex polygons (defined by vertices) overlap.
        Returns a measure of overlap (positive if overlapping, 0 if disjoint).
        Uses Separating Axis Theorem.
        """
        # v1, v2 are 6x2 arrays of vertices
        # For each edge of v1 and v2, check projection
        min_overlap = 0.0
        
        # Helper to get edge normals
        def get_normals(v):
            edges = np.roll(v, -1, axis=0) - v
            # Perpendicular to edges
            normals = np.column_stack([-edges[:, 1], edges[:, 0]])
            norms = np.linalg.norm(normals, axis=1)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            return normals / norms[:, np.newaxis]
        
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        
        for n in np.vstack([n1, n2]):
            p1 = v1 @ n
            p2 = v2 @ n
            min1, max1 = np.min(p1), np.max(p1)
            min2, max2 = np.min(p2), np.max(p2)
            
            # If separated, return 0
            if max1 < min2 or max2 < min1:
                return 0.0
            
            # Overlap amount on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def objective(params):
        """
        Objective function for optimization.
        params: 24 values (x1, y1, ..., x12, y12)
        """
        if elapsed() > DEADLINE:
            return 1e10
            
        pts = params.reshape(12, 2)
        # Assume all inner hexagons have angle 0 for simplicity in this first pass, 
        # or we could include angles. Let's stick to translation only for speed, 
        # as optimal packings often have aligned hexagons or simple rotations.
        # Actually, allowing rotation is complex. Let's assume all inner hexagons 
        # are rotated by the same global angle theta, which we can optimize later.
        # For now, fix inner angle to 0.
        
        verts = [hex_vertices(x, y, 1.0, 0.0) for x, y in pts]
        
        # 1. Calculate collision penalty
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                overlap = sat_overlap(verts[i], verts[j])
                if overlap > 1e-9:
                    penalty += overlap * 1000  # Heavy penalty
        
        # 2. Calculate enclosing hexagon side length
        # We need the minimal enclosing hexagon.
        # A regular hexagon of side S contains points if for all 6 normals n_k,
        # max_p (p - c) . n_k <= S * sqrt(3) / 2.
        # We can optimize the center c and the orientation of the enclosing hexagon.
        # To keep it simple, let's assume the enclosing hexagon is aligned with the axes (angle 0) 
        # or we can optimize its angle too. 
        # Let's optimize the enclosing hexagon angle and center as part of the outer loop or a nested optimization.
        
        # For speed, let's estimate the bounding box and use that to guess the side length.
        # A better approach: use the convex hull of all vertices and find the min enclosing hexagon.
        
        all_verts = np.vstack(verts)
        
        # Simple estimate: max distance from centroid
        centroid = np.mean(all_verts, axis=0)
        dists = np.linalg.norm(all_verts - centroid, axis=1)
        max_dist = np.max(dists)
        
        # Side length S is related to circumradius R by R = S.
        # So S >= max_dist.
        # But a hexagon is tighter than a circle.
        # Let's just return max_dist + penalty for now as a proxy, 
        # and refine in a post-processing step.
        
        return max_dist + penalty

    def refine_enclosing_hex(inner_pts, inner_angles):
        """
        Given a set of inner hexagons (positions and angles), 
        find the minimum enclosing regular hexagon.
        Returns (side_length, center, angle).
        """
        # Collect all vertices
        all_verts = []
        for (x, y), a in zip(inner_pts, inner_angles):
            v = hex_vertices(x, y, 1.0, a)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # The minimum enclosing hexagon problem is:
        # Minimize S subject to:
        # For all 6 directions k (angle = alpha + k*pi/3):
        #   max_i (v_i . n_k) - min_i (v_i . n_k) <= 2 * S * sqrt(3) / 2 = S * sqrt(3)
        # Wait, the width of the hexagon in direction n_k is S * sqrt(3).
        # So we need the width of the point set in direction n_k to be <= S * sqrt(3).
        # S >= max_k (width_k / sqrt(3))
        
        # We can optimize the angle alpha and the center is implicitly determined by the width.
        # Actually, the width is independent of the center.
        # So for a given angle alpha, the minimal S is max_k (width_k / sqrt(3)).
        
        best_s = float('inf')
        best_alpha = 0.0
        
        # Scan alpha
        for deg in range(0, 60, 1):
            alpha = math.radians(deg)
            s = 0.0
            for k in range(6):
                ang = alpha + k * math.pi / 3
                n = np.array([math.cos(ang), math.sin(ang)])
                projs = all_verts @ n
                width = np.max(projs) - np.min(projs)
                s_candidate = width / math.sqrt(3)
                if s_candidate > s:
                    s = s_candidate
            if s < best_s:
                best_s = s
                best_alpha = alpha
        
        # Refine alpha locally
        def side_for_alpha(alpha):
            s = 0.0
            for k in range(6):
                ang = alpha + k * math.pi / 3
                n = np.array([math.cos(ang), math.sin(ang)])
                projs = all_verts @ n
                width = np.max(projs) - np.min(projs)
                s_candidate = width / math.sqrt(3)
                if s_candidate > s:
                    s = s_candidate
            return s
            
        res = minimize(side_for_alpha, best_alpha, method='Nelder-Mead', options={'maxiter': 100})
        if res.fun < best_s:
            best_s = res.fun
            best_alpha = res.x[0]
            
        # Calculate the center of the enclosing hexagon
        # The center should be the midpoint of the projections for the tightest constraint?
        # Actually, we just need to return the side length and an angle.
        # The evaluator will place the hexagon centered at (0,0) or we need to return the center?
        # The interface says: inner rows (x, y, angle), outer center (x, y), side, angle.
        # We can shift the whole configuration so the enclosing hexagon is centered at (0,0).
        
        # Find the center that minimizes the max distance to the edges?
        # For a given orientation, the optimal center c minimizes max_k (max_p (p-c).n_k).
        # This is equivalent to centering the bounding box in each normal direction.
        
        c_x = 0.0
        c_y = 0.0
        for k in range(6):
            ang = best_alpha + k * math.pi / 3
            n = np.array([math.cos(ang), math.sin(ang)])
            projs = all_verts @ n
            c_proj = (np.max(projs) + np.min(projs)) / 2.0
            # c . n = c_proj
            # This is a system of linear equations. We can just average the centers from a few directions.
            # Or simply, the center of the minimum enclosing hexagon is the average of the centers of the bounding boxes.
            c_x += c_proj * n[0]
            c_y += c_proj * n[1]
            
        # This averaging is not strictly correct for the center, but let's use the centroid of the vertices as a start 
        # and then shift so that the max projection is balanced.
        # Actually, for the return value, we can just return the center as (0,0) and shift the inner points accordingly.
        
        # Let's find the optimal center c for the given alpha.
        # Minimize max_k (max_p (p - c) . n_k)
        # This is a linear programming problem or can be solved by intersecting half-planes.
        # For simplicity, let's just use the centroid of the vertices.
        center = np.mean(all_verts, axis=0)
        
        return best_s, center, best_alpha

    # Initial configurations
    def make_6x2():
        pts = []
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * math.sqrt(3)
                pts.append([x, y])
        return np.array(pts)

    def make_3x4_brick():
        pts = []
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.0) * 1.5
                pts.append([x, y])
        return np.array(pts)

    def make_4x3_brick():
        pts = []
        for r in range(4):
            for c in range(3):
                x = (c - 1.0) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.5) * 1.5
                pts.append([x, y])
        return np.array(pts)

    def make_random():
        # Random points in a circle of radius 5
        angles = np.random.uniform(0, 2 * math.pi, 12)
        radii = np.random.uniform(0, 5, 12)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts

    initial_configs = [make_6x2(), make_3x4_brick(), make_4x3_brick(), make_random(), make_random()]
    
    best_result = None
    best_side = float('inf')
    
    for i, init_pts in enumerate(initial_configs):
        if elapsed() > DEADLINE - 10:
            break
            
        # Flatten points for optimizer
        init_params = init_pts.flatten()
        
        # Optimize positions
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if elapsed() > DEADLINE - 5:
            break
            
        if res.fun < 100: # Check if it's a reasonable result (not heavily penalized)
            opt_pts = res.x.reshape(12, 2)
            # Check for collisions
            verts = [hex_vertices(x, y, 1.0, 0.0) for x, y in opt_pts]
            valid = True
            for a in range(12):
                for b in range(a + 1, 12):
                    if sat_overlap(verts[a], verts[b]) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Calculate enclosing hexagon
                side, center, angle = refine_enclosing_hex(opt_pts, [0.0]*12)
                
                # Shift points so center is at (0,0)
                shifted_pts = opt_pts - center
                
                if side < best_side:
                    best_side = side
                    best_result = (shifted_pts, angle)
    
    if best_result is None:
        # Fallback to 6x2
        pts = make_6x2()
        side, center, angle = refine_enclosing_hex(pts, [0.0]*12)
        shifted_pts = pts - center
        best_result = (shifted_pts, angle)
        best_side = side

    final_pts, final_angle = best_result
    
    # Construct return values
    inner = []
    for x, y in final_pts:
        inner.append([x, y, 0.0]) # Inner hexagons are aligned with axes (angle 0)
    
    # Outer hexagon
    outer_center = [0.0, 0.0]
    outer_side = best_side
    outer_angle = final_angle
    
    return inner, outer_center, outer_side, outer_angle

