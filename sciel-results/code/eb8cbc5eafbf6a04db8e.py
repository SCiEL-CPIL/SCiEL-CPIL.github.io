import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    return np.array([cx + s * np.cos(angles), cy + s * np.sin(angles)]).T

def _pair_overlap(v1, v2):
    # v1, v2 are (6,2) arrays of vertices
    # Check if polygons intersect
    # Simple check: if any vertex of v1 is inside v2 or vice versa, or edges intersect
    # For speed, use bounding box first
    if not (np.all(v1[:,0] < v2[:,0].max()+1e-6) and np.all(v1[:,0] > v2[:,0].min()-1e-6) and
            np.all(v1[:,1] < v2[:,1].max()+1e-6) and np.all(v1[:,1] > v2[:,1].min()-1e-6)):
        return False
    
    # Check if any vertex of v1 is inside v2
    # Using winding number or ray casting is robust but slow
    # For hexagons, we can check if any vertex is inside the other
    # And check edge intersections
    
    # Helper: point in polygon (CCW vertices)
    def point_in_poly(p, poly):
        # p is (2,), poly is (6,2)
        x, y = p
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # Check vertices of v1 in v2
    for p in v1:
        if point_in_poly(p, v2):
            return True
    # Check vertices of v2 in v1
    for p in v2:
        if point_in_poly(p, v1):
            return True
            
    # Check edge intersections
    # Edge i of v1: v1[i] to v1[(i+1)%6]
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            # Check intersection of p1-p2 and p3-p4
            # Use cross product method
            def ccw(A, B, C):
                return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
                return True
    return False

def _inner_in_outer(inner_verts, outer_verts):
    # Check if all inner vertices are inside outer polygon
    # outer_verts is (6,2)
    # Use point_in_poly for each inner vertex
    # This is O(11 * 6 * 6) = 396 checks, acceptable
    def point_in_poly(p, poly):
        x, y = p
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for v in inner_verts:
        if not point_in_poly(v, outer_verts):
            return False
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    # Initial guess: 7-hex cluster + 4 at corners
    sqrt3 = np.sqrt(3)
    centers = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        # 4 additional at corners 0, 60, 180, 240 (skipping 120, 300 for symmetry? or just pick 4)
        # Let's place them at 0, 60, 120, 180
        [2*sqrt3, 0],
        [sqrt3 + sqrt3/2, 1.5 + 1.5], # Corner 60: C2 + vector (sqrt3/2, 1.5)
        [-sqrt3 - sqrt3/2, 1.5 + 1.5], # Corner 120: C3 + vector (-sqrt3/2, 1.5)
        [-2*sqrt3, 0]
    ])
    
    # Note: The "corners" of the 7-hex cluster are not exactly at C_i + C_i direction.
    # The neighbor of C1 (at sqrt3,0) in the 0 direction is at 2*sqrt3, 0.
    # The neighbor of C2 (at sqrt3/2, 1.5) in the 60 direction is at C2 + (sqrt3/2, 1.5).
    # This creates a "ring" of 12 hexagons around the center?
    # Total 1 + 6 + 4 = 11.
    
    angles = np.zeros(11)
    outer_center = np.array([0, 0])
    outer_angle = 0
    outer_side = 5.0
    
    # Flatten parameters for optimizer
    # x = [cx1, cy1, ax1, ..., cx11, cy11, ax11, ocx, ocy, oa, os]
    # 11*3 + 4 = 37 params
    
    def unpack(params):
        inner = params[:33].reshape(11, 3)
        outer_c = params[33:35]
        outer_a = params[35]
        outer_s = params[36]
        return inner, outer_c, outer_a, outer_s

    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
        inner, outer_c, outer_a, outer_s = unpack(params)
        
        # Generate vertices
        inner_verts = []
        for i in range(11):
            v = _hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2])
            inner_verts.append(v)
        
        outer_v = _hex_vertices(outer_c[0], outer_c[1], outer_s, outer_a)
        
        # Check overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if _pair_overlap(inner_verts[i], inner_verts[j]):
                    penalty += 100.0
        
        # Check containment
        all_inner_verts = np.vstack(inner_verts)
        if not _inner_in_outer(all_inner_verts, outer_v):
            # Calculate how far out the vertices are
            # Simple penalty: count violations
            def point_in_poly(p, poly):
                x, y = p
                inside = False
                n = len(poly)
                j = n - 1
                for i in range(n):
                    xi, yi = poly[i]
                    xj, yj = poly[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                return inside
            for v in all_inner_verts:
                if not point_in_poly(v, outer_v):
                    penalty += 10.0
                    
        return outer_s + penalty

    # Initial params
    p0 = np.concatenate([
        np.column_stack([centers, angles]).flatten(),
        [outer_center[0], outer_center[1], outer_angle, outer_side]
    ])
    
    best_result = None
    best_score = -1.0
    
    # Run optimization
    try:
        res = minimize(objective, p0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < 1e10:
            inner, outer_c, outer_a, outer_s = unpack(res.x)
            # Validate
            inner_verts = [_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(11)]
            outer_v = _hex_vertices(outer_c[0], outer_c[1], outer_s, outer_a)
            
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    if _pair_overlap(inner_verts[i], inner_verts[j]):
                        valid = False
                        break
            if valid:
                all_inner_verts = np.vstack(inner_verts)
                if not _inner_in_outer(all_inner_verts, outer_v):
                    valid = False
            
            if valid:
                # Construct return arrays
                inner_hex_data = inner
                outer_hex_data = np.array([outer_c[0], outer_c[1], outer_a])
                outer_hex_side_length = outer_s
                
                # Check edge length constraint
                # "shifted inner hexagons must keep edge lengths within 1e-8 of 1"
                # This is usually satisfied if we use exact 1.0 in _hex_vertices
                
                score = 1.0 / outer_hex_side_length
                if score > best_score:
                    best_score = score
                    best_result = (inner_hex_data, outer_hex_data, outer_hex_side_length)
                    
    except Exception as e:
        pass

    if best_result:
        return best_result
    
    # Fallback to seed-like result if optimization fails
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length
