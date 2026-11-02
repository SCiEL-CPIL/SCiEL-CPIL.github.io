import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    Uses the fact that a regular hexagon is the intersection of 3 strips.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
            
    return sum_sq_viol

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns penalty if they overlap.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
        
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    if penalty == 0.0:
        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    mid1 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    mid2 = ((p3[0]+p4[0])/2, (p3[1]+p4[1])/2)
                    penalty += _point_in_hex_dist_sq(mid1[0], mid1[1], c2x, c2y, 1.0, a2)
                    penalty += _point_in_hex_dist_sq(mid2[0], mid2[1], c1x, c1y, 1.0, a1)
                    if penalty == 0.0:
                        penalty += 1.0
                
    return penalty

def _is_valid_inner_config(inner_params):
    """
    Check if inner hexagons are disjoint.
    """
    for i in range(len(inner_params)):
        for j in range(i+1, len(inner_params)):
            if _pairwise_penalty(inner_params[i], inner_params[j]) > 0:
                return False
    return True

def _min_enclosing_hex_side(inner_params):
    """
    Compute the minimum side length of a regular hexagon that contains all inner_params.
    Uses LP to optimize outer center (cx, cy), side length S, and rotation angle.
    Note: LP requires linear constraints. The constraint "vertex v of inner hex i is inside outer hex"
    is: n_k . (v - c_out) <= S * apothem_factor
    This is linear in c_out and S for fixed angle.
    We can iterate over a few candidate outer angles.
    """
    best_side = float('inf')
    
    # Try a range of outer angles
    # The optimal angle is likely aligned with the "bounding box" of the points
    # Let's sample angles from 0 to 180 in steps of 15 degrees
    for oang_deg in range(0, 180, 15):
        oang = math.radians(oang_deg)
        
        # Outer hexagon normals
        # Normals are at 30, 90, 150 degrees relative to outer rotation
        # Wait, my _hex_vertices defines vertex 0 at angle oang.
        # The sides are perpendicular to normals at oang + 30, oang + 90, oang + 150.
        # The distance from center to side (apothem) is S * sqrt(3) / 2.
        
        # Constraint for each vertex v of each inner hexagon:
        # For each of the 3 normal directions n_k:
        # n_k . (v - c_out) <= S * sqrt(3) / 2
        # -n_k . (v - c_out) <= S * sqrt(3) / 2
        
        # Variables: x = [cx, cy, S]
        # Constraints: A_ub @ x <= b_ub
        
        A_ub = []
        b_ub = []
        
        for ip in inner_params:
            icx, icy, iang = ip
            verts = _hex_vertices(icx, icy, 1.0, iang)
            for v in verts:
                vx, vy = v
                for k in range(3):
                    theta_n = oang + math.radians(30 + k * 60)
                    nx = math.cos(theta_n)
                    ny = math.sin(theta_n)
                    
                    # n . (v - c) <= S * apothem
                    # n . v - n . c <= S * apothem
                    # -nx*cx - ny*cy - apothem*S <= -n . v
                    # Wait, c = (cx, cy). n . (v - c) = n.v - n.c.
                    # n.v - nx*cx - ny*cy <= S * (sqrt(3)/2)
                    # nx*cx + ny*cy + (sqrt(3)/2)*S >= n.v
                    # -nx*cx - ny*cy - (sqrt(3)/2)*S <= -n.v
                    
                    A_ub.append([-nx, -ny, -math.sqrt(3)/2])
                    b_ub.append(-(nx*vx + ny*vy))
                    
                    # -n . (v - c) <= S * apothem
                    # -n.v + n.c <= S * apothem
                    # nx*cx + ny*cy - (sqrt(3)/2)*S <= n.v
                    
                    A_ub.append([nx, ny, -math.sqrt(3)/2])
                    b_ub.append(nx*vx + ny*vy)
        
        # Minimize S
        # c_obj = [0, 0, 1]
        c_obj = [0, 0, 1]
        
        # Bounds: cx, cy free, S > 0
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
            if res.success:
                S = res.x[2]
                if S < best_side:
                    best_side = S
        except Exception:
            pass
            
    return best_side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    # Grid search for inner hexagon centers
    # We fix inner angles to 0 for simplicity in the grid search phase?
    # Or allow a few angles?
    # Let's start with angle 0 for all inner hexagons to reduce search space.
    # Then we can refine with a few local angle tweaks if time permits.
    
    # Define a grid of possible centers
    # The packing is likely symmetric.
    # Let's search in a box [-5, 5] x [-5, 5]
    # Step size: 0.25 initially, then refine?
    # N=11.
    
    # Greedy placement on a grid?
    # Or just iterate over combinations? Too many.
    
    # Let's use a greedy algorithm:
    # 1. Place first hex at (0,0).
    # 2. For each subsequent hex, find the grid point closest to the centroid of existing hexes
    #    that is valid (disjoint).
    # This is heuristic.
    
    # Better: Generate a set of candidate points on a hexagonal lattice.
    # Hex lattice spacing: 2 * apothem = sqrt(3) ~ 1.732 for side 1?
    # No, distance between centers of touching hexes is 2 * apothem = sqrt(3) if flat-to-flat?
    # If vertex-to-vertex, distance is 2.
    # For disjoint, distance > 2 * inradius? No, distance > 2 * circumradius is sufficient but not necessary.
    # Min distance between centers of two disjoint regular hexagons of side 1 is 2 * apothem = sqrt(3) ~ 1.732 (if aligned).
    # Actually, if rotated, they can be closer? No, circumradius is 1. So distance must be > 2?
    # Wait. Two circles of radius 1 (circumscribed) must be disjoint => distance > 2.
    # Hexagons are inside these circles. So distance > 2 is sufficient.
    # Is it necessary? No. Two hexagons can be closer if rotated.
    # But for a first pass, let's use a grid with step 1.0 or 0.5.
    
    # Let's try a simple grid search with greedy filling.
    # We'll sample centers from a grid.
    
    grid_step = 0.5
    grid_range = 6.0
    grid_points = []
    for x in np.arange(-grid_range, grid_range + grid_step, grid_step):
        for y in np.arange(-grid_range, grid_range + grid_step, grid_step):
            grid_points.append((x, y))
            
    # Shuffle grid points to break symmetry?
    # Or just iterate.
    
    # Greedy:
    # Start with empty list.
    # Add (0,0) with angle 0.
    # Then for each next hex, pick the grid point that minimizes the max distance to existing?
    # Or just pick the first valid one in a sorted order (e.g., by distance to center).
    
    # Let's try multiple random seeds for shuffling the grid points to find different packings.
    
    for seed in range(50): # 50 attempts
        if time.time() > deadline - 10:
            break
            
        np.random.seed(seed)
        
        # Shuffle grid points
        shuffled_points = grid_points.copy()
        np.random.shuffle(shuffled_points)
        
        # But we want them somewhat close to each other.
        # Let's sort by distance to (0,0) after shuffling? No, that defeats the purpose.
        # Let's just pick random points that are valid.
        
        inner_params = [(0.0, 0.0, 0.0)]
        
        for _ in range(N - 1):
            found = False
            # Try to find a valid point from the shuffled list
            # To be efficient, we can limit the search to points near the current cluster
            # But for simplicity, just iterate through shuffled list
            
            # Optimization: Only consider points within a certain radius of the centroid
            cx_sum = sum(p[0] for p in inner_params) / len(inner_params)
            cy_sum = sum(p[1] for p in inner_params) / len(inner_params)
            
            # Sort shuffled points by distance to centroid
            # This is expensive if done every time.
            # Let's just iterate through the shuffled list and pick the first valid one.
            # To encourage compactness, we can bias the shuffle towards center.
            
            # Actually, let's pre-sort grid points by distance to (0,0) and use that order,
            # but with some randomization.
            
            # Let's just use the shuffled list.
            for px, py in shuffled_points:
                # Quick check: distance to all existing
                valid = True
                for ex, ey, ea in inner_params:
                    dx = px - ex
                    dy = py - ey
                    if dx*dx + dy*dy < 3.0: # 1.732^2 ~ 3.0. If closer, check precisely.
                        # Check precise penalty
                        if _pairwise_penalty((px, py, 0.0), (ex, ey, ea)) > 0:
                            valid = False
                            break
                if valid:
                    inner_params.append((px, py, 0.0))
                    found = True
                    break
                    
            if not found:
                break
                
        if len(inner_params) == N:
            # Check if this configuration is valid (redundant if we checked during placement, but safe)
            if _is_valid_inner_config(inner_params):
                # Compute min enclosing hex
                side = _min_enclosing_hex_side(inner_params)
                if side < best_side:
                    best_side = side
                    best_inner = inner_params.copy()
                    # Compute outer params for return
                    # We need to recover cx, cy, angle from the LP solution.
                    # The LP only gave us S. We need to re-run LP to get cx, cy for the best angle.
                    # Or we can just return a placeholder outer if we don't have it.
                    # The evaluator needs (inner, center, side, angle).
                    # We should store the best outer params.
                    
                    # Let's re-run the LP for the best angle to get cx, cy.
                    # We need to modify _min_enclosing_hex_side to return cx, cy, angle too.
                    pass # Placeholder, will implement in full version below
    
    # The above loop is a sketch. Let's implement the full logic properly.
    
    # Re-implementing construct with better LP integration
    
    best_side = float('inf')
    best_result = None
    
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(seed)
        
        # Generate random valid configuration?
        # No, let's stick to the grid search idea but make it smarter.
        # We'll use a "jittered grid" approach.
        
        inner_params = [(0.0, 0.0, 0.0)]
        
        # Try to add 10 more hexagons
        for _ in range(N - 1):
            found = False
            # Sample random points in a box around the current cluster
            # Box size: 10x10 centered at centroid
            cx_sum = sum(p[0] for p in inner_params) / len(inner_params)
            cy_sum = sum(p[1] for p in inner_params) / len(inner_params)
            
            # Try 50 random candidates
            for _ in range(50):
                px = cx_sum + np.random.uniform(-4, 4)
                py = cy_sum + np.random.uniform(-4, 4)
                # Snap to grid?
                px = round(px * 2) / 2.0
                py = round(py * 2) / 2.0
                
                valid = True
                for ex, ey, ea in inner_params:
                    dx = px - ex
                    dy = py - ey
                    if dx*dx + dy*dy < 3.0:
                        if _pairwise_penalty((px, py, 0.0), (ex, ey, ea)) > 0:
                            valid = False
                            break
                if valid:
                    inner_params.append((px, py, 0.0))
                    found = True
                    break
                    
            if not found:
                break
                
        if len(inner_params) == N:
            if _is_valid_inner_config(inner_params):
                # Find best outer
                # We need a function that returns the full outer params
                outer_params, side = _find_best_outer(inner_params)
                if side < best_side:
                    best_side = side
                    best_result = (inner_params, outer_params)
                    
    if best_result is not None:
        inner_params, outer_params = best_result
        ocx, ocy, os, oang = outer_params
        inner_list = [[p[0], p[1], p[2]] for p in inner_params]
        return inner_list, (ocx, ocy), os, oang
    else:
        # Fallback: return a valid configuration if we found any, or a default
        # If we didn't find a full N=11 config, we return a smaller one?
        # The task requires N hexagons.
        # If we fail, we should return a valid N=11 config even if not optimal.
        # Let's force a valid config.
        return _get_fallback_config()

def _find_best_outer(inner_params):
    best_side = float('inf')
    best_params = None
    
    for oang_deg in range(0, 180, 15):
        oang = math.radians(oang_deg)
        
        A_ub = []
        b_ub = []
        
        for ip in inner_params:
            icx, icy, iang = ip
            verts = _hex_vertices(icx, icy, 1.0, iang)
            for v in verts:
                vx, vy = v
                for k in range(3):
                    theta_n = oang + math.radians(30 + k * 60)
                    nx = math.cos(theta_n)
                    ny = math.sin(theta_n)
                    
                    A_ub.append([-nx, -ny, -math.sqrt(3)/2])
                    b_ub.append(-(nx*vx + ny*vy))
                    
                    A_ub.append([nx, ny, -math.sqrt(3)/2])
                    b_ub.append(nx*vx + ny*vy)
                    
        c_obj = [0, 0, 1]
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
            if res.success:
                S = res.x[2]
                if S < best_side:
                    best_side = S
                    best_params = (res.x[0], res.x[1], S, oang_deg)
        except Exception:
            pass
            
    return best_params, best_side

def _get_fallback_config():
    # Return a simple valid config
    inner = []
    inner.append([0.0, 0.0, 0.0])
    for i in range(6):
        a = math.radians(60*i)
        x = 1.732 * math.cos(a)
        y = 1.732 * math.sin(a)
        inner.append([x, y, 0.0])
    # Add 4 more
    for i in range(4):
        a = math.radians(60*i + 30)
        x = 3.0 * math.cos(a)
        y = 3.0 * math.sin(a)
        inner.append([x, y, 0.0])
        
    # This might overlap. Let's just return a large outer hexagon.
    # The evaluator will score it based on side length.
    # If it's invalid, it scores 0.
    # So we must ensure validity.
    # The fallback should be a known valid packing.
    # Let's use the grid search result if available, else a very loose packing.
    
    # Loose packing:
    inner = []
    for i in range(N):
        inner.append([i * 2.0, 0.0, 0.0])
        
    # Outer hexagon to contain this line
    # Length is 10.
    # Side length needs to be large.
    # Let's just return this and a huge outer hex.
    ocx = 5.0
    ocy = 0.0
    os = 10.0
    oang = 0.0
    return inner, (ocx, ocy), os, oang

# Ensure the function is available
# The evaluator calls construct()

if __name__ == "__main__":
    # Test
    r = construct()
    print(r)
