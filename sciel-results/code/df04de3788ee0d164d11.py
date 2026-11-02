import math
import time
import numpy as np

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
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
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -a to align hexagon with axes
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
        # Check edge intersections
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

def _outer_penalty(inner_params, outer_params):
    """
    inner_params: list of (cx, cy, angle)
    outer_params: (cx, cy, s, angle)
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    
    penalty = 0.0
    for ip in inner_params:
        icx, icy, iang = ip
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            penalty += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
            
    return penalty

def _get_minimal_outer_size(inner_params, outer_angle):
    """
    Given a set of inner hexagons (all axis-aligned relative to global, but we assume they are placed 
    such that we want to find the minimal outer hexagon of a specific orientation that contains them).
    
    Actually, for a regular hexagon, the minimal size containing a set of points depends on the orientation.
    If we fix the outer hexagon's center at the centroid of the inner hexagons (or origin if symmetric)
    and its angle, we can compute the minimal side length.
    
    For a regular hexagon with side S and center (0,0) and angle 0 (flat top? pointy top?),
    the distance from center to edge (apothem) is S * sqrt(3) / 2.
    The distance from center to vertex is S.
    
    Let's define the outer hexagon with vertices at angles 0, 60, ... relative to its local frame.
    If the outer hexagon has side S, its support function in direction theta is:
    h(theta) = S * (sqrt(3)/2) / cos((theta - 30*k) mod 60 - 30) ... wait.
    
    Simpler:
    A regular hexagon with side S centered at origin, rotated by alpha.
    The distance from center to any edge is apothem = S * sqrt(3) / 2.
    The edges are perpendicular to directions 30 + 60*k + alpha.
    So, for the outer hexagon to contain a point P at distance d from center in direction phi,
    we need d * cos(phi - (30 + 60*k + alpha)) <= apothem for all k? No.
    
    The condition for a point (x,y) to be inside a hexagon centered at (0,0) with side S and rotation alpha:
    Let u = x*cos(-alpha) - y*sin(-alpha)
    Let v = x*sin(-alpha) + y*cos(-alpha)
    The hexagon is defined by |v| <= S*sqrt(3)/2 and |u +/- v/2| <= S*sqrt(3)/2?
    No, the standard hexagon inequalities for side S (pointy top, alpha=0):
    |y| <= S*sqrt(3)/2
    |x| + |y|/sqrt(3) <= S  (Wait, vertex at (S,0)? No, vertex at (S,0) means side S is distance from center to vertex? No, side length is S. Distance to vertex is S. Distance to edge is S*sqrt(3)/2.)
    
    Let's use the vertex definition.
    Vertices at S * (cos(60k), sin(60k)).
    The edges connect these vertices.
    The line for edge between v0(S,0) and v1(S/2, S*sqrt(3)/2) is:
    x * (S*sqrt(3)/2) - y * (S/2) = S^2 * sqrt(3) / 2 ?
    Let's check normal. Normal is (sqrt(3), -1). Length 2.
    Unit normal (sqrt(3)/2, -1/2).
    Distance from origin to line is S * sqrt(3) / 2.
    So the condition is:
    x * (sqrt(3)/2) - y * (1/2) <= S * sqrt(3) / 2
    And the 5 other similar inequalities.
    
    So, given a set of points (inner hexagon vertices), and a fixed outer rotation alpha,
    we can find the minimal S such that all points are inside.
    For each point (px, py), rotate by -alpha to (rx, ry).
    The constraints are:
    1. rx * cos(30) + ry * sin(30) <= S * sqrt(3)/2
    2. rx * cos(90) + ry * sin(90) <= S * sqrt(3)/2
    3. rx * cos(150) + ry * sin(150) <= S * sqrt(3)/2
    And the negatives.
    Basically, max over all 6 directions of (r * cos(phi - dir)) / (sqrt(3)/2) <= S.
    Where dir are 30, 90, 150, 210, 270, 330.
    
    So S_min = max_{p in points, k in 0..5} ( r_p * cos(phi_p - (30 + 60k + alpha)) ) / (sqrt(3)/2)
    
    We assume the outer hexagon is centered at the origin (0,0).
    This is valid if the inner packing is symmetric or we center the inner packing at the origin.
    """
    max_s = 0.0
    apothem_factor = math.sqrt(3) / 2
    alpha_rad = math.radians(outer_angle)
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for ip in inner_params:
        cx, cy, iang = ip
        v = _hex_vertices(cx, cy, 1.0, iang)
        all_verts.extend(v)
        
    for vx, vy in all_verts:
        # Rotate point by -alpha to align with outer hexagon's local frame
        rx = vx * math.cos(-alpha_rad) - vy * math.sin(-alpha_rad)
        ry = vx * math.sin(-alpha_rad) + vy * math.cos(-alpha_rad)
        
        # Check 6 constraints
        for k in range(6):
            theta = math.radians(30 + k * 60)
            proj = rx * math.cos(theta) + ry * math.sin(theta)
            # proj <= S * apothem_factor
            if proj > max_s * apothem_factor:
                s_needed = proj / apothem_factor
                if s_needed > max_s:
                    max_s = s_needed
                    
    return max_s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = float('inf')
    best_result = None
    
    # Strategy:
    # 1. Define a set of candidate inner configurations based on hexagonal lattice symmetry.
    # 2. For each configuration, try different outer rotations.
    # 3. Calculate the minimal outer size analytically.
    # 4. Verify validity (no overlaps).
    # 5. Keep the best valid result.
    
    # Candidate 1: Center + 6 Neighbors + 4 from Ring 2
    # Ring 1 neighbors are at distance 1.5 * 2 = 3? No.
    # Distance between centers of touching unit hexagons is 2 * apothem = sqrt(3) * 2? No.
    # Side 1. Apothem sqrt(3)/2. Distance between centers of two touching hexagons (sharing an edge) is 2 * apothem = sqrt(3).
    # Wait. If two hexagons share an edge, the distance between centers is 2 * apothem = sqrt(3).
    # Let's verify.
    # Hexagon 1 center (0,0). Rightmost edge is at x = sqrt(3)/2.
    # Hexagon 2 center (sqrt(3), 0). Leftmost edge is at x = sqrt(3) - sqrt(3)/2 = sqrt(3)/2.
    # So they touch. Distance is sqrt(3).
    
    # So, place 1 at (0,0).
    # Place 6 at distance sqrt(3) from center, angles 0, 60, ...
    # These 7 form a cluster.
    # We need 4 more.
    # Next ring: distance 2*sqrt(3) from center?
    # If we place hexagons at distance 2*sqrt(3) along the axes (0, 60, ...), they will touch the ring 1 hexagons?
    # Hexagon at (sqrt(3), 0). Next one along x-axis at (2*sqrt(3), 0).
    # Distance between them is sqrt(3). They touch.
    # So we can place 6 more at distance 2*sqrt(3) along the 6 axes.
    # That gives 1 + 6 + 6 = 13. We only need 11.
    # So we can pick any 4 of the 6 from the second ring.
    
    # Due to symmetry, picking 4 adjacent ones, or 4 alternating, etc. might yield different bounding boxes.
    # Let's try a few subsets of the 6 second-ring positions.
    
    r1 = math.sqrt(3)
    r2 = 2 * math.sqrt(3)
    
    base_inner = [(0.0, 0.0, 0.0)]
    for i in range(6):
        a = math.radians(60 * i)
        x = r1 * math.cos(a)
        y = r1 * math.sin(a)
        base_inner.append((x, y, 0.0))
        
    ring2_positions = []
    for i in range(6):
        a = math.radians(60 * i)
        x = r2 * math.cos(a)
        y = r2 * math.sin(a)
        ring2_positions.append((x, y, 0.0))
        
    # We need to choose 4 from ring2_positions.
    # There are C(6,4) = 15 combinations.
    # Let's iterate through them.
    
    import itertools
    
    for combo in itertools.combinations(range(6), 4):
        inner_params = base_inner.copy()
        for idx in combo:
            inner_params.append(ring2_positions[idx])
            
        # Try a few outer rotations
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            # Check validity (overlaps)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if _pairwise_penalty(inner_params[i], inner_params[j]) > 1e-9:
                        valid = False
                        break
                if not valid: break
                
            if not valid:
                continue
                
            # Calculate minimal outer size
            s_outer = _get_minimal_outer_size(inner_params, outer_angle)
            
            # The score is the outer side length.
            # We also need to return the center. We assumed (0,0).
            # Is (0,0) the optimal center?
            # For a symmetric cluster, yes.
            # For an asymmetric subset of ring 2, the centroid might be better?
            # But the problem allows the outer hexagon to be anywhere.
            # However, if we shift the outer hexagon, we might not need to shift the inner ones if we just calculate the bounding box.
            # The function _get_minimal_outer_size assumes the outer is centered at (0,0).
            # If the inner cluster is not centered at (0,0), this will give a larger S than necessary.
            # To be safe, we can calculate the centroid of the inner hexagons and shift everything to center it at (0,0) before calculating S.
            
            # Let's calculate centroid of inner centers
            cx_avg = sum(p[0] for p in inner_params) / N
            cy_avg = sum(p[1] for p in inner_params) / N
            
            # Shift inner params to center at (0,0)
            centered_inner = [(p[0] - cx_avg, p[1] - cy_avg, p[2]) for p in inner_params]
            
            # Recalculate S with centered inner
            s_outer_centered = _get_minimal_outer_size(centered_inner, outer_angle)
            
            # The actual outer hexagon in the global frame would be centered at (cx_avg, cy_avg)
            # and have side s_outer_centered.
            
            if s_outer_centered < best_score:
                best_score = s_outer_centered
                # Store the result in global coordinates
                # inner: original positions
                # outer center: (cx_avg, cy_avg)
                # outer side: s_outer_centered
                # outer angle: outer_angle
                best_result = (inner_params, (cx_avg, cy_avg), s_outer_centered, outer_angle)
                
            if time.time() > deadline:
                break
                
        if time.time() > deadline:
            break
            
    if best_result is None:
        # Fallback: just return the base 7 + 4 arbitrary, with a large outer box
        # This should not happen
        inner_params = base_inner + ring2_positions[:4]
        cx_avg = sum(p[0] for p in inner_params) / N
        cy_avg = sum(p[1] for p in inner_params) / N
        s_outer = 10.0
        best_result = (inner_params, (cx_avg, cy_avg), s_outer, 0.0)
        
    inner, center, side, angle = best_result
    
    # Format result
    inner_list = []
    for p in inner:
        inner_list.append([p[0], p[1], p[2]])
        
    return inner_list, center, side, angle

if __name__ == "__main__":
    r = construct()
    print(r)
