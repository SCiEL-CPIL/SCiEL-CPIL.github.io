"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate centers for a 3x4 hexagonal grid packing.
    # Unit hexagons with flat-top orientation.
    # Distance between centers in a row is 3 (2 * apothem * 3? No, 2 * 1 * sin(60) * 2 = 2 * sqrt(3) * ? No.)
    # Side length s=1.
    # Flat-top hexagon:
    # Width (horizontal distance between parallel vertical sides) = 2s = 2.
    # Height (vertical distance between parallel horizontal sides) = sqrt(3)s = sqrt(3).
    # Wait, flat-top means flat top and bottom.
    # Vertices at: (s,0), (s/2, sqrt(3)/2 s), (-s/2, sqrt(3)/2 s), (-s,0), (-s/2, -sqrt(3)/2 s), (s/2, -sqrt(3)/2 s).
    # Width = 2s. Height = sqrt(3)s.
    # To tile without overlap:
    # Horizontal spacing between columns: 1.5s (distance from center to left/right vertex is s, so gap is s? No.
    # Let's check:
    # Col 0 centers at x=0. Vertices at +/-1, +/-0.5.
    # Col 1 centers at x=1.5. Vertices at 1.5-1=0.5, 1.5+1=2.5.
    # At x=0.5, Col 0 has vertex. Col 1 has vertex at x=0.5? No, Col 1 leftmost is 0.5.
    # So they touch at x=0.5.
    # Vertical offset for Col 1: sqrt(3)/2.
    # So centers for col j (0..3) are at x = 1.5 * j.
    # y offset for odd j is sqrt(3)/2.
    # y spacing within a column is sqrt(3).
    
    centers = []
    for j in range(4): # 4 columns
        x = 1.5 * j
        for i in range(3): # 3 rows
            y = i * math.sqrt(3)
            if j % 2 == 1:
                y += math.sqrt(3) / 2.0
            centers.append((x, y))
            
    # 2. Center the set of points at (0,0)
    centers = np.array(centers)
    mean_x = np.mean(centers[:, 0])
    mean_y = np.mean(centers[:, 1])
    centers[:, 0] -= mean_x
    centers[:, 1] -= mean_y
    
    # 3. Optimize the outer hexagon rotation
    # The outer hexagon must contain all inner hexagons.
    # This is equivalent to finding the minimum enclosing hexagon.
    # The width of the convex hull of the *centers* expanded by the inradius of the inner hexagons?
    # No, we need the support function of the Minkowski sum of the set of points and the unit hexagon.
    # Or simpler: The maximum projection of the *outer boundary* of the packing onto the normals of the outer hexagon.
    # The outer boundary of a unit hexagon centered at p is p + H.
    # The projection of p + H onto normal n is p.n + H.n.
    # H.n is the support function of the unit hexagon.
    # For a flat-top unit hexagon, the support function h(theta) depends on the angle of the normal.
    # If the inner hexagons are not rotated, H is fixed.
    # If the outer hexagon is rotated by alpha, its normals are at angles alpha + k*pi/3.
    # We need S >= max( projection ) / (support of outer hexagon unit? No).
    # Let R be the circumradius of the outer hexagon? No, side length s_outer.
    # The distance from center to side (apothem) is a_outer = s_outer * sqrt(3)/2.
    # The condition is: For all normals n_k of the outer hexagon,
    # max_i ( (c_i + H_inner).dot(n_k) ) <= a_outer.
    # So a_outer >= max_k max_i ( c_i.dot(n_k) + h_inner(n_k) ).
    # And s_outer = 2 * a_outer / sqrt(3).
    
    # h_inner(n) for a flat-top unit hexagon:
    # The hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees.
    # The support function is piecewise linear.
    # Actually, since we can rotate the outer hexagon arbitrarily, we can just compute the max projection of the *expanded* set.
    # Let's define the "effective radius" R_eff(theta) = max_i ( c_i.dot(u(theta)) + h_inner(u(theta)) ).
    # We want to minimize max_{k=0..5} R_eff(alpha + k*pi/3) / cos(pi/6)? No.
    # The apothem a is the max projection onto the normals.
    # So we want to minimize M(alpha) = max_{k=0..5} R_eff(alpha + k*pi/3).
    # Then s_outer = 2 * M(alpha) / sqrt(3).
    
    # Let's compute h_inner(n) for flat-top hexagon.
    # Vertices v_j = (cos(pi/3 j), sin(pi/3 j)).
    # h(n) = max_j v_j.dot(n).
    
    def h_inner(theta):
        n = np.array([math.cos(theta), math.sin(theta)])
        max_proj = -np.inf
        for j in range(6):
            v = np.array([math.cos(math.pi/3 * j), math.sin(math.pi/3 * j)])
            max_proj = max(max_proj, np.dot(v, n))
        return max_proj
    
    def objective(alpha):
        max_val = 0.0
        for k in range(6):
            theta = alpha + k * math.pi / 3.0
            n = np.array([math.cos(theta), math.sin(theta)])
            h_val = h_inner(theta)
            # Max projection of centers
            proj_centers = np.dot(centers, n)
            max_c = np.max(proj_centers)
            total_max = max_c + h_val
            if total_max > max_val:
                max_val = total_max
        return max_val
    
    # Grid search for alpha
    best_alpha = 0.0
    best_val = float('inf')
    num_steps = 360
    for k in range(num_steps):
        alpha = k * math.pi / (3 * num_steps) # Period is pi/3
        val = objective(alpha)
        if val < best_val:
            best_val = val
            best_alpha = alpha
            
    # Refine with local search
    step = math.pi / (3 * num_steps)
    for _ in range(10):
        # Try neighbors
        found = False
        for da in [-step, step]:
            new_alpha = best_alpha + da
            new_val = objective(new_alpha)
            if new_val < best_val:
                best_alpha = new_alpha
                best_val = new_val
                found = True
        if not found:
            break
            
    side_length = 2.0 * best_val / math.sqrt(3.0)
    
    # 4. Construct the return values
    # inner rows: (x, y, angle_degrees)
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # outer center, side, angle
    center = (0.0, 0.0)
    angle_degrees = math.degrees(best_alpha)
    
    return inner, center, side_length, angle_degrees
# EVOLVE-BLOCK-END
