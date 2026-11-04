import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize_scalar
from math import sqrt, pi, degrees, radians, cos, sin

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    a = radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * pi / 3.0
        # flat-top orientation? Use pointy-top? For unit hexagon, let's use vertices at 0,60,...
        # The "outer" hexagon orientation is also variable, so consistency in angle reference matters.
        # Use a standard: vertices at angle + k*60 deg
        vx = s * cos(theta)
        vy = s * sin(theta)
        verts.append((cx + vx, cy + vy))
    return verts

def construct():
    """
    Returns (inner, center, side, angle):
      inner: list of (x, y, angle_degrees) for 12 unit hexagons
      center: (cx, cy) outer hexagon center
      side: outer side length
      angle: outer rotation in degrees
    """
    # Hex lattice arrangement: center at origin, place unit hexagons on a triangular lattice
    # with spacing 2.0 (center-to-center for non-overlapping flat-top or pointy-top hexagons)
    # For pointy-top hexagons, the horizontal spacing is sqrt(3) and vertical is 1.5*2 = 3.0?
    # Let's use a simpler approach: place 12 hexagons in a 3x3 grid-like pattern with center omitted?
    # Actually, 12 hexagons can be placed as a 4x3 grid or 3x4, or a hexagonal cluster.
    
    # Use a honeycomb pattern: positions for 12 hexagons
    # In a honeycomb, adjacent centers are distance 2 apart (for unit hexagons)
    # Let's place them in rows:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons, offset
    # Row 2: 4 hexagons
    # But that's 12.
    
    # For pointy-top hexagons:
    # Center spacing: horizontal = sqrt(3), vertical = 1.5*2 = 3? No.
    # For unit side hexagon:
    #   Width (flat-top) = 2, Height (flat-top) = sqrt(3)
    #   Width (pointy-top) = sqrt(3), Height (pointy-top) = 2
    # Adjacent centers in a tiling are distance 2*sqrt(3)/sqrt(3)... let me compute.
    # In a regular tiling of unit hexagons, the center-to-center distance for adjacent hexagons is 2*sqrt(3)/sqrt(3) = 2? No.
    # Actually, for unit side hexagon, the apothem is sqrt(3)/2 * s = sqrt(3)/2.
    # Distance between centers = 2 * apothem = sqrt(3). Wait, that's for flat-top adjacent.
    # Let me just use: for flat-top hexagons, horizontal spacing = 3/2 * 2 = 3? No.
    
    # Simpler: use axial coordinates.
    # Flat-top: 
    #   x = 3/2 * q
    #   y = sqrt(3) * (r + q/2)
    # for axial (q, r), with distance between adjacent = sqrt(3) for unit hexagon? 
    # Actually the distance between centers of adjacent flat-top hexagons is 2*sqrt(3)/sqrt(3) = 2? 
    # Let me just place them manually.
    
    # Place 12 unit hexagons in a compact arrangement.
    # Use pointy-top orientation for all inner hexagons (angle 0).
    # Centers at:
    #   For a honeycomb tiling of pointy-top hexagons with side 1:
    #   The distance between adjacent centers is 2*sqrt(3)/sqrt(3) = 2? 
    #   Actually, for pointy-top: width = sqrt(3), height = 2.
    #   Horizontal spacing between columns = sqrt(3), vertical spacing within column = 2.
    #   Offset rows by sqrt(3)/2 horizontally and 1 vertically? 
    
    # Let's use a simple grid:
    # Positions in a 4x3 grid of hexagons, but that might not be optimal.
    
    # Better: use a hexagonal cluster of 12 hexagons:
    # Center hex + 6 around it + 5 more? That's 1+6+? 
    # Actually 1+6=7, need 12 more. 
    # Let's do a 3x4 rectangular-ish arrangement.
    
    # I'll place them on a triangular lattice with spacing 2*sqrt(3)/sqrt(3) = 2.0? 
    # No, the minimum distance between centers of non-overlapping unit hexagons is 2 (for flat-top, horizontal neighbors) or 2*sqrt(3)/sqrt(3)... 
    
    # Let me just use: centers at integer multiples of sqrt(3) in x and 1.5 in y, with offset.
    # For flat-top hexagons:
    #   x-spacing = 3/2 * 2 = 3? No, x-spacing = 3/2 * s? s=1, so 1.5
    #   y-spacing = sqrt(3)
    #   Odd columns offset by sqrt(3)/2 in y
    
    # Let's place 12 hexagons:
    # Column 0: y = 0, sqrt(3), 2*sqrt(3) -> 3 hexagons
    # Column 1: y = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2 -> 3 hexagons
    # Column 2: y = 0, sqrt(3), 2*sqrt(3) -> 3 hexagons
    # Column 3: y = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2 -> 3 hexagons
    # Total: 12
    
    sqrt3 = sqrt(3)
    centers = []
    for col in range(4):
        x = col * 1.5 * 2.0  # x-spacing = 3 for flat-top unit hexagons? 
        # Actually for flat-top hexagon of side s:
        # x-distance between adjacent columns = 3/2 * s = 1.5
        # y-distance between adjacent rows = sqrt(3) * s = sqrt(3)
        x = col * 1.5
        if col % 2 == 0:
            for row in range(3):
                y = row * sqrt3
                centers.append((x, y))
        else:
            for row in range(3):
                y = (row + 0.5) * sqrt3
                centers.append((x, y))
    
    # But these are for flat-top hexagons. Let's set inner angle to 0 (flat-top).
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
    
    # Now find minimum enclosing regular hexagon.
    all_verts = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts)
    
    all_verts = np.array(all_verts)
    
    def _find_min_hexagon(points, n_iter=50):
        """Find minimum side length regular hexagon enclosing all points."""
        def _hex_containment_check(points, A, cx, cy, theta):
            """Check if all points are inside a regular hexagon of side A centered at (cx,cy) with rotation theta.
            Regular hexagon with side A: the apothem is A*sqrt(3)/2.
            The hexagon can be described as intersection of 6 half-planes.
            """
            apothem = A * sqrt(3) / 2.0
            # Normals to the 6 sides
            normals = []
            for k in range(6):
                angle = theta + k * pi / 3.0
                normals.append((cos(angle), sin(angle)))
            normals = np.array(normals)
            
            # For each point, check that dot(point - center, normal) <= apothem for all 6 normals
            diffs = points - np.array([cx, cy])
            projections = diffs @ normals.T
            max_proj = np.max(projections, axis=0)
            return np.all(max_proj <= apothem + 1e-9)
        
        def _objective(theta):
            """For a given hexagon rotation theta, find minimum A and center (cx,cy) such that all points are inside."""
            apothem_min_A = lambda A: A * sqrt(3) / 2.0
            normals = []
            for k in range(6):
                angle = theta + k * pi / 3.0
                normals.append((cos(angle), sin(angle)))
            normals = np.array(normals)
            
            # LP: minimize A
            # Variables: [A, cx, cy]
            # Constraints: for each point p_i and each normal n_k:
            #   dot(n_k, p_i) - dot(n_k, c) <= apothem(A) = A * sqrt(3)/2
            #   => dot(n_k, p_i) - dot(n_k, c) - A*sqrt(3)/2 <= 0
            #   => -sqrt(3)/2 * A - n_kx * cx - n_ky * cy <= -dot(n_k, p_i)
            #   => sqrt(3)/2 * A + n_kx * cx + n_ky * cy >= dot(n_k, p_i)
            
            # We want to minimize A.
            # LP form: c^T x -> minimize, A_ub x <= b_ub
            # Let x = [A, cx, cy]
            # Objective: c = [1, 0, 0]
            # Constraints: for each (i, k): sqrt(3)/2 * A + n_kx * cx + n_ky * cy >= dot(n_k, p_i)
            # Rewrite as: -sqrt(3)/2 * A - n_kx * cx - n_ky * cy <= -dot(n_k, p_i)
            
            n_points = len(points)
            n_constraints = n_points * 6
            
            c_obj = np.array([1.0, 0.0, 0.0])
            A_ub = np.zeros((n_constraints, 3))
            b_ub = np.zeros(n_constraints)
            
            idx = 0
            for i in range(n_points):
                for k in range(6):
                    nkx, nky = normals[k]
                    dot_nk_pi = nkx * points[i, 0] + nky * points[i, 1]
                    A_ub[idx, 0] = -sqrt(3) / 2.0
                    A_ub[idx, 1] = -nkx
                    A_ub[idx, 2] = -nky
                    b_ub[idx] = -dot_nk_pi
                    idx += 1
            
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (None, None), (None, None)], method='highs')
            if res.success:
                return res.x[0], res.x[1], res.x[2]
            else:
                return np.inf, 0.0, 0.0
        
        best_A = np.inf
        best_cx = 0.0
        best_cy = 0.0
        best_theta = 0.0
        
        # Try a range of theta values
        thetas = np.linspace(0, pi / 3, 60, endpoint=False)
        for th in thetas:
            A, cx, cy = _objective(th)
            if A < best_A:
                best_A = A
                best_cx = cx
                best_cy = cy
                best_theta = th
        
        # Refine around best_theta
        if best_A < np.inf:
            res = minimize_scalar(lambda t: _objective(t)[0], bounds=(best_theta - pi/60, best_theta + pi/60), method='bounded')
            if res.success:
                best_theta = res.x
                best_A, best_cx, best_cy = _objective(best_theta)
        
        return best_A, best_cx, best_cy, best_theta
    
    side, cx, cy, theta_rad = _find_min_hexagon(all_verts)
    theta_deg = degrees(theta_rad)
    
    return inner, (cx, cy), side, theta_deg

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side:.4f}, Center: {center}, Angle: {angle:.2f}")
    for row in inner:
        print(row)
