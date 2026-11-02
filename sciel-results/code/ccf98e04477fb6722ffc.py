"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon centered at (cx,cy) with side length and rotation."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3.0
            vertices.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return np.array(vertices)
    
    def polygon_collision_check(poly1, poly2, eps=1e-9):
        """Check if two convex polygons collide (share interior or boundary)."""
        # For simplicity in this packing problem, we check if any vertex of one is inside the other
        # This is a conservative check; for hexagons with rotation, we need more robust checks
        def point_in_polygon(px, py, poly):
            n = len(poly)
            inside = False
            for i in range(n):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % n]
                if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
                    inside = not inside
            return inside
        
        v1 = poly1
        v2 = poly2
        
        # Check if any vertex of poly1 is inside poly2
        for i in range(len(v1)):
            if point_in_polygon(v1[i, 0], v1[i, 1], v2):
                return True
        
        # Check if any vertex of poly2 is inside poly1
        for i in range(len(v2)):
            if point_in_polygon(v2[i, 0], v2[i, 1], v1):
                return True
        
        return False
    
    def get_hexagon_corners(cx, cy, side=1.0, angle_deg=0.0):
        """Get the 6 corners of a unit hexagon centered at (cx,cy) with given rotation."""
        angle_rad = math.radians(angle_deg)
        corners = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3.0
            corners.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return np.array(corners)
    
    def min_enclosing_hexagon(inner_hexes, outer_center, outer_angle_deg, max_iter=100, tol=1e-6):
        """
        Find minimum side length of outer hexagon centered at outer_center with rotation outer_angle_deg
        that contains all inner hexagons.
        
        Uses linear programming approach: for a convex polygon, the minimum enclosing hexagon
        with fixed orientation can be found by finding the maximum over all supporting lines.
        
        For a hexagon with fixed orientation, the constraint is that each vertex of each inner hexagon
        must be inside the outer hexagon. The outer hexagon is defined by 6 half-planes.
        
        We binary search on the side length and check feasibility.
        """
        ocx, ocy = outer_center
        oa_rad = math.radians(outer_angle_deg)
        
        # The 6 normal directions of the outer hexagon
        normals = []
        for i in range(6):
            a = oa_rad + i * math.pi / 3.0
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        # For each inner hexagon, get all vertices
        all_vertices = []
        for hx, hy, ha in inner_hexes:
            corners = get_hexagon_corners(hx, hy, 1.0, ha)
            all_vertices.append(corners)
        
        def is_feasible(side_len):
            """Check if outer hexagon with given side length contains all inner hexagons."""
            # The outer hexagon centered at (ocx, ocy) with side length s and rotation oa_rad
            # A point (px, py) is inside if for all 6 normals n_i:
            # n_i . (p - center) <= s * cos(pi/6) = s * sqrt(3)/2
            # Wait, for a regular hexagon with side length s, the apothem (distance from center to side) is s * sqrt(3)/2
            apothem = side_len * math.sqrt(3) / 2.0
            
            for corners in all_vertices:
                for v in corners:
                    rel = v - np.array([ocx, ocy])
                    for n in normals:
                        if np.dot(rel, n) > apothem + 1e-9:
                            return False
            return True
        
        # Binary search for minimum side length
        lo = 0.0
        hi = 100.0  # Upper bound
        
        # First check if hi is feasible
        if not is_feasible(hi):
            hi = 200.0
            if not is_feasible(hi):
                hi = 500.0
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return hi  # Return current best guess
            mid = (lo + hi) / 2.0
            if is_feasible(mid):
                hi = mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        
        return hi
    
    def try_configuration(inner_hexes, outer_center, outer_angle_deg):
        """Try a configuration and return the outer side length if valid, else None."""
        if time.time() > deadline:
            return None
        
        # Check for collisions between inner hexagons
        n = len(inner_hexes)
        corners_list = []
        for hx, hy, ha in inner_hexes:
            corners = get_hexagon_corners(hx, hy, 1.0, ha)
            corners_list.append(corners)
        
        for i in range(n):
            for j in range(i + 1, n):
                if polygon_collision_check(corners_list[i], corners_list[j]):
                    return None
        
        # Find minimum enclosing hexagon
        side_len = min_enclosing_hexagon(inner_hexes, outer_center, outer_angle_deg)
        return side_len
    
    # Initial configuration: grid of hexagons
    # For regular hexagons, the optimal packing in a hexagonal lattice has specific spacing
    # Horizontal spacing between centers: 2 * sqrt(3) for same-row (if rotated 0 deg)
    # Actually, for flat-top hexagons with side 1:
    # Width = 2, Height = sqrt(3)
    # Horizontal spacing = 1.5 (center to center for adjacent columns)
    # Vertical spacing = sqrt(3)
    
    best_side = float('inf')
    best_config = None
    
    # Try different lattice arrangements
    # For N=11 hexagons, try a 3-4-3 or 2-3-3-3-2 or similar arrangement
    
    arrangements = [
        # (rows, cols per row)
        (1, [11]),
        (2, [6, 5]),
        (3, [4, 4, 3]),
        (3, [3, 5, 3]),
        (4, [3, 4, 4, 0]),  # Invalid, will be filtered
        (5, [2, 3, 3, 3, 2]),
        (4, [2, 4, 4, 1]),
        (3, [4, 3, 4]),
        (2, [5, 6]),
        (1, [11]),
        (5, [2, 2, 3, 2, 2]),
        (4, [3, 3, 3, 2]),
        (3, [5, 4, 2]),
        (4, [2, 3, 4, 2]),
    ]
    
    for row_arr in arrangements:
        if time.time() > deadline:
            break
        
        n_rows = row_arr[0]
        cols = row_arr[1]
        total = sum(cols)
        if total != N:
            continue
        
        # Generate hexagon positions in a hexagonal lattice
        # For flat-top hexagons (angle=0):
        # Center spacing: horizontal = 3 (for non-adjacent columns), but for adjacent columns it's 1.5
        # Let's use a standard hex grid
        
        inner_hexes = []
        hex_spacing_h = 3.0  # Horizontal distance between centers in same row
        hex_spacing_v = math.sqrt(3)  # Vertical distance between rows
        
        # Actually, for unit hexagons with flat top:
        # The distance between centers of two horizontally adjacent hexagons is 3 (since width is 2, gap is 1)
        # Wait, for touching hexagons: center-to-center distance = 2 * apothem = 2 * sqrt(3)/2 * 2 = 2*sqrt(3)? No.
        # For two touching flat-top hexagons side by side, the distance between centers is 3 (1.5 + 1.5)
        # The width of a flat-top hexagon with side 1 is 2, and the "radius" to vertex is 1.
        # Center to side distance (apothem) = sqrt(3)/2
        # So for two hexagons touching on their sides, center distance = 2 * apothem = sqrt(3)
        
        # Let me reconsider. For flat-top regular hexagon with side s=1:
        # - Width (flat to flat) = sqrt(3)
        # - Height (point to point) = 2
        # - Apothem = sqrt(3)/2
        # Two hexagons sharing a vertical edge: center distance = 2 * apothem = sqrt(3)
        
        # For a hexagonal lattice of flat-top hexagons:
        # Same row: horizontal spacing = 3 (each hex is 2 wide, but they interlock)
        # Actually, let's just use the standard hex grid spacing.
        
        # For flat-top hexagons in a hex grid:
        # Column spacing: 1.5 (horizontal component)
        # Row spacing: sqrt(3)
        # Odd columns offset by sqrt(3)/2 vertically
        
        # Let me just place them in a simple grid and check for collisions.
        
        positions = []
        for r in range(n_rows):
            for c in range(cols[r]):
                # Place in a grid with some spacing
                x = c * 2.2  # Slight gap to avoid collision
                y = r * math.sqrt(3)
                # Offset every other row
                if r % 2 == 1:
                    x += 1.1
                positions.append((x, y, 0.0))
        
        inner_hexes = positions
        
        # Try different outer rotations
        for outer_angle in [0.0, 30.0, 60.0, 15.0, 45.0]:
            if time.time() > deadline:
                break
            
            # Center the configuration
            xs = [p[0] for p in inner_hexes]
            ys = [p[1] for p in inner_hexes]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            outer_center = (cx, cy)
            
            side_len = try_configuration(inner_hexes, outer_center, outer_angle)
            if side_len is not None and side_len < best_side:
                best_side = side_len
                best_config = (inner_hexes, outer_center, outer_angle)
    
    # If no valid configuration found, return a trivial one
    if best_config is None:
        # Fallback: place all in a line with large spacing
        inner_hexes = [[3.0 * i, 0.0, 0.0] for i in range(N)]
        return inner_hexes, (0.0, 0.0), 20.0, 0.0
    
    inner_hexes, outer_center, outer_angle = best_config
    return inner_hexes, outer_center, best_side, outer_angle
# EVOLVE-BLOCK-END
