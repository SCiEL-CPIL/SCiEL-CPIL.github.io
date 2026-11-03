import numpy as np
from scipy.optimize import minimize
import time

N = 12
_start_time = time.time()
_DEADLINE = 140.0  # 10s safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + np.pi * k / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    # Rotate back
    ca, sa = np.cos(-a), np.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # Hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees, side length s
    # The hexagon has apothem = side * sqrt(3)/2
    # In local coords, the hexagon edges are at x = +/- apothem for the left/right edges
    # Actually, for a hexagon with a vertex at angle 0 (pointing right), the edges are:
    # Right: x = side (vertex), but the edge is from (side/2, side*sqrt(3)/2) to (side/2, -side*sqrt(3)/2)
    # Let me use the half-width approach
    # For a regular hexagon with side s, the distance from center to each edge (apothem) is s*sqrt(3)/2
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees (if vertex at 0)
    # Actually if vertices are at 0, 60, 120, 180, 240, 300, then edges are perpendicular to directions 30, 90, 150, 210, 270, 330
    apothem = side * np.sqrt(3) / 2
    # Edge normals at 30, 90, 150, 210, 270, 330 degrees
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        nx = np.cos(theta)
        ny = np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-9:
            return False
    return True

def _hex_contains_hexagon(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if outer hexagon contains the inner unit hexagon."""
    # Check all 6 vertices of inner hexagon
    inner_verts = _hex_vertices(inner_cx, inner_cy, 1.0, inner_angle)
    for v in inner_verts:
        if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def _check_valid(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside outer hexagon and don't overlap each other."""
    cx, cy = outer_center
    # Check each inner hexagon is inside outer
    for (ix, iy, iang) in inner:
        if not _hex_contains_hexagon(ix, iy, iang, cx, cy, outer_side, outer_angle):
            return False
    # Check pairwise non-overlap: for two convex polygons, check if any edge of one intersects any edge of other
    # Simpler: check if any vertex of one is inside the other (for convex shapes, if no vertex is inside and no edges cross, they don't overlap)
    # But this is not sufficient. Let's use a more robust check.
    # For two convex polygons, they overlap if and only if there is no separating axis.
    # Separating axes are the edge normals of both polygons.
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            vi = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vj = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _polygons_overlap(vi, vj):
                return False
    return True

def _polygons_overlap(p1, p2):
    """Check if two convex polygons overlap using SAT."""
    polys = [p1, p2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            # Edge from poly[i] to poly[(i+1)%n]
            edge = poly[(i + 1) % n] - poly[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal = normal / np.linalg.norm(normal)
            # Project both polygons onto this normal
            proj1 = np.dot(p1, normal)
            proj2 = np.dot(p2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False
    return True

def _min_enclosing_hexagon(inner_positions, max_side=10.0):
    """Find minimal side length hexagon (with any center and rotation) containing all inner hexagons."""
    # We'll optimize over outer center (cx, cy), outer side (s), and outer rotation (theta)
    # Objective: minimize s
    # Constraints: all inner hexagon vertices inside outer hexagon
    
    def objective(params):
        cx, cy, s, theta = params
        if s < 1.0:
            return 1000.0
        # Check all inner hexagon vertices
        max_violation = 0.0
        for (ix, iy, iang) in inner_positions:
            verts = _hex_vertices(ix, iy, 1.0, iang)
            for v in verts:
                if not _point_in_hexagon(v[0], v[1], cx, cy, s, theta):
                    # Compute how far outside
                    dx = v[0] - cx
                    dy = v[1] - cy
                    a = np.radians(theta)
                    ca, sa = np.cos(-a), np.sin(-a)
                    lx = dx * ca - dy * sa
                    ly = dx * sa + dy * ca
                    apothem = s * np.sqrt(3) / 2
                    for k in range(6):
                        th = np.radians(30 + 60 * k)
                        nx = np.cos(th)
                        ny = np.sin(th)
                        dist = lx * nx + ly * ny
                        if dist > apothem:
                            max_violation = max(max_violation, dist - apothem)
        return s + 10.0 * max_violation
    
    # Initial guess: center at centroid, side large enough
    all_pts = []
    for (ix, iy, iang) in inner_positions:
        verts = _hex_vertices(ix, iy, 1.0, iang)
        all_pts.extend(verts.tolist())
    all_pts = np.array(all_pts)
    cx0 = all_pts[:, 0].mean()
    cy0 = all_pts[:, 1].mean()
    s0 = max(np.max(all_pts[:, 0]) - np.min(all_pts[:, 0]), np.max(all_pts[:, 1]) - np.min(all_pts[:, 1])) / np.sqrt(3) + 2.0
    theta0 = 0.0
    
    best_result = None
    best_val = float('inf')
    
    # Try multiple initial rotations
    for theta_init in [0, 30, 60, 90, 120, 150]:
        x0 = [cx0, cy0, s0, theta_init]
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    
    if best_result is None:
        return None, float('inf')
    
    cx, cy, s, theta = best_result
    # Verify
    if _check_valid(inner_positions, (cx, cy), s, theta):
        return (cx, cy, s, theta), s
    else:
        # Try to fix by increasing s slightly
        for factor in [1.01, 1.05, 1.1, 1.2]:
            s2 = s * factor
            if _check_valid(inner_positions, (cx, cy), s2, theta):
                return (cx, cy, s2, theta), s2
        return None, float('inf')

# EVOLVE-BLOCK-START
def construct():
    deadline = _start_time + _DEADLINE
    
    # Candidate arrangements for 12 unit hexagons
    # Arrangement 1: Triangular lattice, 3-4-3-2 rows
    arrangements = []
    
    # Arrangement A: Compact triangular cluster
    # Using triangular lattice with spacing sqrt(3)
    d = np.sqrt(3)
    # Positions in triangular lattice coordinates
    # Let's try a few patterns
    
    # Pattern 1: 4-4-4 in a rectangular-ish grid with triangular offset
    # Row 0: 4 hexagons at x = 0, d, 2d, 3d
    # Row 1: 4 hexagons at x = d/2, 3d/2, 5d/2, 7d/2, y = d*sqrt(3)/2
    # Row 2: 4 hexagons at x = 0, d, 2d, 3d, y = d*sqrt(3)
    # Total: 12
    d3 = np.sqrt(3)
    h = d3 * np.sqrt(3) / 2  # = 1.5
    
    # Pattern 1: 3 rows of 4
    inner1 = []
    for row in range(3):
        for col in range(4):
            x = col * d3 + (row % 2) * d3 / 2
            y = row * h
            inner1.append([x, y, 0.0])
    arrangements.append(inner1)
    
    # Pattern 2: 2-4-4-2 (more symmetric)
    inner2 = []
    # Row 0: 2 hexagons
    for col in range(2):
        x = col * d3 + d3
        y = 0
        inner2.append([x, y, 0.0])
    # Row 1: 4 hexagons
    for col in range(4):
        x = col * d3 + d3 / 2
        y = h
        inner2.append([x, y, 0.0])
    # Row 2: 4 hexagons
    for col in range(4):
        x = col * d3 + d3 / 2
        y = 2 * h
        inner2.append([x, y, 0.0])
    # Row 3: 2 hexagons
    for col in range(2):
        x = col * d3 + d3
        y = 3 * h
        inner2.append([x, y, 0.0])
    arrangements.append(inner2)
    
    # Pattern 3: Hexagonal cluster - center + 6 around + 5 more
    # Central hexagon at (0,0)
    # 6 hexagons at distance d3 in directions 0, 60, 120, 180, 240, 300
    # Then 5 more at the "corners"
    inner3 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        x = d3 * np.cos(angle)
        y = d3 * np.sin(angle)
        inner3.append([x, y, 0.0])
    # Add 5 more at the next ring positions (but only 5 to make 12)
    # The next ring has 12 positions at distance 2*d3
    # Pick 5 that are compact
    for k in range(5):
        angle = np.radians(60 * k + 30)  # Offset by 30 degrees
        x = 2 * d3 * np.cos(angle)
        y = 2 * d3 * np.sin(angle)
        inner3.append([x, y, 0.0])
    arrangements.append(inner3)
    
    # Pattern 4: 5-4-3 arrangement
    inner4 = []
    for row in range(3):
        count = 5 - row
        for col in range(count):
            x = col * d3 + row * d3 / 2
            y = row * h
            inner4.append([x, y, 0.0])
    arrangements.append(inner4)
    
    # Pattern 5: 4-5-3 arrangement
    inner5 = []
    for row in range(3):
        counts = [4, 5, 3]
        for col in range(counts[row]):
            x = col * d3 + row * d3 / 2
            y = row * h
            inner5.append([x, y, 0.0])
    arrangements.append(inner5)
    
    # Pattern 6: 3-5-4 arrangement
    inner6 = []
    for row in range(3):
        counts = [3, 5, 4]
        for col in range(counts[row]):
            x = col * d3 + row * d3 / 2
            y = row * h
            inner6.append([x, y, 0.0])
    arrangements.append(inner6)
    
    # Pattern 7: 6-6 in two rows
    inner7 = []
    for row in range(2):
        for col in range(6):
            x = col * d3 + (row % 2) * d3 / 2
            y = row * h
            inner7.append([x, y, 0.0])
    arrangements.append(inner7)
    
    # Pattern 8: 3-4-3-2 with better centering
    inner8 = []
    # Center the cluster
    # Row 0: 3, Row 1: 4, Row 2: 3, Row 3: 2
    offsets = [1.5, 1.0, 1.5, 2.0]  # x-offsets to center each row
    for row in range(4):
        counts = [3, 4, 3, 2]
        for col in range(counts[row]):
            x = col * d3 + offsets[row] * d3
            y = row * h
            inner8.append([x, y, 0.0])
    arrangements.append(inner8)
    
    best_side = float('inf')
    best_result = None
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        outer, side = _min_enclosing_hexagon(arr)
        if outer is not None and side < best_side:
            best_side = side
            best_result = (arr, outer)
    
    if best_result is None:
        # Fallback: return a valid but large configuration
        inner = [[k * 2.0, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    inner, (cx, cy, s, theta) = best_result
    return inner, (cx, cy), s, theta
# EVOLVE-BLOCK-END
