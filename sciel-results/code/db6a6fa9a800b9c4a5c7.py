"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def polygon_area(vertices):
    """Compute area of a polygon given its vertices using the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside a regular hexagon."""
    # Transform to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = -np.radians(angle_deg)
    x_local = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    y_local = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    
    # Regular hexagon with side length s has vertices at distance s from center
    # The hexagon can be described as the intersection of 6 half-planes
    # In local coords, the hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # Actually, for a flat-top hexagon, vertices are at angles 0, 60, ..., 300
    # The distance from center to each side (apothem) is s * sqrt(3)/2
    # The half-width in x direction is s, in y direction is s * sqrt(3)/2
    
    # Let's use the standard representation: a regular hexagon with side s centered at origin
    # with one vertex pointing right has vertices at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The constraints are:
    # |x| <= s
    # |y| <= s * sqrt(3)/2
    # |x| + |y|/sqrt(3) <= s  (this isn't quite right)
    
    # Better approach: check against all 6 edges
    vertices = hex_vertices(0, 0, side, 0)
    # For each edge, check if point is on the inside
    for i in range(6):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 6]
        # Edge vector
        ex = x2 - x1
        ey = y2 - y1
        # Normal pointing inward (for CCW polygon, left normal is inward)
        nx = ey
        ny = -ex
        # Check if point is on the inside of this edge
        val = nx * x_local + ny * y_local
        # The center should give a positive value if normal is inward
        if val < -1e-10:
            return False
    return True

def max_distance_to_inner_hexagons(cx, cy, outer_side, outer_angle):
    """Find the maximum distance from outer hexagon center to any point on inner hexagon boundaries."""
    # Sample points on each inner hexagon's boundary
    max_dist = 0.0
    for i in range(N):
        ix, iy, iangle = inner_positions[i]
        # Get vertices of this inner hexagon
        verts = hex_vertices(ix, iy, 1.0, iangle)
        # Add midpoints of edges for better sampling
        all_points = []
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            all_points.append(v1)
            all_points.append((v1 + v2) / 2)
        for p in all_points:
            dx = p[0] - cx
            dy = p[1] - cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > max_dist:
                max_dist = dist
    return max_dist

def is_point_in_outer_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if a point is inside the outer hexagon."""
    return point_in_hexagon(px, py, cx, cy, side, angle_deg)

def objective(params):
    """Objective function: minimize the outer hexagon side length needed to contain all inner hexagons."""
    # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    global inner_positions
    inner_positions = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        inner_positions.append((x, y, a))
    
    # Find the minimum outer hexagon that contains all inner hexagons
    # We'll optimize the outer hexagon's center, side, and angle
    # But for the objective, let's just compute the bounding hexagon
    
    # Simple approach: compute the convex hull of all inner hexagon vertices
    all_vertices = []
    for i in range(N):
        x, y, a = inner_positions[i]
        verts = hex_vertices(x, y, 1.0, a)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Compute the minimum enclosing regular hexagon
    # This is complex, so let's use a simpler heuristic:
    # Find the center and maximum extent
    cx = np.mean(all_vertices[:, 0])
    cy = np.mean(all_vertices[:, 1])
    
    # For each angle, compute the required side length
    min_side = float('inf')
    for outer_angle in np.linspace(0, 60, 13):
        # For this outer angle, find the minimum side length
        # The outer hexagon with side s and angle theta contains point (px, py) if
        # point_in_hexagon(px, py, cx, cy, s, theta) is true
        # We need to find the minimum s such that all points are inside
        
        # Binary search on s
        lo = 0.0
        hi = 100.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if all(is_point_in_outer_hexagon(v[0], v[1], cx, cy, mid, outer_angle) for v in all_vertices):
                hi = mid
            else:
                lo = mid
        min_side = min(min_side, hi)
    
    return min_side

# Use a different approach: place hexagons in a known optimal pattern
# For 12 unit hexagons, a good arrangement is a hexagonal cluster
# The optimal packing of 12 hexagons is known to form a larger hexagon shape

def construct():
    """Construct 12 unit hexagons inside a minimal outer hexagon."""
    # Known optimal arrangement for 12 hexagons:
    # Arrange in a hexagonal pattern with rows of 3, 4, 4, 3 or similar
    
    # Let's try a symmetric arrangement:
    # Center hexagon at origin, with 6 around it, and 5 more
    
    # Actually, the minimal container for 12 unit hexagons:
    # A regular hexagon of side length 4 can contain 13 hexagons in a hexagonal pattern
    # (1 in center, 6 in first ring, 6 in second ring... wait, that's 1+6+12=19 for side 3)
    
    # For side length s outer hexagon, the number of unit hexagons that fit in a hexagonal lattice
    # is roughly proportional to s^2. For s=4, we can fit about 13-16 hexagons.
    
    # Let's use a hexagonal lattice arrangement:
    # Place hexagons at positions that form a compact cluster
    
    # Hexagonal lattice positions for 12 hexagons:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    positions = []
    # Using axial coordinates for hexagonal grid
    # Convert to Cartesian: x = 1.5 * q, y = sqrt(3)/2 * q + sqrt(3) * r
    
    sqrt3 = np.sqrt(3)
    
    # Define positions in hex grid (q, r)
    hex_positions = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, -1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (1, -1), (-1, 1),
    ]
    
    # This gives 13 positions, we need 12. Let's remove one from the edge.
    # Actually, let's use a different arrangement.
    
    # Better: use a 4-3-3-2 arrangement or similar
    # Let me just use a simple grid and optimize
    
    # Start with a reasonable initial guess
    # Place hexagons in a roughly circular/hexagonal pattern
    
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    radius = 2.5
    positions = [(radius * np.cos(a), radius * np.sin(a), 0.0) for a in angles]
    
    # Add one at center
    # Actually, let's just return a simple arrangement and let the evaluator check
    
    # Simple arrangement: two rows of 6
    positions = []
    for i in range(6):
        positions.append((i * 2.0, 0.0, 0.0))
        positions.append((i * 2.0 + 1.0, sqrt3, 0.0))
    
    # The bounding box is roughly from x=0 to x=11, y=0 to y=sqrt3
    # A hexagon that contains this would need side length around 6-7
    
    # Let's compute a better estimate
    all_points = []
    for x, y, a in positions:
        verts = hex_vertices(x, y, 1.0, a)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    cx = np.mean(all_points[:, 0])
    cy = np.mean(all_points[:, 1])
    
    # Find minimum enclosing hexagon
    min_side = float('inf')
    for outer_angle in np.linspace(0, 60, 13):
        lo, hi = 0.0, 100.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if all(is_point_in_outer_hexagon(v[0], v[1], cx, cy, mid, outer_angle) for v in all_points):
                hi = mid
            else:
                lo = mid
        min_side = min(min_side, hi)
    
    inner = [[p[0], p[1], p[2]] for p in positions]
    return inner, (cx, cy), min_side, 0.0

# EVOLVE-BLOCK-END
