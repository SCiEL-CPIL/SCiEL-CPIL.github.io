"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import itertools

N = 12

# EVOLVE-BLOCK-START

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side length, and rotation in degrees."""
    theta = np.radians(angle_deg)
    angles = theta + np.pi / 3 * np.arange(6)
    # Regular hexagon vertices are at distance = side from center
    # Wait: for a regular hexagon, the circumradius equals the side length.
    # So vertices are at distance 'side' from center.
    vx = cx + side * np.cos(angles)
    vy = cy + side * np.sin(angles)
    return list(zip(vx, vy))

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, angle_deg."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    theta = -np.radians(angle_deg)
    # Rotate point by -angle
    lx = dx * np.cos(theta) - dy * np.sin(theta)
    ly = dx * np.sin(theta) + dy * np.cos(theta)
    
    # In local coords, hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The hexagon can be defined by 6 half-planes.
    # For a hexagon with circumradius = side, the apothem = side * sqrt(3)/2
    # The 6 sides are at distances apothem from center, with normals at angles 0, 60, 120, 180, 240, 300? 
    # Actually, the normals to the sides point at angles 30, 90, 150, 210, 270, 330 (if vertices at 0,60,...)
    # Let's use the standard: vertices at angles k*60 degrees.
    # Side between vertex k and k+1. The normal points outward.
    # For vertex at angle 0 (side,0) and 60 (side/2, side*sqrt(3)/2):
    # Midpoint: (3*side/4, side*sqrt(3)/4)
    # Normal direction: perpendicular to edge. Edge direction: (side/2 - side, side*sqrt(3)/2 - 0) = (-side/2, side*sqrt(3)/2)
    # Normal: (side*sqrt(3)/2, side/2) -> angle 30 degrees.
    # So normals are at 30 + k*60 degrees for k=0..5.
    # The inequality is: dot(normal, point) <= apothem for all 6 normals.
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        alpha = np.radians(30 + 60 * k)
        nx = np.cos(alpha)
        ny = np.sin(alpha)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _hexagon_in_hexagon(inner_c, inner_s, inner_a, outer_c, outer_s, outer_a):
    """Check if inner hexagon is fully inside outer hexagon."""
    # It's sufficient to check that all 6 vertices of the inner hexagon are inside the outer hexagon
    # because both are convex.
    inner_verts = _hexagon_vertices(inner_c[0], inner_c[1], inner_s, inner_a)
    for vx, vy in inner_verts:
        if not _point_in_hexagon(vx, vy, outer_c[0], outer_c[1], outer_s, outer_a):
            return False
    return True

def _min_distance_to_boundary(cx, cy, side, angle_deg, outer_c, outer_s, outer_a):
    """Calculate the minimum distance from the hexagon (center, side, angle) to the outer hexagon boundary.
    This is the minimum over all points on the inner hexagon of the distance to the outer boundary.
    For convex polygons, the minimum distance from a point to the polygon boundary is the minimum distance to any edge.
    But we want the minimum distance from the inner hexagon to the outer hexagon boundary.
    This is equivalent to: min over all vertices of inner hexagon of (distance to outer boundary).
    Actually, it's the minimum over all points in inner hexagon of distance to outer boundary.
    Since distance to a convex set's boundary is a convex function? No.
    However, the minimum distance from a convex polygon P to the boundary of convex polygon Q (where P is inside Q)
    is achieved at a vertex of P or an edge of P.
    Actually, it's simpler: the minimum distance from inner hexagon to outer boundary is the minimum over the 6 vertices
    of the inner hexagon of the distance from that vertex to the outer hexagon boundary.
    Wait, is that true? Consider a long thin rectangle inside a circle. The min distance might be at the middle of a long side.
    But for regular hexagons, due to symmetry, it's likely at vertices or edge midpoints.
    Let's compute the distance from each vertex of the inner hexagon to the outer hexagon boundary,
    and also from each edge midpoint of the inner hexagon to the outer hexagon boundary.
    Actually, the distance from a point to a convex polygon is the minimum distance to any edge of the polygon.
    So for each vertex of inner hex, compute min distance to any edge of outer hex.
    Then take the minimum over all inner hex vertices.
    """
    inner_verts = _hexagon_vertices(cx, cy, side, angle_deg)
    # Outer hex edges
    outer_verts = _hexagon_vertices(outer_c[0], outer_c[1], outer_s, outer_a)
    min_dist = float('inf')
    for vx, vy in inner_verts:
        # Distance from point (vx,vy) to each edge of outer hex
        for i in range(6):
            x1, y1 = outer_verts[i]
            x2, y2 = outer_verts[(i + 1) % 6]
            # Distance from point to line segment
            dx = x2 - x1
            dy = y2 - y1
            len_sq = dx * dx + dy * dy
            if len_sq == 0:
                dist = np.sqrt((vx - x1) ** 2 + (vy - y1) ** 2)
            else:
                t = max(0, min(1, ((vx - x1) * dx + (vy - y1) * dy) / len_sq))
                px = x1 + t * dx
                py = y1 + t * dy
                dist = np.sqrt((vx - px) ** 2 + (vy - py) ** 2)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def construct():
    """
    Place 12 unit hexagons inside a regular hexagon, minimizing the outer side length.
    We use a heuristic approach with local optimization.
    """
    # Strategy:
    # 1. Start with a reasonable initial configuration.
    # 2. Use a simple local search to minimize the outer side length.
    
    # Initial guess: place hexagons in a hexagonal pattern.
    # For N=12, we can use a central hexagon + 6 around it + 5 more? 
    # Or 3 rows: 4, 4, 4? Or 2, 4, 4, 2?
    # Let's try a 3x4 grid-like arrangement in hex coordinates.
    
    # Better: use a known packing. For 12 hexagons, a compact arrangement is:
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons (offset)
    # Row 3: 4 hexagons
    # But in hex lattice, rows alternate.
    
    # Let's use hexagonal coordinates.
    # Center at (0,0).
    # Positions in hex lattice (axial coordinates q, r):
    # For a hexagon of side length s, the distance between centers of adjacent hexagons is 2*s*sqrt(3)/2 = s*sqrt(3).
    # Wait, for unit hexagons (side=1), the distance between centers of adjacent hexagons in a hex lattice is sqrt(3).
    
    # Let's define positions in Cartesian coordinates using a hex lattice with spacing sqrt(3).
    # Hex lattice basis:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 3/2)
    
    # Generate 12 positions in a compact hexagonal cluster.
    # A hexagonal number H_k = 3k(k-1)+1. H_2=7, H_3=19. So 12 is between 7 and 19.
    # Let's take the central hexagon (1), the 6 neighbors (total 7), and 5 more from the next ring.
    
    sqrt3 = np.sqrt(3)
    
    # Central hexagon at (0,0)
    positions = [(0.0, 0.0)]
    
    # First ring: 6 hexagons at distance sqrt3 from center
    for k in range(6):
        angle = np.pi / 3 * k
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions.append((x, y))
    
    # Second ring: we need 5 more. The second ring has 12 positions.
    # Let's pick 5 from the second ring.
    # Second ring positions are at distance 2*sqrt3 from center, or at the corners of the hexagonal cluster of radius 2.
    # Actually, the second ring consists of 12 hexagons. Their centers are at:
    # For k in 0..5:
    #   (2*sqrt3*cos(angle), 2*sqrt3*sin(angle)) for the 6 "corner" positions
    #   and intermediate positions.
    # Let's just generate all 12 second-ring positions and pick 5.
    
    second_ring = []
    for k in range(6):
        angle = np.pi / 3 * k
        # Corner positions
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        second_ring.append((x, y))
        # Intermediate positions (2 per side)
        # Actually, between two adjacent corners of the second ring, there are 2 positions.
        # Let's compute them properly.
        # The second ring in hex coordinates:
        # (q,r) with max(|q|,|r|,|q+r|) = 2
        # Let's list them:
        # (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1), (2,-2)? No.
        # Let's use the standard hex grid.
    
    # Simpler: let's use a different initial configuration.
    # Place 12 hexagons in a 3x4 rectangle-like pattern.
    # Row spacing: 1.5 (vertical distance between rows in hex packing)
    # Column spacing: sqrt3 (horizontal distance between columns)
    # For a 3-row, 4-column pattern:
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3
            y = row * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            positions.append((x, y))
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(x - cx, y - cy) for x, y in positions]
    
    # Initial angles: all 0
    angles = [0.0] * N
    
    # Initial outer side length: large enough to contain all
    # Estimate: the configuration spans about 3*sqrt3 in x and 2*1.5 in y.
    # The outer hexagon needs to contain this.
    # Let's start with L = 8.0
    L = 8.0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Local optimization: try to reduce L by moving hexagons and adjusting L.
    # We'll use a simple iterative approach:
    # 1. For a given L, check if all hexagons fit.
    # 2. If not, increase L.
    # 3. If yes, try to decrease L and see if it still fits.
    # 4. Also try small perturbations of positions and angles.
    
    # Actually, let's do a more systematic search.
    # We'll try several initial configurations and local optimizations.
    
    best_L = float('inf')
    best_config = None
    
    # Try different initial configurations
    configs = []
    
    # Config 1: 3x4 grid as above
    configs.append((positions, angles))
    
    # Config 2: hexagonal cluster (center + 6 + 5)
    positions2 = [(0.0, 0.0)]
    for k in range(6):
        angle = np.pi / 3 * k
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions2.append((x, y))
    # Add 5 more from second ring
    for k in range(5):
        angle = np.pi / 3 * k + np.pi / 6  # Offset by 30 degrees
        x = 2 * sqrt3 * np.cos(angle)
        y = 2 * sqrt3 * np.sin(angle)
        positions2.append((x, y))
    xs2 = [p[0] for p in positions2]
    ys2 = [p[1] for p in positions2]
    cx2 = (min(xs2) + max(xs2)) / 2
    cy2 = (min(ys2) + max(ys2)) / 2
    positions2 = [(x - cx2, y - cy2) for x, y in positions2]
    configs.append((positions2, [0.0] * N))
    
    # Config 3: 2 rows of 6
    positions3 = []
    for row in range(2):
        for col in range(6):
            x = col * sqrt3
            y = row * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            positions3.append((x, y))
    xs3 = [p[0] for p in positions3]
    ys3 = [p[1] for p in positions3]
    cx3 = (min(xs3) + max(xs3)) / 2
    cy3 = (min(ys3) + max(ys3)) / 2
    positions3 = [(x - cx3, y - cy3) for x, y in positions3]
    configs.append((positions3, [0.0] * N))
    
    for pos, ang in configs:
        # Local optimization
        L_current = 8.0
        for iteration in range(100):
            # Check if current configuration fits in outer hexagon of side L_current
            fits = True
            for i in range(N):
                if not _hexagon_in_hexagon((pos[i][0], pos[i][1]), 1.0, ang[i], 
                                           outer_center, L_current, outer_angle):
                    fits = False
                    break
            
            if fits:
                # Try to reduce L
                L_new = L_current * 0.98
                if L_new < 1.0:
                    L_new = 1.0
                # Check if it still fits
                fits_new = True
                for i in range(N):
                    if not _hexagon_in_hexagon((pos[i][0], pos[i][1]), 1.0, ang[i], 
                                               outer_center, L_new, outer_angle):
                        fits_new = False
                        break
                if fits_new:
                    L_current = L_new
                else:
                    # Try to improve by moving hexagons
                    # Random perturbation
                    import random
                    idx = random.randint(0, N - 1)
                    dx = random.uniform(-0.1, 0.1)
                    dy = random.uniform(-0.1, 0.1)
                    da = random.uniform(-10, 10)
                    pos[idx] = (pos[idx][0] + dx, pos[idx][1] + dy)
                    ang[idx] = (ang[idx] + da) % 360
            else:
                # Increase L
                L_current *= 1.02
        
        # Evaluate final L
        if L_current < best_L:
            best_L = L_current
            best_config = (pos, ang)
    
    # Final configuration
    pos, ang = best_config
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_L, 0.0

# EVOLVE-BLOCK-END
