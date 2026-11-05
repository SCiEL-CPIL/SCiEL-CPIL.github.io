"""Packing 11 unit regular hexagons inside a minimal regular hexagon.

Approach:
- Build a rigid cluster of 11 unit hexagons in a compact arrangement.
- Compute the convex hull of the cluster (vertices of all hexagons).
- For a given global rotation angle, compute the minimal enclosing regular hexagon
  side length (with the same rotation as the cluster's reference frame, i.e., the
  outer hexagon can also rotate independently).
- Actually, the outer hexagon has its own rotation. The minimal enclosing regular
  hexagon of a convex polygon, where the outer hexagon can rotate freely, is:
  For each orientation of the outer hexagon, the side length needed is max over
  the 6 "width" directions (normals to the 6 sides) of the support function.
  The width in direction theta is max_{p in hull} p·u(theta) - min_{p in hull} p·u(theta),
  where u(theta) = (cos(theta), sin(theta)).
  For a regular hexagon with side length s and rotation phi, the 6 side normals are
  at angles phi + k*pi/3 for k=0..5. The side length s must satisfy:
  s >= width(phi + k*pi/3) / sqrt(3) for all k, because the distance between opposite
  parallel sides of a regular hexagon with side s is s*sqrt(3).
  
  So for a given outer rotation phi, the required side length is:
  s(phi) = max_{k=0..5} width(phi + k*pi/3) / sqrt(3)
  
  We minimize s(phi) over phi in [0, pi/3) by ternary search (or golden-section).
  Since s(phi) is piecewise linear and periodic with period pi/3, we can find the
  minimum precisely.
"""

import numpy as np
from scipy.spatial import ConvexHull

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side length, and rotation in degrees."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _build_cluster():
    """Build 11 unit hexagons in a compact 3-4-3 arrangement (rows of 3, 4, 3).
    
    Unit hexagons with flat-top orientation (angle=0) have:
    - width (horizontal) = 2 * side = 2
    - height (vertical) = sqrt(3) * side = sqrt(3)
    
    For flat-top hexagons, adjacent hexagons in the same row are separated by 2 in x.
    Hexagons in adjacent rows are offset by 1 in x and sqrt(3) in y.
    
    Row 0 (top): 3 hexagons
    Row 1 (middle): 4 hexagons
    Row 2 (bottom): 3 hexagons
    """
    hexagons = []
    # Row 1 (middle, y=0): 4 hexagons centered at x = -1.5, -0.5, 0.5, 1.5
    for i in range(4):
        cx = (i - 1.5) * 1.0  # spacing 1? No, for flat-top, horizontal spacing is 2*side/sin(60)... 
        # Actually for flat-top hexagons with side=1:
        # The horizontal distance between centers of adjacent hexagons in the same row is 2*side*cos(30°)? 
        # Let me reconsider. A regular hexagon with side s and flat-top has vertices at angles 0, 60, 120, 180, 240, 300.
        # The width (left to right) is 2s. The height (top to bottom) is s*sqrt(3).
        # Two flat-top hexagons side by side: their centers are 2s apart horizontally? No.
        # Actually, for flat-top, the horizontal extent is from -s to s, so width 2s.
        # If two hexagons are adjacent horizontally, the distance between centers is 2s * cos(0) ... 
        # Let me just use pointy-top or compute properly.
        # 
        # For a regular hexagon with side s, the distance from center to vertex is s.
        # Flat-top: vertices at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2).
        # Horizontal width: 2s. Vertical height: s*sqrt(3).
        # Adjacent hexagons in the same row (flat-top): centers separated by 2s * cos(30°)? 
        # No. If you place two flat-top hexagons next to each other sharing a vertical edge, the centers are 2*(s*sqrt(3)/2) = s*sqrt(3) apart horizontally? 
        # Wait, the rightmost vertex of one is at x=s, leftmost of next at x=-s+dx. For them to touch, s = -s + dx, so dx = 2s.
        # But that's not right either because the hexagons would overlap.
        # 
        # Actually, for flat-top hexagons, the horizontal spacing between centers of adjacent hexagons in the same row is 2*s*cos(30°) = s*sqrt(3).
        # No wait. Let me just compute: the right edge of a flat-top hexagon is not a vertical line. The vertices on the right are at (s, 0) and (s/2, ±s*sqrt(3)/2).
        # The rightmost point is (s, 0). The leftmost point of the next hexagon should be at (s, 0) if they touch at that vertex.
        # So center of next hex is at (s + s, 0) = (2s, 0)? That gives spacing 2s.
        # But then they only touch at one point. For a proper packing, adjacent hexagons in a hexagonal grid share an edge.
        # 
        # For a hexagonal grid of flat-top hexagons:
        # - Horizontal spacing: s * sqrt(3)
        # - Vertical spacing: 1.5 * s
        # - Odd rows offset by s * sqrt(3) / 2
        # 
        # Let me verify: with side s=1, horizontal spacing = sqrt(3) ≈ 1.732, vertical spacing = 1.5.
        # 
        # For pointy-top hexagons:
        # - Horizontal spacing: 1.5 * s
        # - Vertical spacing: s * sqrt(3)
        # - Odd rows offset by 1.5 * s / 2 = 0.75 * s
        #
        # I'll use pointy-top for cleaner math. Pointy-top hexagon with side s:
        # Vertices at angles 30, 90, 150, 210, 270, 330 degrees.
        # Width (horizontal) = s * sqrt(3)
        # Height (vertical) = 2s
        # 
        # Hexagonal grid spacing for pointy-top:
        # - Horizontal: 1.5 * s
        # - Vertical: s * sqrt(3)
        # - Offset for odd rows: 0.75 * s

        pass
    
    # Let me restart with a cleaner approach.
    # Use pointy-top hexagons with side=1.
    # Hexagonal grid:
    # - dx = 1.5 * side = 1.5
    # - dy = sqrt(3) * side = sqrt(3)
    # - For row r, x-offset = (r % 2) * 0.75 * side = (r % 2) * 0.75
    
    side = 1.0
    dx = 1.5 * side
    dy = np.sqrt(3) * side
    offset = 0.75 * side
    
    # 3-4-3 arrangement:
    # Row 0 (top): 3 hexagons
    # Row 1 (middle): 4 hexagons  
    # Row 2 (bottom): 3 hexagons
    
    hex_centers = []
    # Row 0: 3 hexagons, centered
    for i in range(3):
        cx = (i - 1) * dx + offset  # offset for row 0 (even row? let's say row 0 is offset)
        cy = 0
        hex_centers.append((cx, cy))
    
    # Row 1: 4 hexagons
    for i in range(4):
        cx = (i - 1.5) * dx  # no offset for odd row
        cy = dy
        hex_centers.append((cx, cy))
    
    # Row 2: 3 hexagons
    for i in range(3):
        cx = (i - 1) * dx + offset
        cy = 2 * dy
        hex_centers.append((cx, cy))
    
    # Verify we have 11
    assert len(hex_centers) == 11, f"Expected 11, got {len(hex_centers)}"
    
    # Build all vertices
    all_vertices = []
    for cx, cy in hex_centers:
        verts = _hex_vertices(cx, cy, side, 90.0)  # pointy-top: rotate by 90 degrees from flat-top
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    return all_vertices, hex_centers

def _width_of_convex_hull(hull, theta):
    """Compute the width of a convex hull in direction theta."""
    u = np.array([np.cos(theta), np.sin(theta)])
    projections = hull @ u
    return np.max(projections) - np.min(projections)

def _min_enclosing_hexagon_side(hull, phi):
    """Compute the minimal side length of a regular hexagon with rotation phi that encloses the convex hull."""
    sqrt3 = np.sqrt(3)
    max_width = 0.0
    for k in range(6):
        theta = phi + k * np.pi / 3
        w = _width_of_convex_hull(hull, theta)
        required_side = w / sqrt3
        if required_side > max_width:
            max_side = required_side
    return max_side

def construct():
    # Build the cluster
    all_vertices, hex_centers = _build_cluster()
    
    # Compute convex hull
    hull_obj = ConvexHull(all_vertices)
    hull = all_vertices[hull_obj.vertices]
    
    # Minimize the enclosing hexagon side length over phi in [0, pi/3)
    # Use ternary search for high precision
    a = 0.0
    b = np.pi / 3
    tol = 1e-12
    max_iter = 200
    
    for _ in range(max_iter):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        f1 = _min_enclosing_hexagon_side(hull, m1)
        f2 = _min_enclosing_hexagon_side(hull, m2)
        if f1 < f2:
            b = m2
        else:
            a = m1
        if b - a < tol:
            break
    
    phi_opt = (a + b) / 2
    min_side = _min_enclosing_hexagon_side(hull, phi_opt)
    
    # Now we need to position the cluster so that the enclosing hexagon is centered at origin
    # The enclosing hexagon has center at some point. We need to find the center of the minimal enclosing hexagon.
    # For a given phi, the enclosing hexagon is determined by the 6 support lines.
    # The center of the hexagon is the intersection of the midlines.
    # 
    # For direction theta, the support line is at distance d_max(theta) from origin in direction u(theta).
    # The opposite support line is at distance d_min(theta).
    # The midline is at (d_max + d_min) / 2 in direction u(theta).
    # 
    # For a regular hexagon, the center is the same for all 6 pairs of opposite sides.
    # So we can compute the center as the average of the midpoints of opposite support lines.
    
    # Compute the center of the enclosing hexagon
    center_x = 0.0
    center_y = 0.0
    for k in range(3):  # Only need 3 pairs of opposite sides
        theta = phi_opt + k * np.pi / 3
        u = np.array([np.cos(theta), np.sin(theta)])
        projections = hull @ u
        d_max = np.max(projections)
        d_min = np.min(projections)
        mid = (d_max + d_min) / 2
        center_x += mid * u[0]
        center_y += mid * u[1]
    center_x /= 3
    center_y /= 3
    
    # The cluster needs to be translated so that the enclosing hexagon is centered at origin.
    # So we subtract (center_x, center_y) from all hexagon centers.
    # Also, the cluster is rotated by phi_opt relative to its original orientation.
    # But wait, the hexagons were built with a fixed orientation (pointy-top, 90 degrees).
    # When we rotate the cluster by phi_opt, each hexagon rotates by phi_opt.
    # 
    # Actually, in our formulation, the hull was computed in the original coordinate system.
    # The enclosing hexagon has rotation phi_opt in the same coordinate system.
    # So the cluster does NOT need to be rotated; it stays in its original orientation.
    # The outer hexagon has rotation phi_opt.
    # We just need to translate the cluster so the enclosing hexagon is centered at origin.
    
    inner = []
    for cx, cy in hex_centers:
        new_cx = cx - center_x
        new_cy = cy - center_y
        inner.append([new_cx, new_cy, 90.0])  # Each hexagon has pointy-top orientation (90 degrees)
    
    outer_center = (0.0, 0.0)
    outer_side = min_side
    outer_angle_deg = np.degrees(phi_opt)
    
    return inner, outer_center, outer_side, outer_angle_deg
