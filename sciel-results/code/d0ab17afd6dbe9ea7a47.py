"""Improved packing of unit hexagons in a regular hexagon.

Strategy:
- Place unit hexagons in a hexagonal lattice (side length 1).
- The optimal arrangement for N hexagons in a hexagonal container is a
  "hex number" configuration: rows of lengths 1,2,...,k,k,...,2,1 where
  N = 3k(k-1)+1.
- For arbitrary N, we use the smallest hex number >= N and place that many
  hexagons, then find the minimal enclosing regular hexagon.
- We optimize the outer hexagon's rotation angle and center to minimize the
  side length.
- Use scipy minimize for the rotation/center optimization.

Key insight: For a hexagonal lattice of unit hexagons, the convex hull is
itself a regular hexagon (for symmetric configurations). The minimal enclosing
regular hexagon can be found by optimizing its orientation.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, and rotation."""
    angle = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top or pointy-top)
    # For a regular hexagon with side length s, the distance from center to vertex is s.
    # Vertices: center + s * (cos(theta), sin(theta)) for theta = 0, 60, ..., 300
    thetas = angle + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([cx + side * np.cos(thetas),
                            cy + side * np.sin(thetas)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside the regular hexagon."""
    # Transform to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    # Rotate point by -angle
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # In local coords, hexagon vertices are at angles 0,60,...,300 with radius = side
    # The hexagon is the intersection of 6 half-planes.
    # For a regular hexagon with circumradius R = side, the inradius (apothem) = side * sqrt(3)/2
    # The 6 sides are at distance apothem from center, at angles 30, 90, 150, 210, 270, 330
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(30 + 60 * i)
        # Normal direction
        nx = np.cos(theta)
        ny = np.sin(theta)
        # Distance from center along normal
        dist = lx * nx + ly * ny
        if dist > apothem:
            return False
    return True

def _hexagon_vertices_all(cx, cy, side, angle_deg):
    """Get all vertices of the outer hexagon."""
    return _hex_vertices(cx, cy, side, angle_deg)

def _unit_hex_vertices(cx, cy, angle_deg):
    """Get vertices of a unit hexagon (side=1) at (cx,cy) with rotation."""
    return _hex_vertices(cx, cy, 1.0, angle_deg)

def _hexagonal_lattice_positions(n):
    """Generate positions for n unit hexagons in a hexagonal lattice arrangement.
    
    Uses axial coordinates. For a hexagonal number H(k) = 3k(k-1)+1,
    we place hexagons in a hexagonal pattern of radius k-1.
    
    For non-hex numbers, we use the smallest hex number >= n and take the first n positions.
    """
    # Find smallest hex number >= n
    k = 1
    while 3 * k * (k - 1) + 1 < n:
        k += 1
    hex_count = 3 * k * (k - 1) + 1
    
    # Generate positions in hexagonal lattice
    # Axial coordinates (q, r) where q + r + s = 0
    # Position in Cartesian: x = 1.5 * q, y = sqrt(3) / 2 * q + sqrt(3) * r
    # For unit hexagons in a hexagonal lattice, the center-to-center distance is sqrt(3)
    
    positions = []
    for q in range(-k + 1, k):
        for r in range(-k + 1, k):
            s = -q - r
            if abs(s) >= k:
                continue
            # Cartesian coordinates
            x = 1.5 * q
            y = np.sqrt(3) / 2 * q + np.sqrt(3) * r
            positions.append((x, y))
    
    # Sort positions to get a nice ordering (by row, then column)
    positions.sort(key=lambda p: (round(p[1], 6), round(p[0], 6)))
    
    # Take first n positions
    return positions[:n]

def _enclosing_hexagon_side(points, angle_deg):
    """Find the minimum side length of a regular hexagon with given rotation
    that encloses all points. The hexagon is centered at the centroid of points
    (or we can optimize center too, but let's start with centroid)."""
    # For a given rotation angle, the minimum enclosing regular hexagon
    # with that rotation can be found by projecting points onto the 6 normal directions
    # and finding the required apothem.
    
    angle = np.radians(angle_deg)
    # The 6 inward normals of the hexagon sides are at angles 30, 90, 150, 210, 270, 330
    # relative to the hexagon's local frame. In global frame, add angle_deg.
    
    # Actually, for a regular hexagon with circumradius R (= side length),
    # the apothem (inradius) is R * sqrt(3)/2.
    # The 6 side normals in local frame are at angles 30, 90, 150, 210, 270, 330 degrees.
    # In global frame, these are at angles (30 + angle_deg), (90 + angle_deg), etc.
    
    # For each normal direction, the maximum projection of points onto that direction
    # gives the required distance from center to that side.
    
    # Let's use the approach: for a hexagon centered at c with rotation theta,
    # the constraint for point p is: for each of the 6 sides,
    # dot(p - c, n_i) <= apothem
    # where n_i are the outward normals.
    
    # To minimize R (side length = circumradius), we need to minimize apothem = R * sqrt(3)/2
    # subject to all points being inside.
    
    # For a fixed center c, the minimum apothem is:
    # max over all 6 normals of max over all points of dot(p - c, n_i)
    
    # We can optimize over center c as well.
    
    # For now, let's compute with center at centroid.
    pts = np.array(points)
    centroid = pts.mean(axis=0)
    
    max_apothem = 0
    for i in range(6):
        theta = np.radians(30 + 60 * i + angle_deg)
        nx = np.cos(theta)
        ny = np.sin(theta)
        projections = pts[:, 0] * nx + pts[:, 1] * ny - (centroid[0] * nx + centroid[1] * ny)
        max_proj = np.max(projections)
        if max_proj > max_apothem:
            max_apothem = max_proj
    
    # Apothem = R * sqrt(3)/2, so R = 2 * apothem / sqrt(3)
    side_length = 2 * max_apothem / np.sqrt(3)
    return side_length, centroid

def _optimize_enclosing_hexagon(points):
    """Optimize the enclosing regular hexagon over rotation and center."""
    pts = np.array(points)
    centroid = pts.mean(axis=0)
    
    def objective(params):
        cx, cy, angle_deg = params
        angle = np.radians(angle_deg)
        max_apothem = 0
        for i in range(6):
            theta = np.radians(30 + 60 * i + angle_deg)
            nx = np.cos(theta)
            ny = np.sin(theta)
            projections = pts[:, 0] * nx + pts[:, 1] * ny - (cx * nx + cy * ny)
            max_proj = np.max(projections)
            if max_proj > max_apothem:
                max_apothem = max_proj
        side_length = 2 * max_apothem / np.sqrt(3)
        return side_length
    
    # Initial guess
    x0 = [centroid[0], centroid[1], 0.0]
    
    # Try multiple starting angles
    best_result = None
    best_val = float('inf')
    
    for angle0 in [0, 30, 60, 90, 15, 45, 75]:
        x0 = [centroid[0], centroid[1], float(angle0)]
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res
    
    if best_result is None:
        # Fallback
        side, cent = _enclosing_hexagon_side(points, 0.0)
        return list(points), (cent[0], cent[1]), side, 0.0
    
    cx, cy, angle_deg = best_result.x
    return list(points), (cx, cy), best_val, angle_deg

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Generate hexagonal lattice positions
    positions = _hexagonal_lattice_positions(N)
    
    # All inner hexagons have 0 rotation (aligned with lattice)
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Optimize enclosing hexagon
    if time.time() - start_time > deadline - 30:
        # Not enough time, use simple estimate
        side, cent = _enclosing_hexagon_side(positions, 0.0)
        return inner, (cent[0], cent[1]), side, 0.0
    
    _, center, side, angle = _optimize_enclosing_hexagon(positions)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Return what we have
        return inner, center, side, angle
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
