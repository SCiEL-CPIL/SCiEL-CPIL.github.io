"""Packing N unit regular hexagons inside a regular hexagon; minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools
import math

N = 11
DEADLINE = 135.0  # safety margin before 150s kill
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon (side length 'side') centered at (cx,cy) with rotation angle_deg."""
    a = np.deg2rad(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3.0
        # regular hexagon vertices at distance 'side' from center
        vx = cx + side * np.cos(theta)
        vy = cy + side * np.sin(theta)
        verts.append([vx, vy])
    return np.array(verts)

def _point_in_polygon(px, py, poly):
    """Ray casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _edge_distance(p1, p2, p3, p4):
    """Minimum distance between segment p1-p2 and segment p3-p4."""
    def dist_point_segment(px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        qx = ax + t * dx
        qy = ay + t * dy
        return math.hypot(px - qx, py - qy)
    
    d1 = dist_point_segment(p1[0], p1[1], p3[0], p3[1], p4[0], p4[1])
    d2 = dist_point_segment(p2[0], p2[1], p3[0], p3[1], p4[0], p4[1])
    d3 = dist_point_segment(p3[0], p3[1], p1[0], p1[1], p2[0], p2[1])
    d4 = dist_point_segment(p4[0], p4[1], p1[0], p1[1], p2[0], p2[1])
    return min(d1, d2, d3, d4)

def _min_dist_between_hexes(verts1, verts2):
    """Minimum distance between two hexagons (convex polygons)."""
    # Check if any vertex of one is inside the other
    for v in verts1:
        if _point_in_polygon(v[0], v[1], verts2):
            return 0.0
    for v in verts2:
        if _point_in_polygon(v[0], v[1], verts1):
            return 0.0
    # Check all edge pairs
    min_d = float('inf')
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        p1 = verts1[i]
        p2 = verts1[(i + 1) % n1]
        for j in range(n2):
            p3 = verts2[j]
            p4 = verts2[(j + 1) % n2]
            d = _edge_distance(p1, p2, p3, p4)
            if d < min_d:
                min_d = d
    return min_d

def _min_dist_to_boundary(verts_inner, outer_verts):
    """Minimum distance from inner hexagon vertices/edges to outer hexagon boundary."""
    # For each vertex of inner, find min distance to outer boundary
    min_d = float('inf')
    n_out = len(outer_verts)
    for v in verts_inner:
        # Check if vertex is inside outer
        if not _point_in_polygon(v[0], v[1], outer_verts):
            # Outside: distance to boundary
            for i in range(n_out):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_out]
                d = _edge_distance([v[0], v[1]], [v[0], v[0]+0.001], p1, p2)
                if d < min_d:
                    min_d = d
            continue
        # Inside: min distance to any outer edge
        for i in range(n_out):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % n_out]
            d = _edge_distance([v[0], v[1]], [v[0], v[0]+0.001], p1, p2)
            if d < min_d:
                min_d = d
    return min_d

def _min_clearance(centers, angles, outer_center, outer_side, outer_angle):
    """Compute minimum clearance: min of (dist between inner hexes)/2 and dist to boundary."""
    inner_verts_list = []
    for k in range(N):
        v = _hex_vertices(centers[k, 0], centers[k, 1], 1.0, angles[k])
        inner_verts_list.append(v)
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    min_clear = float('inf')
    
    # Boundary clearance
    for k in range(N):
        d = _min_dist_to_boundary(inner_verts_list[k], outer_verts)
        if d < min_clear:
            min_clear = d
    
    # Pairwise clearance
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_between_hexes(inner_verts_list[i], inner_verts_list[j])
            if d < min_clear:
                min_clear = d
    
    return min_clear

def _is_valid(centers, angles, outer_center, outer_side, outer_angle, tol=0.001):
    """Check if all inner hexagons are inside outer and non-overlapping."""
    return _min_clearance(centers, angles, outer_center, outer_side, outer_angle) >= -tol

def _pack_hex_lattice():
    """
    Place N unit hexagons on a hexagonal lattice (no rotation) and find the minimal enclosing hexagon.
    Uses the known optimal lattice packing for hexagons.
    """
    # Hexagonal lattice positions for N hexagons
    # For N hexagons, arrange in hexagonal cluster
    # The hexagonal lattice has basis vectors: a1 = (2, 0), a2 = (1, sqrt(3))
    # Actually for side-1 hexagons, the lattice spacing is 2 in one direction and 2*sqrt(3)/2*... 
    # Let me think: two adjacent unit hexagons (flat-top) have centers 2 apart horizontally.
    # For pointy-top: centers are 2*sqrt(3)/2 * ... 
    # 
    # Standard hex lattice for unit hexagons (side=1):
    # The distance between centers of adjacent hexagons = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) for one direction
    # Actually no. For regular hexagons of side 1, the center-to-center distance for edge-sharing is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3).
    # Wait: apothem of unit hexagon = sqrt(3)/2. So center-to-center = 2 * sqrt(3)/2 = sqrt(3).
    # Hmm, that's for flat-to-flat contact. The distance between centers = 2 * inradius = sqrt(3).
    
    # Let me use the standard hex lattice:
    # a1 = (2, 0)  -- this is for hexagons with flat sides horizontal, where adjacent centers are 2 apart? 
    # No. Let me be careful.
    
    # A regular hexagon with side length s has:
    # - circumradius = s
    # - inradius (apothem) = s * sqrt(3)/2
    # 
    # Two unit hexagons sharing an edge: center-to-center distance = 2 * inradius = sqrt(3).
    # 
    # Hex lattice basis for unit hexagons:
    # If we orient hexagons with a flat side at top:
    #   a1 = (sqrt(3), 0)  -- horizontal neighbor
    #   a2 = (sqrt(3)/2, 3/2)  -- diagonal neighbor
    # Wait, let me reconsider.
    
    # Actually, the simplest: use axial coordinates on a hex grid.
    # For pointy-top hexagons of size 1 (circumradius = 1):
    #   pixel(x, y) = (x * sqrt(3), y * 3/2 + (x & 1) * 3/4)  -- this is for hex size 1
    # But that's for a specific orientation.
    
    # Let me just use: centers at positions that form a hexagonal lattice with spacing sqrt(3).
    # Basis: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2.0, 1.5])
    
    # Generate hexagonal lattice points in axial coordinates
    # For a hexagonal cluster of N points, we need to find the right arrangement.
    # For N=11: one center + 6 in first ring + 4 in second ring (or some arrangement)
    # Actually, hexagonal numbers: 1, 7, 19, ... So 11 is not a perfect hexagonal number.
    # 11 = 7 + 4, so we take the 7-point cluster (center + ring 1) plus 4 from ring 2.
    
    # Generate all lattice points up to ring 2
    lattice_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Axial to pixel
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            # Hex distance from origin
            dist = max(abs(q), abs(r), abs(-q - r))
            if dist <= 2:
                lattice_points.append((x, y, dist))
    
    # Sort by distance from origin
    lattice_points.sort(key=lambda p: p[2])
    
    # Take the first N points
    selected = lattice_points[:N]
    centers = np.array([[p[0], p[1]] for p in selected])
    angles = np.zeros(N)
    
    return centers, angles

def _minimize_outer_hex(centers, angles, max_side=10.0, max_iter=500):
    """
    Given fixed inner hexagon positions and angles, find the minimal enclosing regular hexagon.
    We optimize over outer center (2 vars), outer side (1 var), outer angle (1 var).
    """
    def objective(params):
        ox, oy, side, oangle = params
        if side <= 0:
            return 1e6
        clearance = _min_clearance(centers, angles, (ox, oy), side, oangle)
        # We want all clearances >= 0, minimize side
        # Penalty if any clearance < 0
        if clearance < 0:
            return side + 100.0 * (-clearance) ** 2
        return side
    
    # Initial guess: center at centroid of inner hexes
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    # Estimate side from extent
    extent = np.max(np.sqrt((centers[:, 0] - cx)**2 + (centers[:, 1] - cy)**2)) + 2.0
    x0 = [cx, cy, extent, 0.0]
    
    best_side = max_side
    best_params = x0
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 0.001, 'fatol': 0.001})
    
    if result.fun < best_side:
        best_side = result.fun
        best_params = result.x
    
    # Try different initial angles
    for oangle in [0, 30, 60, 90, 120, 150]:
        x0 = [cx, cy, extent, float(oangle)]
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter // 2, 'xatol': 0.001, 'fatol': 0.001})
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    
    return best_params, best_side

def construct():
    global _start_time
    _start_time = time.time()
    
    # Strategy 1: Hexagonal lattice packing, optimize enclosing hexagon
    centers, angles = _pack_hex_lattice()
    outer_params, side1 = _minimize_outer_hex(centers, angles)
    
    if _elapsed() > DEADLINE - 10:
        inner = [[float(centers[k, 0]), float(centers[k, 1]), float(angles[k])] for k in range(N)]
        return inner, (float(outer_params[0]), float(outer_params[1])), float(outer_params[2]), float(outer_params[3])
    
    # Strategy 2: Force-directed relaxation to optimize inner positions
    # Start from lattice, then relax
    centers_fd = centers.copy()
    angles_fd = angles.copy()
    
    def fd_objective(params):
        # params: 3N variables: x0,y0,a0, x1,y1,a1, ...
        c = params[:2*N].reshape(N, 2)
        a = params[2*N:]
        # Repulsion: push apart if too close
        # Attraction: pull toward center
        force = np.zeros_like(c)
        for i in range(N):
            for j in range(i + 1, N):
                d = c[i] - c[j]
                dist = np.linalg.norm(d)
                if dist < 0.01:
                    d = np.array([0.1, 0.1])
                    dist = 0.14
                if dist < 2.5:  # interaction range
                    f = 10.0 / (dist ** 2)
                    force[i] += f * d / dist
                    force[j] -= f * d / dist
            # Pull toward origin
            force[i] -= 0.5 * c[i]
        return force
    
    # Simple iterative relaxation
    for it in range(200):
        if _elapsed() > DEADLINE - 15:
            break
        force = np.zeros_like(centers_fd)
        for i in range(N):
            for j in range(i + 1, N):
                d = centers_fd[i] - centers_fd[j]
                dist = np.linalg.norm(d)
                if dist < 0.01:
                    d = np.array([0.1, 0.1])
                    dist = 0.14
                if dist < 3.0:
                    f = 5.0 / (dist ** 3)
                    force[i] += f * d
                    force[j] -= f * d
            force[i] -= 0.3 * centers_fd[i]
        centers_fd += 0.01 * force
    
    # Now optimize enclosing hexagon for the relaxed configuration
    outer_params2, side2 = _minimize_outer_hex(centers_fd, angles_fd)
    
    if _elapsed() > DEADLINE - 5:
        # Pick the better of the two
        if side2 < side1:
            inner = [[float(centers_fd[k, 0]), float(centers_fd[k, 1]), float(angles_fd[k])] for k in range(N)]
            return inner, (float(outer_params2[0]), float(outer_params2[1])), float(outer_params2[2]), float(outer_params2[3])
        else:
            inner = [[float(centers[k, 0]), float(centers[k, 1]), float(angles[k])] for k in range(N)]
            return inner, (float(outer_params[0]), float(outer_params[1])), float(outer_params[2]), float(outer_params[3])
    
    # Strategy 3: Try rotating individual hexagons to improve packing
    # For each hexagon, try a few rotation angles
    best_side = min(side1, side2)
    best_centers = centers.copy() if side1 <= side2 else centers_fd.copy()
    best_angles = angles.copy() if side1 <= side2 else angles_fd.copy()
    best_outer = outer_params.copy() if side1 <= side2 else outer_params2.copy()
    
    # Try small perturbations of angles
    for trial in range(10):
        if _elapsed() > DEADLINE - 5:
            break
        trial_angles = best_angles + np.random.uniform(-15, 15, N) * 0.1
        trial_centers = best_centers + np.random.uniform(-0.1, 0.1, (N, 2))
        op, s = _minimize_outer_hex(trial_centers, trial_angles, max_iter=200)
        if s < best_side:
            best_side = s
            best_centers = trial_centers
            best_angles = trial_angles
            best_outer = op
    
    inner = [[float(best_centers[k, 0]), float(best_centers[k, 1]), float(best_angles[k])] for k in range(N)]
    return inner, (float(best_outer[0]), float(best_outer[1])), float(best_outer[2]), float(best_outer[3])

