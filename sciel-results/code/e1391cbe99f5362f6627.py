"""Packing 12 unit hexagons in smallest outer hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = np.radians(angle_deg)
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300
    angles = angle + np.radians(np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def polygon_area(vertices):
    """Shoelace formula for polygon area."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = -np.radians(angle_deg)
    lx = dx * np.cos(angle) - dy * np.sin(angle)
    ly = dx * np.sin(angle) + dy * np.cos(angle)
    # For a flat-top hexagon with side s, the vertices are at distance s from center
    # The hexagon contains points where max(|lx|/s, |ly|/(s*sqrt(3)/2)) <= 1 approximately
    # More precisely: the hexagon is the intersection of 6 half-planes
    # For flat-top: vertices at angles 0,60,120,180,240,300
    # The half-planes are: |lx| <= s, |ly| <= s*sqrt(3)/2, |lx + ly/sqrt(3)| <= s, etc.
    # Actually for a regular hexagon with circumradius = side:
    # The inradius = side * sqrt(3)/2
    # Conditions: |lx| <= side, |ly| <= side*sqrt(3)/2, 
    # |lx/2 + ly*sqrt(3)/2| <= side, |-lx/2 + ly*sqrt(3)/2| <= side
    # Wait, let me use the proper formulation.
    # A regular hexagon with side length s has vertices at distance s from center.
    # The six half-planes can be defined by the normals to the edges.
    # For flat-top orientation, edge normals are at angles 30, 90, 150, 210, 270, 330
    # The inradius (distance from center to edge) = s * sqrt(3)/2
    r_in = side * np.sqrt(3) / 2.0
    normals = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    for i in range(6):
        nx = np.cos(normals[i])
        ny = np.sin(normals[i])
        if lx * nx + ly * ny > r_in:
            return False
    return True

def hexagons_disjoint(v1, v2, v3, v4, v5, v6, angle1, angle2, angle3, angle4, angle5, angle6,
                       cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, cx5, cy5, cx6, cy6):
    """Check if 6 hexagons are pairwise disjoint (no interior overlap)."""
    # For simplicity, check if centers are far enough apart
    # Two unit hexagons are disjoint if distance between centers > 2 * inradius = sqrt(3)
    # when aligned, but this is a necessary not sufficient condition for arbitrary rotations
    # For now, use center distance as a quick check
    centers = [(cx1, cy1), (cx2, cy2), (cx3, cy3), (cx4, cy4), (cx5, cy5), (cx6, cy6)]
    for i in range(6):
        for j in range(i+1, 6):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < np.sqrt(3) * 0.99:  # Small tolerance
                return False
    return True

def enclosing_hexagon_side(hex_centers, outer_center, outer_angle_deg):
    """Compute the minimum side length of a regular hexagon (with given center and angle)
    that contains all hexagon centers expanded by the hexagon's extent."""
    # For each inner hexagon, we need to ensure all its vertices are inside the outer hexagon
    # This is complex. Instead, let's use a simpler approach:
    # The outer hexagon must contain all vertices of all inner hexagons.
    # For a given outer hexagon (center, side, angle), a point is inside if it satisfies
    # the 6 half-plane constraints.
    # We want to find the minimum side such that all points are inside.
    
    # For a regular hexagon with center (ox, oy), side s, and rotation angle:
    # A point (px, py) is inside if for all 6 edge normals n_i:
    #   (px - ox) * n_i.x + (py - oy) * n_i.y <= s * sqrt(3)/2
    # where n_i are the outward normals to the edges.
    
    # For flat-top hexagon, the edge normals are at angles 0, 60, 120, 180, 240, 300
    # Wait, for a hexagon with vertices at angles 0,60,...,300, the edge midpoints are at 30,90,...
    # The outward normal to each edge points from center to edge midpoint.
    
    outer_angle = np.radians(outer_angle_deg)
    # Edge normal directions (perpendicular to edges, pointing outward)
    # For flat-top: vertices at 0,60,120,180,240,300
    # Edges are between consecutive vertices. The normal to edge between vertex at angle a and a+60
    # points in direction a+30
    normal_angles = outer_angle + np.radians(np.array([30, 90, 150, 210, 270, 330]))
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for i, (cx, cy, ang) in enumerate(hex_centers):
        verts = hex_vertices(cx, cy, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    # For each point, compute the required side length
    # (px - ox) * cos(na) + (py - oy) * sin(na) <= s * sqrt(3)/2
    # So s >= [(px - ox) * cos(na) + (py - oy) * sin(na)] * 2 / sqrt(3) for each constraint
    # But we need the max over all points and all normals
    
    max_required = 0.0
    for px, py in all_points:
        dx = px - outer_center[0]
        dy = py - outer_center[1]
        for na in normal_angles:
            val = dx * np.cos(na) + dy * np.sin(na)
            if val > 0:  # Only positive projections matter
                required_s = val * 2.0 / np.sqrt(3)
                if required_s > max_required:
                    max_required = required_s
    return max_required

def objective(params):
    """Objective: outer hexagon side length.
    params: [cx1, cy1, ang1, ..., cx6, cy6, ang6, outer_cx, outer_cy, outer_ang]
    6*3 + 3 = 21 parameters
    """
    # Unpack
    inner = []
    for i in range(6):
        cx = params[i*3]
        cy = params[i*3+1]
        ang = params[i*3+2]
        inner.append((cx, cy, ang))
    outer_cx = params[18]
    outer_cy = params[19]
    outer_ang = params[20]
    
    # Check disjointness (penalty if overlapping)
    centers = [(inner[i][0], inner[i][1]) for i in range(6)]
    penalty = 0.0
    for i in range(6):
        for j in range(i+1, 6):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = np.sqrt(3)  # Minimum center distance for non-overlapping unit hexagons
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    # Compute enclosing hexagon side
    side = enclosing_hexagon_side(inner, (outer_cx, outer_cy), outer_ang)
    
    return side + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish within 120s
    
    # Initial guess: honeycomb arrangement of 6 hexagons in a ring + center
    # But we need 12. Let's start with a reasonable packing.
    # For 12 hexagons, use a hexagonal cluster.
    
    # Honeycomb lattice: place hexagons at axial coordinates
    # Axial to Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
    # For a hexagonal arrangement of 12, use:
    # Center + 6 in first ring + 5 in second ring (partial)
    
    # Let's use a simple grid-like arrangement
    sqrt3 = np.sqrt(3)
    
    # Place 12 hexagons in a compact arrangement
    # Rows: 4, 3, 3, 2 or similar
    positions = []
    # Row 0: 4 hexagons
    for k in range(4):
        positions.append((k * sqrt3, 0.0))
    # Row 1: 3 hexagons, offset
    for k in range(3):
        positions.append((sqrt3/2 + k * sqrt3, 1.5))
    # Row 2: 3 hexagons
    for k in range(3):
        positions.append((k * sqrt3, 3.0))
    # Row 3: 2 hexagons
    for k in range(2):
        positions.append((sqrt3/2 + k * sqrt3, 4.5))
    
    # Convert to parameter vector
    # 12 hexagons * 3 + 3 outer = 39 parameters
    # This is too many. Let's reduce by fixing rotations to 0 initially.
    
    # Actually, let's use a smarter approach: optimize only positions and outer hex,
    # with fixed rotation of inner hexagons (0 degrees).
    
    # Start with the positions above, all angles 0
    inner_init = []
    for x, y in positions:
        inner_init.append((x, y, 0.0))
    
    # Center of mass
    cx_mean = np.mean([p[0] for p in inner_init])
    cy_mean = np.mean([p[1] for p in inner_init])
    
    # Translate to center at origin
    inner_init = [(x - cx_mean, y - cy_mean, 0.0) for x, y, a in inner_init]
    
    # Initial outer hexagon: large enough
    # Compute bounding box
    all_x = [p[0] for p in inner_init]
    all_y = [p[1] for p in inner_init]
    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    initial_side = max(width, height) / 2 + 2
    
    # Parameter vector: 12*3 + 3 = 39
    params_init = []
    for x, y, a in inner_init:
        params_init.extend([x, y, a])
    params_init.extend([0.0, 0.0, 0.0])  # Outer center and angle
    
    # Objective function for full problem
    def full_objective(params):
        inner = []
        for i in range(12):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            inner.append((cx, cy, ang))
        outer_cx = params[36]
        outer_cy = params[37]
        outer_ang = params[38]
        
        # Disjointness penalty
        centers = [(inner[i][0], inner[i][1]) for i in range(12)]
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = np.sqrt(3)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        side = enclosing_hexagon_side(inner, (outer_cx, outer_cy), outer_ang)
        return side + 1000.0 * penalty
    
    # Use Nelder-Mead for optimization
    result = minimize(full_objective, params_init, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Return best found so far
        best_params = result.x
    else:
        # Try to improve with more iterations if time allows
        remaining = deadline - elapsed - 5
        if remaining > 10:
            result2 = minimize(full_objective, result.x, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7})
            best_params = result2.x
        else:
            best_params = result.x
    
    # Extract result
    inner = []
    for i in range(12):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang = best_params[i*3+2]
        inner.append([cx, cy, ang])
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_ang = best_params[38]
    
    # Compute actual side length
    side = enclosing_hexagon_side([(p[0], p[1], p[2]) for p in inner], (outer_cx, outer_cy), outer_ang)
    
    return inner, (outer_cx, outer_cy), side, outer_ang
# EVOLVE-BLOCK-END
