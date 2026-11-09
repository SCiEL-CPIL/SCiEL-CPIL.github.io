"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Vertices of a regular hexagon centered at (cx,cy) with given rotation."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3.0
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        a = math.radians(angle_deg)
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        dx_r = dx * math.cos(-a) - dy * math.sin(-a)
        dy_r = dx * math.sin(-a) + dy * math.cos(-a)
        # Hexagon with flat top: |x| <= side, |y| <= side*sqrt(3)/2, and |x|+|y|/sqrt(3) <= side
        # Actually for a regular hexagon with vertices at angles 0,60,...:
        # The hexagon is the intersection of 6 half-planes.
        # For a hexagon with "pointy top" (vertex at top), the constraints are:
        # |x| <= side
        # |y| <= side * sqrt(3) / 2
        # |x| + |y|/sqrt(3) <= side  -- not quite
        # Let me use the standard: for hexagon with vertices at (side*cos(k*pi/3), side*sin(k*pi/3))
        # The hexagon is { (x,y) : |x| <= side, |y| <= side*sqrt(3)/2, |x|+sqrt(3)*|y| <= 2*side }
        # Wait, let me just check all 6 edges.
        verts = hex_vertices(0, 0, 0, side)
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            # Edge from (x1,y1) to (x2,y2), interior is to the left (for CCW)
            ex = x2 - x1
            ey = y2 - y1
            # Normal pointing inward: rotate edge by 90 deg CCW, normalize
            nx = -ey
            ny = ex
            # Check if point is on the correct side
            val = nx * dx_r + ny * dy_r - (nx * x1 + ny * y1)
            if val < -1e-10:
                return False
        return True

    def hex_overlap_dist(v1, v2):
        """Approximate overlap penalty between two hexagons.
        v1, v2 are (cx, cy, angle_deg) for unit hexagons.
        Returns a penalty (0 if no overlap, positive if overlapping)."""
        cx1, cy1, a1 = v1
        cx2, cy2, a2 = v2
        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = math.sqrt(dx * dx + dy * dy)
        # Minimum distance for non-overlap of two unit hexagons is 2*inradius = sqrt(3) ≈ 1.732
        # But depends on orientation. Use a conservative bound.
        if dist >= 2.0:
            return 0.0
        # Check if any vertex of hex1 is inside hex2 or vice versa
        verts1 = hex_vertices(cx1, cy1, a1, 1.0)
        verts2 = hex_vertices(cx2, cy2, a2, 1.0)
        for vx, vy in verts1:
            if point_in_hex(vx, vy, cx2, cy2, a2, 1.0):
                return 1.0
        for vx, vy in verts2:
            if point_in_hex(vx, vy, cx1, cy1, a1, 1.0):
                return 1.0
        # Check edge intersections - simplified: if centers are close, assume overlap
        if dist < 1.5:
            return 1.0
        return 0.0

    def required_outer_side(params):
        """Given positions and angles of inner hexagons, compute minimum outer hexagon side.
        params: flat array [x0,y0,a0, x1,y1,a1, ...]
        Returns the minimum side length L of a regular hexagon (centered at origin, axis-aligned)
        that contains all inner hexagons."""
        # We'll assume outer hexagon is centered at (0,0) with angle 0 for simplicity.
        # Actually, we should optimize the outer hexagon too. For now, assume centered at origin.
        # The outer hexagon with side L, centered at origin, angle 0:
        # Vertices at (L*cos(k*pi/3), L*sin(k*pi/3)) for k=0..5
        # A point (px,py) is inside if for all edges, it's on the correct side.
        # Edge k: from vertex k to vertex k+1.
        # For a regular hexagon, the distance from center to each edge (inradius) is L*sqrt(3)/2.
        # The hexagon is the intersection of 6 half-planes:
        # For edge normal at angle k*pi/3 (pointing outward), the constraint is:
        # px*cos(k*pi/3) + py*sin(k*pi/3) <= L*sqrt(3)/2  for k=0..5
        # Wait, the normals point outward at angles 0, 60, 120, 180, 240, 300.
        # So the constraint is: max_k (px*cos(k*pi/3) + py*sin(k*pi/3)) <= L*sqrt(3)/2
        # Therefore L >= 2/sqrt(3) * max_k (px*cos(k*pi/3) + py*sin(k*pi/3)) over all points.
        
        max_proj = 0.0
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            a = params[3*i+2]
            verts = hex_vertices(cx, cy, a, 1.0)
            for vx, vy in verts:
                for k in range(6):
                    theta = k * math.pi / 3.0
                    proj = vx * math.cos(theta) + vy * math.sin(theta)
                    if proj > max_proj:
                        max_proj = proj
        L = 2.0 / sqrt3 * max_proj
        return L

    def fitness(params):
        """Total fitness: outer side length + penalty for overlaps."""
        L = required_outer_side(params)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = (params[3*i], params[3*i+1], params[3*i+2])
                v2 = (params[3*j], params[3*j+1], params[3*j+2])
                if hex_overlap_dist(v1, v2) > 0:
                    penalty += 10.0
        return L + penalty

    # Initial guess: hexagonal lattice arrangement
    # Place 12 hexagons in a compact pattern
    # Hex lattice: neighbors at distance 2 (for unit hexagons, center-to-center = 2 for non-overlapping? No, 2*inradius = sqrt(3) for flat-to-flat, but center-to-center for touching is 2*inradius = sqrt(3) if flat-to-flat, or 2*circumradius=2 if vertex-to-vertex. Actually for regular hexagons, the minimum center distance for non-overlap is 2*inradius = sqrt(3) ≈ 1.732 when flat sides touch.)
    # Let me use center distance = sqrt(3) for adjacent hexagons.
    
    # A 2x3 grid in hex lattice:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # That's 12.
    
    hex_spacing = sqrt3  # center-to-center for touching
    initial_params = []
    for row in range(4):
        for col in range(3):
            x = (col - 1) * hex_spacing
            y = (row - 1.5) * hex_spacing * 0.866  # hex row offset
            if row % 2 == 1:
                x += hex_spacing / 2
            initial_params.extend([x, y, 0.0])
    
    # Bounds for DE
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 180))   # angle (hexagon has 6-fold symmetry, so 0-60 is enough, but use 0-180 for safety)
    
    # Global search with differential evolution
    best_result = None
    best_fitness = float('inf')
    
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                fitness, bounds, 
                maxiter=50, popsize=15,
                tol=1e-6, seed=42,
                polish=False
            )
            if de_result.fun < best_fitness:
                best_fitness = de_result.fun
                best_result = de_result.x
        except Exception:
            pass
    
    # Local refinement
    if best_result is not None and time.time() < deadline - 10:
        try:
            local_result = minimize(
                fitness, best_result,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if local_result.fun < best_fitness:
                best_fitness = local_result.fun
                best_result = local_result.x
        except Exception:
            pass
    
    if best_result is None:
        best_result = np.array(initial_params)
        best_fitness = fitness(best_result)
    
    # Extract final result
    inner = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i+1]
        a = best_result[3*i+2]
        inner.append([x, y, a])
    
    # Compute actual required L
    L = required_outer_side(best_result)
    
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END

