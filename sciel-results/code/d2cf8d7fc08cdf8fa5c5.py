"""Structural rewrite: hexagonal lattice packing with Nelder-Mead refinement of outer size."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon with side s, center (cx,cy), rotated by angle_deg."""
    a = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, ... (pointy-top convention)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    angles = angles + a
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _poly_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            proj1 = normal @ v1
            proj2 = normal @ v2
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return False
    return True

def _outer_contains(inner_verts, outer_verts):
    """Check if all inner vertices are inside or on the outer hexagon."""
    # Use point-in-convex-polygon test via cross products
    n = len(outer_verts)
    for p in inner_verts:
        inside = True
        for i in range(n):
            j = (i + 1) % n
            edge = outer_verts[j] - outer_verts[i]
            normal = np.array([-edge[1], edge[0]])
            # Check if p is on the same side (inside) as the polygon center
            # For a convex polygon, all points should be on the same side of each edge
            # We check relative to the first vertex
            ref = outer_verts[0]
            if np.dot(normal, p - outer_verts[i]) > np.dot(normal, ref - outer_verts[i]) + 1e-9:
                inside = False
                break
        if not inside:
            return False
    return True

def _hex_in_hex(inner_verts, outer_verts):
    """Check if inner hexagon is fully inside outer hexagon."""
    return _outer_contains(inner_verts, outer_verts)

def _make_lattice_positions(n):
    """Generate hexagonal lattice positions for n unit hexagons (side=1).
    
    A regular hexagon of side length s has area (3*sqrt(3)/2)*s^2.
    For unit hexagons packed in a hexagonal lattice, the spacing between centers
    in the hexagonal lattice is sqrt(3) (for pointy-top) or 2 (for flat-top with proper rotation).
    
    For unit hexagons (side=1) with flat-top orientation:
    - Width = 2, Height = sqrt(3)
    - Horizontal spacing = 1.5 (for adjacent columns)
    - Vertical spacing = sqrt(3)
    - Offset for odd columns = sqrt(3)/2
    
    Actually, let's use the standard hexagonal close-packing for regular hexagons:
    For hexagons of side 1, the center-to-center distance in a hex lattice is sqrt(3).
    The lattice vectors are:
      a1 = (sqrt(3), 0)
      a2 = (sqrt(3)/2, 3/2)
    """
    positions = []
    # Use axial coordinates for hexagonal grid
    # For n hexagons, arrange in a hexagonal or roughly hexagonal cluster
    # Start with a hexagonal number pattern
    # Hex numbers: 1, 7, 19, 37, 61, ...
    # For N=11, we'll use a compact arrangement
    
    # Simple approach: place in rows
    # Row 0: 1 hex
    # Row 1: 2 hex
    # Row 2: 3 hex
    # Row 3: 2 hex
    # Row 4: 1 hex
    # Total: 9... need 11
    
    # Better: use a hexagonal lattice and pick the n closest points to origin
    # Lattice vectors for hexagonal packing of unit hexagons (side=1)
    # The minimum center distance is sqrt(3) for non-overlapping
    dx = math.sqrt(3)
    dy = 1.5
    
    # Generate candidate lattice points
    candidates = []
    max_radius = 5
    for row in range(-max_radius, max_radius + 1):
        for col in range(-max_radius, max_radius + 1):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            candidates.append((x, y, row * row + col * col))
    
    candidates.sort(key=lambda c: c[2])
    positions = []
    for x, y, _ in candidates:
        if len(positions) >= n:
            break
        positions.append([x, y, 0.0])  # angle 0
    
    return positions

def _compute_objective(params, n, outer_center, outer_angle):
    """Compute outer side length needed for given inner positions.
    
    params: flattened array of [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}]
    We fix inner angles to 0 for simplicity (structural choice).
    """
    inner_positions = params.reshape(-1, 2)
    
    # Get inner hexagon vertices
    inner_verts_list = []
    for i in range(n):
        cx, cy = inner_positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        inner_verts_list.append(verts)
    
    # Check pairwise overlaps
    for i in range(n):
        for j in range(i + 1, n):
            if _poly_overlap(inner_verts_list[i], inner_verts_list[j]):
                return 1e6  # penalty for overlap
    
    # Find minimum outer side length such that all inner hexagons fit
    # The outer hexagon has center outer_center, angle outer_angle
    # We need to find the minimum s such that all inner hexes are inside
    
    # Binary search on outer side length
    lo, hi = 1.0, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], mid, outer_angle)
        all_inside = True
        for iv in inner_verts_list:
            if not _hex_in_hex(iv, outer_verts):
                all_inside = False
                break
        if all_inside:
            hi = mid
        else:
            lo = mid
    
    return hi

def construct():
    start_time = time.time()
    
    n = N
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Initial lattice positions
    positions = _make_lattice_positions(n)
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    # Nelder-Mead optimization
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return 1e8
        return _compute_objective(params, n, outer_center, outer_angle)
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    # Run Nelder-Mead with time check
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': 2000,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        )
        if result.fun < best_f:
            best_x = result.x
            best_f = result.fun
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5:
        # Return best found
        inner = [[best_x[2*i], best_x[2*i+1], 0.0] for i in range(n)]
        return inner, outer_center, best_f, outer_angle
    
    # Try a few more local optimizations from perturbed starts
    rng = np.random.default_rng(42)
    for attempt in range(3):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        x_perturbed = best_x + rng.normal(0, 0.1, size=best_x.shape)
        try:
            result = minimize(
                objective,
                x_perturbed,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'maxfev': 500,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'adaptive': True
                }
            )
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    inner = [[best_x[2*i], best_x[2*i+1], 0.0] for i in range(n)]
    return inner, outer_center, best_f, outer_angle
