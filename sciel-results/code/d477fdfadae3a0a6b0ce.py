# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, angle a_deg (degrees)."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return 6 outward edge normals for a hexagon given its 6 CCW vertices."""
    # Edge i goes from vertex i to vertex (i+1)%6
    edges = np.roll(vertices, -1, axis=0) - vertices
    # Outward normal: rotate edge by -90 deg (since CCW polygon, right side is outward)
    # For CCW polygon, outward normal is (dy, -dx) normalized? Let's think:
    # Edge vector e = (ex, ey). For CCW, interior is to the left. So outward is to the right = (ey, -ex).
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms
    return normals

def _separation_check(v1, v2, normals1, normals2, tol=1e-6):
    """
    Check if two hexagons (given vertices and normals) are separated.
    For each of the 12 normals (6 from each), project both hexagons and check if intervals are apart by > tol.
    Returns True if separated (non-intersecting), False otherwise.
    """
    # Combine all 12 normals
    all_normals = np.vstack([normals1, normals2])
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Intervals are apart if max1 < min2 - tol or max2 < min1 - tol
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    return False

def _all_separated(inner_hexes, tol=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for cx, cy, s, a_deg in inner_hexes:
        v = _hex_vertices(cx, cy, s, a_deg)
        nm = _hex_edge_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_check(vertices_list[i], vertices_list[j], normals_list[i], normals_list[j], tol):
                return False
    return True

def _containment_check(inner_hexes, outer_cx, outer_cy, outer_s, outer_a_deg, tol=1e-6):
    """
    Check that all inner hexagon vertices are inside the outer hexagon.
    For each outer edge from p1 to p2, all inner vertices p must satisfy cross(p2-p1, p-p1) >= -tol.
    """
    outer_v = _hex_vertices(outer_cx, outer_cy, outer_s, outer_a_deg)
    for i in range(6):
        p1 = outer_v[i]
        p2 = outer_v[(i + 1) % 6]
        e = p2 - p1
        for cx, cy, s, a_deg in inner_hexes:
            inner_v = _hex_vertices(cx, cy, s, a_deg)
            for p in inner_v:
                d = p - p1
                cross_val = e[0] * d[1] - e[1] * d[0]
                if cross_val < -tol:
                    return False
    return True

def _objective(params, n_inner=11):
    """
    Objective: minimize outer side length.
    params: [outer_cx, outer_cy, outer_a_deg, outer_s, 
             x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
    We fix outer center at (0,0) by symmetry? No, let's optimize freely but shift.
    Actually, to reduce parameters, fix outer center at (0,0) and outer angle at 0.
    Then params: [outer_s, x0, y0, a0, ..., x10, y10, a10]
    Total: 1 + 11*3 = 34 params.
    """
    outer_s = params[0]
    inner_params = params[1:]
    inner_hexes = []
    for i in range(n_inner):
        x = inner_params[3*i]
        y = inner_params[3*i + 1]
        a = inner_params[3*i + 2]
        inner_hexes.append((x, y, 1.0, a))
    
    # Outer hexagon centered at (0,0) with angle 0
    outer_cx, outer_cy, outer_a_deg = 0.0, 0.0, 0.0
    
    penalty = 0.0
    
    # Check separation
    if not _all_separated(inner_hexes, tol=1e-6):
        penalty += 100.0 * outer_s
    
    # Check containment
    if not _containment_check(inner_hexes, outer_cx, outer_cy, outer_s, outer_a_deg, tol=1e-6):
        penalty += 100.0 * outer_s
    
    # Ensure side >= sqrt(11)
    if outer_s < np.sqrt(11):
        penalty += 1000.0 * (np.sqrt(11) - outer_s)
    
    return outer_s + penalty

def hexagon_packing_11():
    n = 11
    # Initial guess: hexagonal lattice arrangement
    # Center at origin, 6 around, 4 more in second ring
    # Unit hexagon side = 1, so distance between centers of adjacent hexagons = sqrt(3) for edge-to-edge
    # Actually, for regular hexagons of side 1, the distance between centers of two hexagons sharing an edge is sqrt(3).
    # Let's use a compact arrangement.
    
    # Start with a simple grid and optimize
    # Use a hexagonal lattice: positions in axial coordinates
    # For 11 hexagons, a good starting point is:
    # Center (0,0), 6 neighbors, and 4 more at distance 2*sqrt(3)
    
    sqrt3 = np.sqrt(3)
    # Axial to pixel: x = sqrt3 * (q + r/2), y = 1.5 * r
    # For hexagon side 1, the "radius" to center of adjacent hex is sqrt(3)
    
    initial_positions = [
        (0, 0, 0),           # center
        (sqrt3, 0, 0),       # right
        (sqrt3/2, 1.5, 0),   # top-right
        (-sqrt3/2, 1.5, 0),  # top-left
        (-sqrt3, 0, 0),      # left
        (-sqrt3/2, -1.5, 0), # bottom-left
        (sqrt3/2, -1.5, 0),  # bottom-right
        (2*sqrt3, 0, 0),     # far right
        (1.5*sqrt3, 1.5, 0), # far top-right
        (-1.5*sqrt3, 1.5, 0),# far top-left
        (-2*sqrt3, 0, 0),    # far left
    ]
    
    # Initial outer side: estimate from bounding box
    xs = [p[0] for p in initial_positions]
    ys = [p[1] for p in initial_positions]
    # Hexagon circumradius = side length, so we need side >= max distance from center to any vertex
    # For now, start with a reasonable estimate
    max_r = max(np.sqrt(x*x + y*y) for x, y, _ in initial_positions) + 1.0
    initial_outer_s = max_r + 0.5
    
    # Build initial params
    params = np.zeros(1 + 3*n)
    params[0] = initial_outer_s
    for i, (x, y, a) in enumerate(initial_positions):
        params[1 + 3*i] = x
        params[2 + 3*i] = y
        params[3 + 3*i] = a
    
    # Optimize
    result = minimize(_objective, params, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    best_params = result.x
    outer_s = best_params[0]
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = best_params[1 + 3*i]
        inner_hex_data[i, 1] = best_params[2 + 3*i]
        inner_hex_data[i, 2] = best_params[3 + 3*i]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    # If optimization failed or gave poor result, fallback to a known valid packing
    if result.fun > 100 or outer_s < np.sqrt(11) - 0.1:
        # Fallback: use the seed program's arrangement but with a tighter outer hex
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_s = 8.0
    
    return inner_hex_data, outer_hex_data, float(outer_s)

# EVOLVE-BLOCK-END
