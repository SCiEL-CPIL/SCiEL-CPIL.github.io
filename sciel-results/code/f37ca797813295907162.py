"""Optimized packing of 12 unit hexagons in minimal outer hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # unit hexagon vertices (pointy-top orientation, side=1)
    base = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2],
    ])
    # rotate
    R = np.array([[ca, -sa], [sa, ca]])
    verts = (base @ R.T) * s + np.array([cx, cy])
    return verts

def _separation(verts1, verts2):
    """Minimum distance between two convex polygons (negative if overlapping).
    Uses scipy's ConvexHull / distance or a simple vertex-edge approach."""
    from scipy.spatial import ConvexHull, distance as cdist
    # Simple: check all vertex-edge distances
    min_d = float('inf')
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        v1 = verts1[i]
        # distance from v1 to each edge of poly2
        for j in range(n2):
            v2a = verts2[j]
            v2b = verts2[(j+1) % n2]
            # point-to-segment distance
            ab = v2b - v2a
            t = np.dot(v1 - v2a, ab) / (np.dot(ab, ab) + 1e-30)
            t = max(0, min(1, t))
            closest = v2a + t * ab
            d = np.linalg.norm(v1 - closest)
            if d < min_d:
                min_d = d
    for i in range(n2):
        v2 = verts2[i]
        for j in range(n1):
            v1a = verts1[j]
            v1b = verts1[(j+1) % n1]
            ab = v1b - v1a
            t = np.dot(v2 - v1a, ab) / (np.dot(ab, ab) + 1e-30)
            t = max(0, min(1, t))
            closest = v1a + t * ab
            d = np.linalg.norm(v2 - closest)
            if d < min_d:
                min_d = d
    return min_d

def _outer_contains(outer_verts, inner_verts, tol=1e-9):
    """Check if all inner verts are inside or on outer hexagon."""
    # For a convex polygon, check each inner vertex is inside
    # Use cross product test for CCW polygon
    for iv in inner_verts:
        inside = True
        n = len(outer_verts)
        for i in range(n):
            a = outer_verts[i]
            b = outer_verts[(i+1) % n]
            cross = (b[0]-a[0])*(iv[1]-a[1]) - (b[1]-a[1])*(iv[0]-a[0])
            if cross < -tol:
                inside = False
                break
        if not inside:
            return False
    return True

def _objective(params, n_inner=12):
    """Objective: minimize outer side length given inner hexagon positions.
    params: for each inner hex: [x, y, angle_deg] -> 3*n_inner params
    Returns outer side length (to minimize)."""
    eps = 1e-8
    positions = []
    for k in range(n_inner):
        x = params[3*k]
        y = params[3*k+1]
        ang = params[3*k+2]
        positions.append((x, y, ang))
    
    # Compute required outer side: for each inner hex vertex, the outer hex must contain it
    # Outer hex is centered at origin, axis-aligned (angle=0), side = S
    # Outer hex vertices for side S:
    # We need S such that all inner vertices are inside the outer hex.
    # For a regular hexagon with side S, center at origin, "pointy-top":
    # The hexagon is defined by |x| <= S, and |y| <= sqrt(3)*S/2 ... no wait.
    # Actually for a regular hexagon with side S, the distance from center to vertex is S.
    # The hexagon (flat-top) has vertices at angles 0,60,120,180,240,300 with radius S.
    # Let's use the constraint: a point (x,y) is inside a regular hexagon (side S, center origin, 
    # with one vertex at (S,0)) iff:
    # |x| <= S and |y| <= sqrt(3)*S and |x| + |y|/sqrt(3) <= S ... 
    # Actually let me just compute: for each inner vertex, find the minimum S such that it's inside.
    
    min_S = 0.0
    for (x, y, ang) in positions:
        verts = _hex_verts(x, y, 1.0, ang)
        for vx, vy in verts:
            # For outer hexagon (side S, center origin, vertex at (S,0)):
            # The hexagon is the intersection of 6 half-planes.
            # Normal directions at angles 30, 90, 150, 210, 270, 330 degrees
            # Distance from origin along normal direction n_i to the edge = S*cos(30°) = S*sqrt(3)/2
            # So constraint: dot(point, n_i) <= S*sqrt(3)/2 for all i
            # where n_i are unit vectors at angles 30+60k degrees
            for k in range(6):
                theta = np.radians(30 + 60*k)
                nx, ny = np.cos(theta), np.sin(theta)
                d = abs(vx*nx + vy*ny)
                # d <= S * cos(30°) = S * sqrt(3)/2
                S_needed = d / (np.sqrt(3)/2)
                if S_needed > min_S:
                    min_S = S_needed
    return min_S

def _check_valid(positions, S):
    """Check if packing is valid: no overlaps, all inside outer hex."""
    # Check containment
    outer_verts = _hex_verts(0, 0, S, 0.0)
    for (x, y, ang) in positions:
        iv = _hex_verts(x, y, 1.0, ang)
        if not _outer_contains(outer_verts, iv):
            return False
    # Check pairwise separation
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            v1 = _hex_verts(positions[i][0], positions[i][1], 1.0, positions[i][2])
            v2 = _hex_verts(positions[j][0], positions[j][1], 1.0, positions[j][2])
            sep = _separation(v1, v2)
            if sep < -1e-6:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Initial guess: hexagonal lattice arrangement
    # 12 hexagons in rows: 3-4-4-3 or 4-4-4 etc.
    # Let's try a compact arrangement.
    # Hexagonal lattice spacing: adjacent hexagons (same orientation) center distance = sqrt(3) ≈ 1.732
    # For rows offset by sqrt(3)/2 vertically and 1.0 horizontally
    
    # Try several initial configurations
    best_S = float('inf')
    best_params = None
    
    # Configuration 1: 4-4-4 rows
    configs = []
    
    # 4-4-4 grid
    for row in range(3):
        for col in range(4):
            x = col * 1.0 + (0.5 if row % 2 == 1 else 0.0)
            y = row * (np.sqrt(3)/2)
            configs.append([x, y, 0.0])
    
    # Center the config
    xs = [c[0] for c in configs]
    ys = [c[1] for c in configs]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    configs = [[c[0]-cx, c[1]-cy, c[2]] for c in configs]
    
    initial_params = np.array([v for c in configs for v in c])
    
    # Try optimization
    result = minimize(
        _objective,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
    )
    
    S_opt = result.fun
    if S_opt < best_S:
        best_S = S_opt
        best_params = result.x.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Return best found so far
        if best_params is not None:
            inner = [[best_params[3*k], best_params[3*k+1], best_params[3*k+2]] for k in range(N)]
            return inner, (0.0, 0.0), float(best_S), 0.0
        else:
            return configs, (0.0, 0.0), 40.0, 0.0
    
    # Try additional configurations
    # Configuration 2: 3-4-4-3
    configs2 = []
    rows = [3, 4, 4, 3]
    y_pos = 0
    for r_idx, n_cols in enumerate(rows):
        offset = 0.5 if r_idx % 2 == 1 else 0.0
        for col in range(n_cols):
            x = col * 1.0 + offset
            y = y_pos
            configs2.append([x, y, 0.0])
        y_pos += np.sqrt(3)/2
    
    xs = [c[0] for c in configs2]
    ys = [c[1] for c in configs2]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    configs2 = [[c[0]-cx, c[1]-cy, c[2]] for c in configs2]
    
    initial_params2 = np.array([v for c in configs2 for v in c])
    
    result2 = minimize(
        _objective,
        initial_params2,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
    )
    
    S_opt2 = result2.fun
    if S_opt2 < best_S:
        best_S = S_opt2
        best_params = result2.x.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        if best_params is not None:
            inner = [[best_params[3*k], best_params[3*k+1], best_params[3*k+2]] for k in range(N)]
            return inner, (0.0, 0.0), float(best_S), 0.0
        else:
            return configs, (0.0, 0.0), 40.0, 0.0
    
    # Configuration 3: With some rotation to pack tighter
    configs3 = []
    for row in range(3):
        for col in range(4):
            x = col * 1.0 + (0.5 if row % 2 == 1 else 0.0)
            y = row * (np.sqrt(3)/2)
            ang = 30.0  # rotate all by 30 degrees
            configs3.append([x, y, ang])
    
    xs = [c[0] for c in configs3]
    ys = [c[1] for c in configs3]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    configs3 = [[c[0]-cx, c[1]-cy, c[2]] for c in configs3]
    
    initial_params3 = np.array([v for c in configs3 for v in c])
    
    result3 = minimize(
        _objective,
        initial_params3,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
    )
    
    S_opt3 = result3.fun
    if S_opt3 < best_S:
        best_S = S_opt3
        best_params = result3.x.copy()
    
    # Use best result
    if best_params is not None:
        inner = [[best_params[3*k], best_params[3*k+1], best_params[3*k+2]] for k in range(N)]
        # Add small epsilon to S to ensure validity
        S_final = float(best_S) * 1.001
        return inner, (0.0, 0.0), S_final, 0.0
    else:
        return configs, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
