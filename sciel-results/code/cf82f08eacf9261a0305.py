"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_rad=0.0):
    """Return 6 vertices of a unit regular hexagon rotated by angle_rad."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        verts.append([math.cos(theta), math.sin(theta)])
    return np.array(verts)

def _outer_normals(angle_rad=0.0):
    """Return 6 outward normals of the outer hexagon (edge normals)."""
    normals = []
    for k in range(6):
        theta = angle_rad + k * math.pi / 3.0
        normals.append([math.cos(theta), math.sin(theta)])
    return np.array(normals)

def _point_in_polygon(pt, polygon):
    """Ray casting point-in-polygon test. polygon is (n,2)."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside

def _segment_distance(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    elif a < 1e-10:
        t2 = f / e
        t2 = max(0, min(1, t2))
        return np.linalg.norm(r + d2 * t2)
    elif e < 1e-10:
        t1 = -np.dot(d1, r) / a
        t1 = max(0, min(1, t1))
        return np.linalg.norm(r + d1 * t1)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom < 1e-10:
        t1 = 0.0
    else:
        t1 = (b * f - c * e) / denom
        t1 = max(0, min(1, t1))
    
    t2 = (b * t1 + f) / e
    t2 = max(0, min(1, t2))
    if t2 < 0:
        t2 = 0
        t1 = max(0, min(1, -c / a))
    elif t2 > 1:
        t2 = 1
        t1 = max(0, min(1, (b - c) / a))
    
    return np.linalg.norm(r + d1 * t1 - d2 * t2)

def _poly_distance(poly1, poly2):
    """Minimum distance between two convex polygons. Returns 0 if overlapping."""
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for v in poly1:
        if _point_in_polygon(v, poly2):
            return 0.0
    for v in poly2:
        if _point_in_polygon(v, poly1):
            return 0.0
    
    # Check all edge pairs
    n1 = len(poly1)
    n2 = len(poly2)
    min_dist = float('inf')
    for i in range(n1):
        for j in range(n2):
            d = _segment_distance(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2])
            if d < min_dist:
                min_dist = d
    return min_dist

def _compute_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute the minimum side length of an outer hexagon (at given rotation)
    that contains all inner hexagons.
    """
    outer_normals = _outer_normals(outer_angle_rad)
    
    max_extent = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        projections = verts @ outer_normals.T
        abs_proj = np.abs(projections)
        max_extent = max(max_extent, np.max(abs_proj))
    
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def _is_valid(centers, angles, outer_side, outer_angle_rad=0.0):
    """Check if the configuration is valid (no overlaps, all inside outer)."""
    polys = []
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        polys.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_distance(polys[i], polys[j]) < 1e-6:
                return False
    
    # Check containment using projection method
    outer_normals = _outer_normals(outer_angle_rad)
    limit = outer_side * math.sqrt(3) / 2.0
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        projections = verts @ outer_normals.T
        if np.max(np.abs(projections)) > limit + 1e-6:
            return False
    return True

def _seed_solution():
    """Generate a valid seed solution using a hexagonal lattice."""
    d = math.sqrt(3)
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(0.0)
    
    # 4 more from second ring
    ring2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_positions.append([x, y])
    
    # Pick 4 that are well-separated
    selected = [ring2_positions[i] for i in [0, 3, 6, 9]]
    for pos in selected:
        centers.append(pos)
        angles.append(0.0)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    S = _compute_side_length(centers, angles)
    S *= 1.01
    
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 145.0
    
    centers0, angles0, S0 = _seed_solution()
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    best_outer_angle = 0.0
    
    if not _is_valid(best_centers, best_angles, best_S):
        d = math.sqrt(3) * 1.5
        centers_fb = []
        angles_fb = []
        centers_fb.append([0, 0])
        angles_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            centers_fb.append([d * math.cos(theta), d * math.sin(theta)])
            angles_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            centers_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            angles_fb.append(0.0)
        best_centers = np.array(centers_fb)
        best_angles = np.array(angles_fb)
        best_S = _compute_side_length(best_centers, best_angles)
        best_S *= 1.01
    
    def objective(params):
        """
        params: 2N+1 array: [x1, y1, ..., xN, yN, outer_angle]
        angles are fixed to 0 for simplicity in this version, or we can optimize them too.
        Let's optimize angles too: 3N+1 params?
        Actually, let's stick to positions + outer angle first, as inner rotation
        might complicate the lattice structure. But the prompt says "each inner hexagon
        may translate and rotate".
        
        Let's try optimizing: x_i, y_i for i in 1..N, and outer_angle.
        Inner angles are fixed to 0. This is a 2N+1 dimensional problem.
        """
        if time.time() > deadline - 5.0:
            return 1e10
        
        pos = params[:2*N].reshape(N, 2)
        outer_angle = params[2*N]
        
        # Compute S
        S = _compute_side_length(pos, angles0, outer_angle)
        
        # Compute penalty
        penalty = 0.0
        polys = []
        for i in range(N):
            verts = _hex_vertices(0.0) + pos[i]
            polys.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                d = _poly_distance(polys[i], polys[j])
                if d < 0.01:
                    penalty += (0.01 - d) ** 2
        
        return S + 100.0 * penalty
    
    # Differential Evolution
    # Bounds: positions roughly in [-5, 5], outer angle in [0, pi/3]
    bounds = [(-5, 5)] * (2*N) + [(0, math.pi/3)]
    
    x0 = np.hstack([best_centers.flatten(), [0.0]])
    
    try:
        if time.time() < deadline - 30.0:
            res = differential_evolution(objective, bounds, seed=42, maxiter=100, popsize=15, tol=1e-4, polish=True)
            if res.fun < best_S:
                pos = res.x[:2*N].reshape(N, 2)
                outer_angle = res.x[2*N]
                # Verify
                S_val = _compute_side_length(pos, angles0, outer_angle)
                if _is_valid(pos, angles0, S_val * 1.001, outer_angle):
                    best_S = S_val
                    best_centers = pos
                    best_outer_angle = outer_angle
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10.0:
        x0 = np.hstack([best_centers.flatten(), [best_outer_angle]])
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000, 'maxfev': 10000})
            if res.fun < best_S:
                pos = res.x[:2*N].reshape(N, 2)
                outer_angle = res.x[2*N]
                S_val = _compute_side_length(pos, angles0, outer_angle)
                if _is_valid(pos, angles0, S_val * 1.001, outer_angle):
                    best_S = S_val
                    best_centers = pos
                    best_outer_angle = outer_angle
        except Exception:
            pass
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
