import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    cx, cy = center
    angle_rad = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center, radius = side
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        verts.append((x, y))
    return np.array(verts)

def polygons_disjoint_sat(poly1, poly2, tol=1e-9):
    """Check if two convex polygons are disjoint using SAT."""
    polys = [poly1, poly2]
    for i, poly in enumerate(polys):
        n = len(poly)
        for j in range(n):
            p1 = poly[j]
            p2 = poly[(j + 1) % n]
            # Edge vector
            edge = p2 - p1
            # Normal (perpendicular)
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # If gap exists, polygons are disjoint
            if max1 + tol < min2 or max2 + tol < min1:
                return True  # Disjoint (separated)
    return False  # Overlapping

def hexagons_disjoint(inner, tol=1e-9):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner)
    verts = []
    for i in range(n):
        cx, cy, ang = inner[i]
        verts.append(hex_vertices((cx, cy), 1.0, ang))
    for i in range(n):
        for j in range(i + 1, n):
            if not polygons_disjoint_sat(verts[i], verts[j], tol):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_rotation_deg, outer_side):
    """Check if inner hexagons fit in outer hexagon. Return required outer side."""
    ocx, ocy = outer_center
    orad = np.radians(outer_rotation_deg)
    # Outer hexagon face normals
    normals = []
    for k in range(6):
        a = orad + k * np.pi / 3
        normals.append(np.array([np.cos(a), np.sin(a)]))
    
    max_dist = 0.0
    for i in range(len(inner)):
        cx, cy, ang = inner[i]
        verts = hex_vertices((cx, cy), 1.0, ang)
        for v in verts:
            rel = v - np.array(outer_center)
            for nrm in normals:
                d = np.dot(rel, nrm)
                if d > max_dist:
                    max_dist = d
    
    # Inradius of outer hexagon = outer_side * sqrt(3)/2
    # Need: max_dist <= outer_side * sqrt(3)/2
    # So outer_side >= max_dist / (sqrt(3)/2)
    required_side = max_dist / (np.sqrt(3) / 2)
    return required_side

def objective(params):
    """Objective: outer side length, with penalty for overlaps."""
    n = N
    # params: 3*n for inner (x, y, angle) + 2 for outer center + 1 for outer rotation
    inner_params = params[:3*n]
    outer_cx = params[3*n]
    outer_cy = params[3*n + 1]
    outer_rot = params[3*n + 2]
    
    inner = []
    for i in range(n):
        x = inner_params[3*i]
        y = inner_params[3*i + 1]
        ang = inner_params[3*i + 2]
        inner.append([x, y, ang])
    
    # Check overlaps
    if not hexagons_disjoint(inner, tol=1e-6):
        # Return large penalty
        return 1e6
    
    outer_center = (outer_cx, outer_cy)
    required_side = compute_min_outer_side(inner, outer_center, outer_rot, 0)
    return required_side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration: hexagonal cluster
    # Center hexagon at origin, 6 around it, then 5 more
    # Hexagon spacing: for two unit hexagons to touch, center distance = 2 * inradius * 2? 
    # Actually, for two regular hexagons of side 1 to be tangent, the distance between centers
    # depends on orientation. For parallel hexagons, the minimum distance is 2 * inradius = 2*sqrt(3)/2 = sqrt(3) ≈ 1.732
    # But if we rotate one, they can be closer.
    
    # Let's use a grid-like initial guess
    # Arrange in a hexagonal pattern
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # Row 3: 1 hexagon
    
    # Actually, let me use a simpler initial guess: all at origin with small offsets
    # Then optimize
    
    # Better initial guess: hexagonal lattice
    # For unit hexagons in a hexagonal lattice, the center-to-center distance for touching is sqrt(3)
    # if aligned, but can be less with rotation.
    
    # Let me place them in a pattern that's known to work well
    # 12 hexagons: 1 center, 6 ring, 5 more
    
    dx = np.sqrt(3) + 0.1  # Slight gap to start disjoint
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    # 6 surrounding
    for k in range(6):
        a = k * np.pi / 3
        x = dx * np.cos(a)
        y = dx * np.sin(a)
        inner.append([x, y, 0.0])
    # 5 more: place in the "second ring" gaps
    # The second ring in hexagonal packing has 12 positions, but we only need 5
    # Place them at the vertices of a larger hexagon
    dx2 = 2 * dx + 0.1
    for k in range(5):
        a = k * 2 * np.pi / 5  # Not quite right for hexagonal symmetry
        # Better: use angles that fit hexagonal symmetry
        pass
    
    # Let me use a different approach: place all 12 in a 3x4 grid initially
    # Then optimize
    inner = []
    for i in range(3):
        for j in range(4):
            x = j * (np.sqrt(3) + 0.2)
            y = i * (1.5 + 0.2)
            inner.append([x - 1.5 * (np.sqrt(3) + 0.2), y - 1.0 * (1.5 + 0.2), 0.0])
    
    # Center the configuration
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for i in range(len(inner)):
        inner[i][0] -= cx
        inner[i][1] -= cy
    
    outer_center = [0.0, 0.0]
    outer_rot = 0.0
    
    # Build initial params
    params = np.zeros(3*N + 3)
    for i in range(N):
        params[3*i] = inner[i][0]
        params[3*i + 1] = inner[i][1]
        params[3*i + 2] = inner[i][2]
    params[3*N] = outer_center[0]
    params[3*N + 1] = outer_center[1]
    params[3*N + 2] = outer_rot
    
    best_params = params.copy()
    best_score = objective(params)
    
    # Use Nelder-Mead optimization
    try:
        result = minimize(objective, params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
        if time.time() > deadline - 20:
            raise TimeoutError()
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    n_restarts = 0
    while time.time() < deadline - 30:
        n_restarts += 1
        # Random perturbation
        params2 = best_params + np.random.randn(len(best_params)) * 0.5
        try:
            result2 = minimize(objective, params2, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            if time.time() > deadline - 20:
                break
            if result2.fun < best_score:
                best_params = result2.x
                best_score = result2.fun
        except Exception:
            break
    
    # Extract best solution
    inner_result = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        ang = best_params[3*i + 2]
        inner_result.append([x, y, ang])
    
    outer_cx = best_params[3*N]
    outer_cy = best_params[3*N + 1]
    outer_rot = best_params[3*N + 2]
    
    # Compute actual required outer side
    outer_center = (outer_cx, outer_cy)
    required_side = compute_min_outer_side(inner_result, outer_center, outer_rot, 0)
    
    # Verify disjointness
    if not hexagons_disjoint(inner_result, tol=1e-5):
        # Fallback to seed solution
        inner_result = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        outer_center = (0.0, 0.0)
        required_side = 40.0
        outer_rot = 0.0
    
    return inner_result, outer_center, required_side, outer_rot
