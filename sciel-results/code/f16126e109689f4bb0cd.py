"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # 110s deadline to be safe

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Unit hexagon vertices at angles 0, 60, ..., 300
    base_angles = np.arange(6) * np.pi / 3.0
    vx = np.cos(base_angles)
    vy = np.sin(base_angles)
    # Rotate
    rx = cos_a * vx - sin_a * vy
    ry = sin_a * vx + cos_a * vy
    return np.column_stack((cx + rx, cy + ry))


def _hexagon_contains_point(hx, hy, ha_deg, px, py):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with rotation ha_deg."""
    # Transform point into hexagon's local frame
    dx = px - hx
    dy = py - hy
    a = np.radians(ha_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    lx = cos_a * dx + sin_a * dy
    ly = -sin_a * dx + cos_a * dy
    # Unit hexagon in local frame: |x| <= 1, |x/2 + y*sqrt(3)/2| <= 1, |-x/2 + y*sqrt(3)/2| <= 1
    # Actually for unit hexagon (side 1), the vertices are at distance 1 from center.
    # The hexagon is the intersection of 3 strips.
    # For a regular hexagon with circumradius 1:
    # |x| <= 1
    # |x/2 + y*sqrt(3)/2| <= 1
    # |-x/2 + y*sqrt(3)/2| <= 1
    # Wait, let me verify. A unit hexagon (side=1) has vertices at (1,0), (1/2, sqrt(3)/2), etc.
    # The distance from center to a side (apothem) is sqrt(3)/2.
    # The condition for being inside:
    # For the side perpendicular to x-axis: |x| <= 1? No.
    # Let's use the half-plane representation.
    # The 6 sides have normals at angles 0, 60, 120, 180, 240, 300.
    # The distance from center to each side is apothem = sqrt(3)/2.
    # So the condition is: for each normal direction n_i, dot(p, n_i) <= sqrt(3)/2.
    # Actually, since the hexagon is centered at origin in local frame:
    # |lx| <= 1 is NOT correct. The vertex is at (1,0), so x can be 1.
    # But the side between (1,0) and (1/2, sqrt(3)/2) has equation...
    # Let me just use the 6 half-plane inequalities.
    # Normals at angles 0, 60, 120, 180, 240, 300.
    # For normal at angle 0: x <= 1 (but also x >= -1)
    # Actually, the support function of a unit hexagon in direction theta is:
    # h(theta) = max_i dot(v_i, n) where v_i are vertices.
    # For a regular hexagon with circumradius 1, the support function is:
    # h(theta) = 1 / cos((theta mod 60) - 30) ... no that's not right either.
    # Let me just check all 6 vertices and use the fact that the hexagon is convex.
    # Point is inside iff it satisfies all 6 half-plane constraints.
    # The 6 sides, for a hexagon with vertices at angles 0,60,...,300:
    # Side 0: between vertices at 300 and 0. Normal points at 330 deg. 
    # This is getting complicated. Let me use a simpler approach.
    # A point is inside a convex polygon iff it is on the same side of all edges.
    # But for efficiency, let me use the known formula.
    # For a regular hexagon with circumradius R=1, centered at origin, with one vertex at (1,0):
    # The 6 vertices are: (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2,-sqrt(3)/2), (1/2,-sqrt(3)/2)
    # The 6 edges:
    # Edge 0: (1,0) to (1/2, sqrt(3)/2). Outward normal: (sqrt(3)/2, 1/2). Distance: sqrt(3)/2.
    # So constraint: lx*sqrt(3)/2 + ly*1/2 <= sqrt(3)/2
    # Edge 1: (1/2, sqrt(3)/2) to (-1/2, sqrt(3)/2). Outward normal: (0, 1). Distance: sqrt(3)/2.
    # Constraint: ly <= sqrt(3)/2
    # Edge 2: (-1/2, sqrt(3)/2) to (-1,0). Outward normal: (-sqrt(3)/2, 1/2). Distance: sqrt(3)/2.
    # Constraint: -lx*sqrt(3)/2 + ly*1/2 <= sqrt(3)/2
    # Edge 3: (-1,0) to (-1/2,-sqrt(3)/2). Outward normal: (-sqrt(3)/2, -1/2). Distance: sqrt(3)/2.
    # Constraint: -lx*sqrt(3)/2 - ly*1/2 <= sqrt(3)/2
    # Edge 4: (-1/2,-sqrt(3)/2) to (1/2,-sqrt(3)/2). Outward normal: (0, -1). Distance: sqrt(3)/2.
    # Constraint: -ly <= sqrt(3)/2, i.e., ly >= -sqrt(3)/2
    # Edge 5: (1/2,-sqrt(3)/2) to (1,0). Outward normal: (sqrt(3)/2, -1/2). Distance: sqrt(3)/2.
    # Constraint: lx*sqrt(3)/2 - ly*1/2 <= sqrt(3)/2
    
    sqrt3 = np.sqrt(3)
    d = sqrt3 / 2.0
    return (lx * sqrt3/2 + ly * 0.5 <= d + 1e-9 and
            ly <= d + 1e-9 and
            -lx * sqrt3/2 + ly * 0.5 <= d + 1e-9 and
            -lx * sqrt3/2 - ly * 0.5 <= d + 1e-9 and
            -ly <= d + 1e-9 and
            lx * sqrt3/2 - ly * 0.5 <= d + 1e-9)


def _outer_side_for_points(points):
    """Given points (N,2), return the minimum outer hexagon side length (with center at origin, rotation 0)
    such that all points are inside."""
    # Outer hexagon with side S, center (0,0), rotation 0.
    # Vertices at angles 0,60,...,300, distance S from center.
    # Same structure as inner hexagons but scaled by S.
    # Constraint: for each point p, for each of 6 half-planes, dot(p, n_i) <= S * sqrt(3)/2
    # So S >= max over all points and all 6 directions of dot(p, n_i) / (sqrt(3)/2)
    sqrt3 = np.sqrt(3)
    d = sqrt3 / 2.0
    # 6 normals
    normals = np.array([
        [sqrt3/2, 0.5],
        [0.0, 1.0],
        [-sqrt3/2, 0.5],
        [-sqrt3/2, -0.5],
        [0.0, -1.0],
        [sqrt3/2, -0.5]
    ])
    # Compute dot products
    # points: (N,2), normals: (6,2)
    dots = np.max(points @ normals.T, axis=1)  # (N,) max over 6 directions
    max_dot = np.max(dots)
    return max_dot / d


def _check_validity(centers, angles):
    """Check if the configuration is valid (no overlaps, all inside a reasonable bound)."""
    # Check overlaps using SAT
    for i in range(N):
        for j in range(i+1, N):
            # Check if hexagons i and j overlap
            # Use SAT: project onto 12 axes (6 from each hexagon)
            vi = _hex_vertices(centers[i,0], centers[i,1], angles[i])
            vj = _hex_vertices(centers[j,0], centers[j,1], angles[j])
            # Get edge normals
            def edge_normals(verts):
                normals = []
                for k in range(6):
                    e = verts[(k+1)%6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                return np.array(normals)
            ni = edge_normals(vi)
            nj = edge_normals(vj)
            all_normals = np.vstack([ni, nj])
            overlap = True
            for n in all_normals:
                proj_i = vi @ n
                proj_j = vj @ n
                if proj_i.max() < proj_j.min() - 1e-9 or proj_j.max() < proj_i.min() - 1e-9:
                    overlap = False
                    break
            if overlap:
                return False
    return True


def _objective(params):
    """Objective: minimize outer side length + penalty for overlaps."""
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    # Compute outer side needed
    side = _outer_side_for_points(centers)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            vi = _hex_vertices(centers[i,0], centers[i,1], angles[i])
            vj = _hex_vertices(centers[j,0], centers[j,1], angles[j])
            # SAT
            def edge_normals(verts):
                normals = []
                for k in range(6):
                    e = verts[(k+1)%6] - verts[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
                return np.array(normals)
            ni = edge_normals(vi)
            nj = edge_normals(vj)
            all_normals = np.vstack([ni, nj])
            for n in all_normals:
                proj_i = vi @ n
                proj_j = vj @ n
                gap = min(proj_i.max() - proj_j.min(), proj_j.max() - proj_i.min())
                if gap < 0:
                    penalty += (-gap) ** 2
                break  # Only check one axis for speed? No, need all.
            # Actually let's do full SAT for penalty
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    vi = _hex_vertices(centers[i,0], centers[i,1], angles[i])
                    vj = _hex_vertices(centers[j,0], centers[j,1], angles[j])
                    def edge_normals(verts):
                        normals = []
                        for k in range(6):
                            e = verts[(k+1)%6] - verts[k]
                            n = np.array([-e[1], e[0]])
                            n = n / np.linalg.norm(n)
                            normals.append(n)
                        return np.array(normals)
                    ni = edge_normals(vi)
                    nj = edge_normals(vj)
                    all_normals = np.vstack([ni, nj])
                    for n in all_normals:
                        proj_i = vi @ n
                        proj_j = vj @ n
                        gap = min(proj_i.max() - proj_j.min(), proj_j.max() - proj_i.min())
                        if gap < 0:
                            penalty += (-gap) ** 2
                        break
    
    return side + 10.0 * penalty


def construct():
    global DEADLINE
    start = time.time()
    
    # Initial configuration: hexagonal packing
    # Place hexagons in a hex grid pattern
    sqrt3 = np.sqrt(3)
    # Hex grid spacing: for non-overlapping unit hexagons, center-to-center distance is 2*sqrt(3)/2 * something
    # Actually, two unit hexagons side by side: the distance between centers when they share an edge is 2 * apothem = sqrt(3).
    # Wait, no. If two hexagons share an edge, the distance between centers is 2 * (sqrt(3)/2) = sqrt(3).
    # But they would be touching, not overlapping.
    # For a hex grid, the neighbors are at distance sqrt(3) in 6 directions.
    # Let's place them in a hex pattern.
    
    # Center at origin
    positions = []
    angles_init = []
    
    # Place N hexagons in a hexagonal arrangement
    # For N=11: 1 in center, 6 in first ring, 4 more in second ring (partial)
    # Or just use a grid
    
    # Simple approach: place in a linear grid first, then optimize
    # Grid spacing: sqrt(3) in x and y directions won't work directly.
    # For hex packing, use:
    # Row 0: x = 0, sqrt(3), 2*sqrt(3), ...
    # Row 1: x = sqrt(3)/2, 3*sqrt(3)/2, ... y = 1.5
    
    row_y = 1.5
    dx = sqrt3
    
    count = 0
    row = 0
    while count < N:
        col = 0
        while count < N:
            x = col * dx + (row % 2) * dx / 2
            y = row * row_y
            positions.append([x, y])
            angles_init.append(0.0)
            count += 1
            col += 1
        row += 1
    
    centers_init = np.array(positions)
    angles_init = np.array(angles_init)
    
    # Center the configuration
    center_of_mass = centers_init.mean(axis=0)
    centers_init -= center_of_mass
    
    params_init = np.concatenate([centers_init.flatten(), angles_init])
    
    best_params = params_init.copy()
    best_side = _outer_side_for_points(centers_init)
    
    # Try Nelder-Mead optimization
    if time.time() < DEADLINE - 30:
        try:
            result = minimize(_objective, params_init, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.success or result.fun < best_side + 10.0:
                test_params = result.x
                test_centers = test_params[:2*N].reshape(N, 2)
                test_angles = test_params[2*N:]
                if _check_validity(test_centers, test_angles):
                    test_side = _outer_side_for_points(test_centers)
                    if test_side < best_side:
                        best_params = test_params
                        best_side = test_side
        except Exception:
            pass
    
    # Try a few random restarts
    restarts = 3
    for r in range(restarts):
        if time.time() > DEADLINE - 20:
            break
        # Perturb initial configuration
        np.random.seed(42 + r)
        perturb = np.random.randn(*params_init.shape) * 0.5
        perturbed = params_init + perturb
        try:
            result = minimize(_objective, perturbed, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            test_params = result.x
            test_centers = test_params[:2*N].reshape(N, 2)
            test_angles = test_params[2*N:]
            if _check_validity(test_centers, test_angles):
                test_side = _outer_side_for_points(test_centers)
                if test_side < best_side:
                    best_params = test_params
                    best_side = test_side
        except Exception:
            pass
    
    # Extract best result
    best_centers = best_params[:2*N].reshape(N, 2)
    best_angles = best_params[2*N:]
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i, 0]), float(best_centers[i, 1]), float(best_angles[i])])
    
    # Outer hexagon: center at origin, side = best_side, rotation = 0
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_rotation = 0.0
    
    return inner, outer_center, outer_side, outer_rotation

# EVOLVE-BLOCK-END
