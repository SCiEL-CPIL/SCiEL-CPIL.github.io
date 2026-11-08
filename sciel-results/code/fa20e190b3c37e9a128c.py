"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy)."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def all_inner_vertices(inner):
    """Get all 6 vertices of each inner hexagon."""
    verts = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        verts.append(v)
    return np.vstack(verts)

def min_outer_side(points, center, theta):
    """Given points, center, and outer hex angle theta, return min side length."""
    ks = np.arange(6)
    n_angles = theta + ks * np.pi / 3
    n_x = np.cos(n_angles)
    n_y = np.sin(n_angles)
    
    proj = points[:, 0] * n_x[None, :] + points[:, 1] * n_y[None, :]
    max_proj = proj.max(axis=0)
    
    c_proj = center[0] * n_x + center[1] * n_y
    
    required = max_proj - c_proj
    s = (2.0 / np.sqrt(3.0)) * required.max()
    return max(s, 0.0)

def solve_outer(points):
    """Find minimum side length outer hexagon containing all points."""
    best_s = np.inf
    best_params = None
    
    # Try different theta values (0 to 60 by symmetry of hexagon)
    for theta_deg in np.linspace(0, 60, 37):
        theta = np.radians(theta_deg)
        
        center_init = points.mean(axis=0)
        
        def obj(c):
            return min_outer_side(points, c, theta)
        
        res = minimize(obj, center_init, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
        
        s = res.fun
        if s < best_s:
            best_s = s
            best_params = (res.x[0], res.x[1], theta_deg)
    
    return best_s, best_params

def check_disjoint(inner, tol=1e-9):
    """Check if inner hexagons are pairwise disjoint (allowing touching)."""
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            # Distance between centers
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            # Two unit hexagons are disjoint if center distance >= sqrt(3)
            # (tangent when distance == sqrt(3))
            if dist < np.sqrt(3) - tol:
                return False
    return True

def check_inside(inner, center, side, angle_deg):
    """Check all inner hexagons are inside outer hexagon."""
    outer_verts = hexagon_vertices(center[0], center[1], side, angle_deg)
    
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        for px, py in v:
            # Point in convex polygon test
            inside = True
            for k in range(6):
                x1, y1 = outer_verts[k]
                x2, y2 = outer_verts[(k+1) % 6]
                # Cross product to check if point is on correct side
                cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
                # All should have same sign for convex polygon
                # Determine orientation
                if cross < -1e-9:
                    # Check if this is consistent
                    pass
            # Better: use the half-plane method
            for k in range(6):
                n_ang = np.radians(angle_deg) + k * np.pi / 3
                nx = np.cos(n_ang)
                ny = np.sin(n_ang)
                # Distance from center to this side
                d = (px - center[0]) * nx + (py - center[1]) * ny
                if d > side * np.sqrt(3) / 2 + 1e-9:
                    return False
    return True

def generate_arrangements():
    """Generate candidate inner hexagon arrangements."""
    arrangements = []
    sqrt3 = np.sqrt(3)
    
    # Arrangement 1: Rows 3-4-3-2
    inner = []
    row_y = [0.0, 1.5, 3.0, 4.5]
    row_counts = [3, 4, 3, 2]
    row_offsets = [0.0, sqrt3/2, 0.0, sqrt3/2]
    for i, (y, count, offset) in enumerate(zip(row_y, row_counts, row_offsets)):
        for j in range(count):
            x = (j - (count - 1) / 2.0) * sqrt3 + offset
            inner.append([x, y, 0.0])
    if check_disjoint(inner):
        arrangements.append(inner)
    
    # Arrangement 2: Rows 2-3-4-3
    inner = []
    row_y = [0.0, 1.5, 3.0, 4.5]
    row_counts = [2, 3, 4, 3]
    row_offsets = [sqrt3/2, 0.0, sqrt3/2, 0.0]
    for i, (y, count, offset) in enumerate(zip(row_y, row_counts, row_offsets)):
        for j in range(count):
            x = (j - (count - 1) / 2.0) * sqrt3 + offset
            inner.append([x, y, 0.0])
    if check_disjoint(inner):
        arrangements.append(inner)
    
    # Arrangement 3: Rows 4-3-4-1
    inner = []
    row_y = [0.0, 1.5, 3.0, 4.5]
    row_counts = [4, 3, 4, 1]
    row_offsets = [0.0, sqrt3/2, 0.0, sqrt3/2]
    for i, (y, count, offset) in enumerate(zip(row_y, row_counts, row_offsets)):
        for j in range(count):
            x = (j - (count - 1) / 2.0) * sqrt3 + offset
            inner.append([x, y, 0.0])
    if check_disjoint(inner):
        arrangements.append(inner)
    
    # Arrangement 4: Hexagonal cluster
    inner = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = k * np.pi / 3
        x = sqrt3 * np.cos(ang)
        y = sqrt3 * np.sin(ang)
        inner.append([x, y, 0.0])
    ring2_axial = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2)]
    for q, r in ring2_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
    if check_disjoint(inner):
        arrangements.append(inner)
    
    # Arrangement 5: Two rows of 6
    inner = []
    for j in range(6):
        x = (j - 2.5) * sqrt3
        inner.append([x, 0.0, 0.0])
    for j in range(6):
        x = (j - 2.5) * sqrt3 + sqrt3/2
        inner.append([x, 1.5, 0.0])
    if check_disjoint(inner):
        arrangements.append(inner)
    
    # Arrangement 6: Three rows of 4
    inner = []
    for i in range(3):
        y = i * 1.5
        offset = (sqrt3/2) if i % 2 == 1 else 0.0
        for j in range(4):
            x = (j - 1.5) * sqrt3 + offset
            inner.append([x, y, 0.0])
    if check_disjoint(inner):
        arrangements.append(inner)
    
    # Arrangement 7: Rows 3-4-3-2 with different offset pattern
    inner = []
    row_y = [0.0, 1.5, 3.0, 4.5]
    row_counts = [3, 4, 3, 2]
    row_offsets = [sqrt3/2, 0.0, sqrt3/2, 0.0]
    for i, (y, count, offset) in enumerate(zip(row_y, row_counts, row_offsets)):
        for j in range(count):
            x = (j - (count - 1) / 2.0) * sqrt3 + offset
            inner.append([x, y, 0.0])
    if check_disjoint(inner):
        arrangements.append(inner)
    
    return arrangements

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    best_result = None
    best_score = np.inf
    
    arrangements = generate_arrangements()
    
    for idx, inner in enumerate(arrangements):
        if time.time() > deadline:
            break
        
        points = all_inner_vertices(inner)
        
        s, params = solve_outer(points)
        
        if s < best_score:
            # Verify validity
            cx, cy, theta_deg = params
            if check_inside(inner, (cx, cy), s + 1e-6, theta_deg):
                best_score = s
                best_result = (inner, (cx, cy), s, theta_deg)
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-END
