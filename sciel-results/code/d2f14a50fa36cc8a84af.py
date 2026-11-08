"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _get_lattice_points(n_points):
    """Generate the n_points closest points to the origin in a hexagonal lattice."""
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = np.sqrt(3) * (i + j / 2.0)
            y = 1.5 * j
            points.append((x, y))
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return points[:n_points]

def _objective(params):
    """Objective function: Minimize S + penalties."""
    xs = params[:2*N:2]
    ys = params[1:2*N:2]
    S = params[2*N]
    alpha = params[2*N+1]
    
    if S <= 0:
        return 1e6
        
    # Containment Penalty
    a_rad = np.radians(alpha)
    normals = []
    for k in range(6):
        theta = a_rad + np.radians(60 * k) + np.radians(30)
        normals.append((np.cos(theta), np.sin(theta)))
    
    apothem = S * np.sqrt(3) / 2.0
    
    cont_pen = 0.0
    for i in range(N):
        verts = _hex_verts(xs[i], ys[i], 1.0, 0.0)
        for v in verts:
            for n in normals:
                proj = v[0]*n[0] + v[1]*n[1]
                if proj > apothem:
                    cont_pen += (proj - apothem)**2
    
    # Overlap Penalty
    min_dist = np.sqrt(3.0)
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                overlap_pen += (min_dist - dist)**2
                
    return S + 100.0 * cont_pen + 100.0 * overlap_pen

def _is_valid(inner, center, side, angle):
    """Check if the configuration is valid."""
    # Check containment
    outer_verts = _hex_verts(center[0], center[1], side, angle)
    # We need to check if inner hexagons are inside outer.
    # Using projection method is faster.
    a_rad = np.radians(angle)
    normals = []
    for k in range(6):
        theta = a_rad + np.radians(60 * k) + np.radians(30)
        normals.append((np.cos(theta), np.sin(theta)))
    apothem = side * np.sqrt(3) / 2.0
    
    for x, y, ang in inner:
        verts = _hex_verts(x, y, 1.0, ang)
        for v in verts:
            for n in normals:
                proj = v[0]*n[0] + v[1]*n[1]
                if proj > apothem + 1e-6:
                    return False
    
    # Check disjointness
    min_dist = np.sqrt(3.0) - 1e-6
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            dx = inner[i][0] - inner[j][0]
            dy = inner[i][1] - inner[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial points
    init_pts = _get_lattice_points(N)
    
    xs = np.array([p[0] for p in init_pts])
    ys = np.array([p[1] for p in init_pts])
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs -= cx
    ys -= cy
    
    init_params = np.zeros(2*N + 2)
    for i in range(N):
        init_params[2*i] = xs[i]
        init_params[2*i+1] = ys[i]
    init_params[2*N] = 4.0
    init_params[2*N+1] = 0.0
    
    best_params = init_params
    best_val = _objective(init_params)
    
    try:
        result = minimize(
            _objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 10000}
        )
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Extract
    xs_res = best_params[:2*N:2]
    ys_res = best_params[1:2*N:2]
    S_res = best_params[2*N]
    alpha_res = best_params[2*N+1]
    
    inner = []
    for i in range(N):
        inner.append([float(xs_res[i]), float(ys_res[i]), 0.0])
        
    # Validate
    if not _is_valid(inner, (0.0, 0.0), S_res, alpha_res):
        # Fallback: Use lattice points with a safe S
        # Find max projection
        max_proj = 0
        for x, y, ang in inner:
            verts = _hex_verts(x, y, 1.0, ang)
            for v in verts:
                # Project onto 6 directions
                for k in range(6):
                    theta = np.radians(60 * k + 30)
                    proj = abs(v[0]*np.cos(theta) + v[1]*np.sin(theta))
                    if proj > max_proj:
                        max_proj = proj
        S_safe = max_proj * 2.0 / np.sqrt(3.0) + 0.1
        return inner, (0.0, 0.0), float(S_safe), 0.0

    return inner, (0.0, 0.0), float(S_res), float(alpha_res)

# EVOLVE-BLOCK-END
