"""Pack 12 unit hexagons into minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
DEADLINE = 150.0  # wall-clock limit
SAFETY_MARGIN = 5.0

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    angles = a + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _outer_constraint(cx, cy, s, theta_deg, vertex):
    """Max over 6 normals of n_i . v, where n_i are outward normals of outer hex."""
    theta = np.radians(theta_deg)
    normals = np.column_stack([np.cos(theta + np.pi * np.arange(6) / 3.0),
                                np.sin(theta + np.pi * np.arange(6) / 3.0)])
    # vertex is (x,y), compute dot products
    return np.max(normals @ vertex)

def _objective(params, n_inner=N):
    """params: [x1,y1,a1, x2,y2,a2, ..., xn,yn,an, cx, cy, s, theta]
    Minimize s such that all inner hex vertices are inside outer hex."""
    inner_params = params[:3 * n_inner]
    cx, cy, s, theta = params[3 * n_inner:3 * n_inner + 4]
    
    # For each inner hexagon, check all 6 vertices
    max_dot = 0.0
    for i in range(n_inner):
        xi = inner_params[3 * i]
        yi = inner_params[3 * i + 1]
        ai = inner_params[3 * i + 2]
        verts = _hex_vertices(xi, yi, ai)
        # Translate to outer center frame
        verts_centered = verts - np.array([cx, cy])
        theta_rad = np.radians(theta)
        normals = np.column_stack([np.cos(theta_rad + np.pi * np.arange(6) / 3.0),
                                    np.sin(theta_rad + np.pi * np.arange(6) / 3.0)])
        dots = normals @ verts_centered.T
        max_dot = max(max_dot, np.max(dots))
    
    # Required s: max_dot <= s * sqrt(3)/2  =>  s >= max_dot / (sqrt(3)/2)
    required_s = max_dot / (np.sqrt(3.0) / 2.0)
    
    # Objective: maximize s (we minimize negative), but we want s to be as small as possible
    # Use: penalize if s < required_s, and minimize s
    penalty = max(0, required_s - s) ** 2
    return s + 100.0 * penalty

def _make_initial_honeycomb():
    """Honeycomb arrangement of 12 hexagons."""
    dx = np.sqrt(3.0)  # horizontal distance between adjacent hex centers
    dy = 1.5  # vertical distance between rows
    positions = []
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y, 0.0])
    return positions

def _make_initial_grid():
    """Grid arrangement."""
    positions = []
    for i in range(3):
        for j in range(4):
            positions.append([i * 2.0, j * 2.0, 0.0])
    return positions

def _make_initial_circular():
    """Circular arrangement."""
    positions = []
    r = 2.0
    for i in range(12):
        angle = 2 * np.pi * i / 12
        positions.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    return positions

def _make_initial_staggered():
    """Staggered grid."""
    positions = []
    dx = np.sqrt(3.0)
    dy = 1.5
    for row in range(4):
        for col in range(3):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append([x, y, 0.0])
    return positions

def _optimize_from_initial(positions, max_iter=200, tol=1e-8):
    """Optimize from given initial positions."""
    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        return None
    
    n = len(positions)
    # Estimate initial s
    all_verts = []
    for pos in positions:
        verts = _hex_vertices(pos[0], pos[1], pos[2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    # Use origin as initial center
    cx0, cy0 = np.mean(all_verts, axis=0)
    # Initial s: max distance from center to any vertex
    dists = np.linalg.norm(all_verts - np.array([cx0, cy0]), axis=1)
    s0 = max(2.0, np.max(dists) / (np.sqrt(3.0) / 2.0) + 0.5)
    
    x0 = np.array([v for pos in positions for v in pos] + [cx0, cy0, s0, 0.0])
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    except Exception:
        return None
    
    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        return None
    
    if not result.success and result.fun > 100:
        return None
    
    best_params = result.x
    s_val = best_params[3 * n + 2]
    
    # Verify: compute actual required s
    inner_params = best_params[:3 * n]
    cx, cy, s, theta = best_params[3 * n:3 * n + 4]
    max_dot = 0.0
    for i in range(n):
        xi = inner_params[3 * i]
        yi = inner_params[3 * i + 1]
        ai = inner_params[3 * i + 2]
        verts = _hex_vertices(xi, yi, ai)
        verts_centered = verts - np.array([cx, cy])
        theta_rad = np.radians(theta)
        normals = np.column_stack([np.cos(theta_rad + np.pi * np.arange(6) / 3.0),
                                    np.sin(theta_rad + np.pi * np.arange(6) / 3.0)])
        dots = normals @ verts_centered.T
        max_dot = max(max_dot, np.max(dots))
    required_s = max_dot / (np.sqrt(3.0) / 2.0)
    
    # Use the actual required s
    final_s = required_s
    
    # Build inner list
    inner = []
    for i in range(n):
        xi = inner_params[3 * i]
        yi = inner_params[3 * i + 1]
        ai = inner_params[3 * i + 2]
        inner.append([xi, yi, ai])
    
    return inner, (cx, cy), final_s, theta, result.fun

def construct():
    """Construct 12 unit hexagons in minimal regular hexagon."""
    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_result = None
    best_score = float('inf')
    
    initial_configs = [
        _make_initial_honeycomb,
        _make_initial_grid,
        _make_initial_circular,
        _make_initial_staggered,
    ]
    
    for init_func in initial_configs:
        if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        
        positions = init_func()
        
        # Try a few random perturbations
        for trial in range(3):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            
            if trial == 0:
                perturbed = positions
            else:
                perturbed = []
                for p in positions:
                    perturbed.append([
                        p[0] + np.random.normal(0, 0.3),
                        p[1] + np.random.normal(0, 0.3),
                        p[2] + np.random.normal(0, 10.0)
                    ])
            
            result = _optimize_from_initial(perturbed, max_iter=150)
            if result is None:
                continue
            
            inner, center, s, angle, obj = result
            if s < best_score:
                best_score = s
                best_result = (inner, center, s, angle)
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
