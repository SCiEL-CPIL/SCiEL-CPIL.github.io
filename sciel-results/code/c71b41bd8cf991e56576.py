import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon with given center, side, rotation (degrees)."""
    rot = np.radians(rot_deg)
    # Pointy-top hexagon: vertices at angles 30, 90, 150, 210, 270, 330 degrees
    angles = np.radians(30 + 60 * np.arange(6)) + rot
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def outer_hexagon_required_side(vertices):
    """
    Given an array of vertices (M x 2), find the minimum side length S of an
    axis-aligned (pointy-top) regular hexagon centered at origin that contains all vertices.
    The 6 outward normals are at 0, 60, 120, 180, 240, 300 degrees.
    Constraint: p . n_k <= S * sqrt(3)/2 for all k.
    So S >= max_k max_i (p_i . n_k) / (sqrt(3)/2)
    """
    normals = np.array([[np.cos(np.radians(60 * k)), np.sin(np.radians(60 * k))] for k in range(6)])
    # vertices: (M, 2)
    dots = vertices @ normals.T  # (M, 6)
    max_proj = np.max(dots, axis=0)  # (6,)
    required = np.max(max_proj) / (np.sqrt(3) / 2)
    return required

def objective(params):
    """
    params: 36-dim array: [x1, y1, r1, x2, y2, r2, ..., x12, y12, r12]
    Returns the required outer hexagon side length.
    """
    vertices_list = []
    for i in range(N):
        x, y, rot = params[3*i], params[3*i+1], params[3*i+2]
        verts = hex_vertices(x, y, 1.0, rot)
        vertices_list.append(verts)
    all_verts = np.vstack(vertices_list)
    return outer_hexagon_required_side(all_verts)

def initial_hex_lattice_config():
    """
    Create an initial configuration of 12 unit hexagons on a hexagonal lattice.
    Hex lattice spacing: for side-1 hexagons, center-to-center distance is sqrt(3) for adjacent.
    We'll arrange them in a compact hexagonal-ish shape.
    
    Let's use a pattern: 
    Row 0 (y=0): 4 hexagons
    Row 1 (y=sqrt(3)): 4 hexagons, offset by sqrt(3)/2
    Row 2 (y=2*sqrt(3)): 4 hexagons
    That's 12.
    """
    dx = np.sqrt(3)  # horizontal spacing for hex lattice
    dy = 1.5  # vertical spacing: for pointy-top, vertical distance between rows is 3/2 * side = 1.5
    # Actually for pointy-top hexagons of side 1:
    # horizontal spacing: sqrt(3)
    # vertical spacing: 1.5
    # offset every other row by sqrt(3)/2
    
    config = []
    # 3 rows of 4
    for row in range(3):
        y = row * 1.5
        offset = (np.sqrt(3) / 2) if row % 2 == 1 else 0
        for col in range(4):
            x = col * np.sqrt(3) + offset
            config.append([x, y, 0.0])
    
    # Center the configuration
    xs = [c[0] for c in config]
    ys = [c[1] for c in config]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for c in config:
        c[0] -= cx
        c[1] -= cy
    
    return config

def construct():
    # Initial configuration
    init_config = initial_hex_lattice_config()
    x0 = np.array([v for c in init_config for v in c])
    
    # Bounds for optimization
    # Reasonable bounds: positions within [-5, 5], rotations [0, 180]
    bounds = [(-5, 5), (-5, 5), (0, 180)] * N
    
    # Try Nelder-Mead first
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_params = result.x
    best_obj = result.fun
    
    # Try a few random restarts if time permits
    import random
    random.seed(42)
    np.random.seed(42)
    
    for restart in range(5):
        # Perturb initial config
        perturbed = init_config.copy()
        for i in range(N):
            perturbed[i][0] += random.uniform(-0.5, 0.5)
            perturbed[i][1] += random.uniform(-0.5, 0.5)
            perturbed[i][2] = random.uniform(0, 180)
        
        x0_r = np.array([v for c in perturbed for v in c])
        result_r = minimize(objective, x0_r, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result_r.fun < best_obj:
            best_obj = result_r.fun
            best_params = result_r.x
    
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        rot = best_params[3*i+2]
        inner.append([x, y, rot])
    
    # Outer hexagon: centered at origin, side = best_obj, rotation = 0
    outer_center = (0.0, 0.0)
    outer_side = float(best_obj)
    outer_rotation = 0.0
    
    return inner, outer_center, outer_side, outer_rotation

