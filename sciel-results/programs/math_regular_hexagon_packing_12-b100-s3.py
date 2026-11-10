# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _get_all_inner_vertices(positions):
    """Return (72, 2) array of all vertices of 12 unit hexagons at given positions (angle 0)."""
    n = len(positions)
    all_verts = np.empty((n * 6, 2))
    for i in range(n):
        cx, cy = positions[i]
        verts = _hex_vertices(cx, cy, 1.0, 0)
        all_verts[i*6:(i+1)*6] = verts
    return all_verts

def _check_separation_fast(positions):
    """
    Check if all pairs of unit hexagons at given positions are separated.
    Returns True if separated, False otherwise.
    Uses Separating Axis Theorem.
    """
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0))
    
    for i in range(n):
        vi = verts_list[i]
        for j in range(i+1, n):
            vj = verts_list[j]
            found_sep = False
            for k in range(6):
                p1 = vi[k]
                p2 = vi[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                    found_sep = True
                    break
            if found_sep:
                continue
            for k in range(6):
                p1 = vj[k]
                p2 = vj[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                    found_sep = True
                    break
            if not found_sep:
                return False
    return True

def _min_containing_hex_side(positions, ocx, ocy, angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at (ocx, ocy)
    with given angle that contains all vertices of unit hexagons at positions.
    """
    all_verts = _get_all_inner_vertices(positions)
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Shift vertices relative to outer center
    shifted = all_verts - np.array([ocx, ocy])
    dots = shifted @ normals.T
    max_dot = np.max(dots)
    
    s = (2.0 / np.sqrt(3.0)) * max_dot
    return s

def _objective(params):
    """
    Objective function for optimization.
    params: (27,) array. First 24 are (x1, y1, ..., x12, y12), next 2 are outer center (ox, oy), last is angle_deg.
    Returns s (outer hexagon side length) or penalty.
    """
    positions = params[:24].reshape(12, 2)
    ocx, ocy = params[24], params[25]
    angle_deg = params[26]
    
    if not _check_separation_fast(positions):
        return 1000.0
    
    s = _min_containing_hex_side(positions, ocx, ocy, angle_deg)
    return s

def _hex_lattice_positions(spacing_scale=1.0):
    """Generate a hexagonal lattice arrangement of 12 unit hexagons."""
    s = spacing_scale
    h = 2.5 * s
    v = 2.165 * s
    
    positions = np.array([
        [0.0, 0.0],
        [-h, 0.0],
        [h, 0.0],
        [-h/2, v],
        [h/2, v],
        [-h/2, -v],
        [h/2, -v],
        [-3*h/2, v],
        [3*h/2, v],
        [-3*h/2, -v],
        [3*h/2, -v],
        [0.0, -2*v],
    ])
    return positions

def _circular_positions(radius=2.0):
    """Generate positions on a circle."""
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return positions

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    inits = []
    
    base_lattice = _hex_lattice_positions(1.0)
    inits.append(base_lattice)
    
    for scale in [0.9, 1.1]:
        inits.append(_hex_lattice_positions(scale))
        
    for r in [1.5, 2.0, 2.5]:
        inits.append(_circular_positions(r))
        
    for _ in range(5):
        inits.append(np.random.randn(12, 2) * 3.0)
        
    for _ in range(5):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.5)

    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        for init_angle in [0, 30, 10, -10]:
            if time.time() > deadline - 30:
                break
                
            # Initial outer center near centroid of positions
            centroid = init_pos.mean(axis=0)
            
            init_params = np.zeros(27)
            init_params[:24] = init_pos.flatten()
            init_params[24] = centroid[0]
            init_params[25] = centroid[1]
            init_params[26] = init_angle
            
            try:
                res = minimize(
                    _objective, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9}
                )
            except Exception:
                continue
                
            if res.fun < best_s and res.fun < 100:
                best_s = res.fun
                best_params = res.x.copy()
                
    # Fallback
    if best_params is None or best_s > 100:
        best_pos = _hex_lattice_positions(1.0)
        best_angle = 0
        centroid = best_pos.mean(axis=0)
        all_verts = _get_all_inner_vertices(best_pos)
        a_rad = np.radians(best_angle)
        normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        shifted = all_verts - centroid
        dots = shifted @ normals.T
        max_dot = np.max(dots)
        best_s = (2.0 / np.sqrt(3.0)) * max_dot
        best_params = np.zeros(27)
        best_params[:24] = best_pos.flatten()
        best_params[24] = centroid[0]
        best_params[25] = centroid[1]
        best_params[26] = best_angle

    final_positions = best_params[:24].reshape(12, 2)
    ocx, ocy = best_params[24], best_params[25]
    final_angle = best_params[26]
    
    # Shift everything so outer hexagon is at origin
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0] - ocx
    inner_hex_data[:, 1] = final_positions[:, 1] - ocy
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, final_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
