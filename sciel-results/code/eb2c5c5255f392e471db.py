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
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0))
    
    for i in range(n):
        vi = verts_list[i]
        for j in range(i+1, n):
            vj = verts_list[j]
            # Separating axis test: check all 12 edge normals
            found_sep = False
            # Check normals of vi
            for k in range(6):
                p1 = vi[k]
                p2 = vi[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                min_i = proj_i.min()
                max_i = proj_i.max()
                min_j = proj_j.min()
                max_j = proj_j.max()
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    found_sep = True
                    break
            if found_sep:
                continue
            # Check normals of vj
            for k in range(6):
                p1 = vj[k]
                p2 = vj[(k+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj_i = vi @ normal
                proj_j = vj @ normal
                min_i = proj_i.min()
                max_i = proj_i.max()
                min_j = proj_j.min()
                max_j = proj_j.max()
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    found_sep = True
                    break
            if not found_sep:
                return False
    return True

def _objective(params):
    """
    Objective function for optimization.
    params: (25,) array. First 24 are (x1, y1, ..., x12, y12). Last is angle_deg.
    Returns s (outer hexagon side length) or penalty.
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    
    # Check separation
    if not _check_separation_fast(positions):
        return 1000.0 # Large penalty
    
    # Compute min containing hexagon side
    all_verts = _get_all_inner_vertices(positions)
    
    # Outer hexagon normals
    a_rad = np.radians(angle_deg)
    # Normals are at angle theta + 30 + 60k
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Dot products
    dots = all_verts @ normals.T
    max_dot = np.max(dots)
    
    s = (2.0 / np.sqrt(3.0)) * max_dot
    
    return s

def _hex_lattice_positions(spacing_scale=1.0):
    """Generate a hexagonal lattice arrangement of 12 unit hexagons."""
    s = spacing_scale
    h = 2.5 * s
    v = 2.165 * s  # sqrt(3)/2 * 2.5
    
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
    
    # Initial configurations
    inits = []
    
    # Hexagonal lattice
    base_lattice = _hex_lattice_positions(1.0)
    inits.append(base_lattice)
    
    # Scaled lattices
    for scale in [0.9, 1.1]:
        inits.append(_hex_lattice_positions(scale))
        
    # Circular
    for r in [1.5, 2.0, 2.5]:
        inits.append(_circular_positions(r))
        
    # Random
    for _ in range(5):
        inits.append(np.random.randn(12, 2) * 3.0)
        
    # Perturbed lattice
    for _ in range(5):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.5)

    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Try a few initial angles for the outer hexagon
        for init_angle in [0, 30, 10, -10]:
            if time.time() > deadline - 20:
                break
                
            # Construct initial params
            init_params = np.zeros(25)
            init_params[:24] = init_pos.flatten()
            init_params[24] = init_angle
            
            # Run Nelder-Mead
            try:
                res = minimize(
                    _objective, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
            except Exception:
                continue
                
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
                
    # If optimization failed or didn't find a valid solution, fallback to lattice
    if best_params is None or best_s > 100:
        best_pos = _hex_lattice_positions(1.0)
        best_angle = 0
        # Calculate s for this fallback
        all_verts = _get_all_inner_vertices(best_pos)
        a_rad = np.radians(best_angle)
        normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        dots = all_verts @ normals.T
        max_dot = np.max(dots)
        best_s = (2.0 / np.sqrt(3.0)) * max_dot
        best_params = np.zeros(25)
        best_params[:24] = best_pos.flatten()
        best_params[24] = best_angle

    # Extract results
    final_positions = best_params[:24].reshape(12, 2)
    final_angle = best_params[24]
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, final_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
