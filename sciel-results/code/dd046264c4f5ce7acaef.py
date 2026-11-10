import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _check_separation(positions):
    """
    Check if all pairs of unit hexagons at given positions are separated.
    Returns True if separated, False otherwise.
    Uses Separating Axis Theorem.
    """
    n = len(positions)
    # Precompute vertices for all hexagons (angle 0 for all inner)
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

def _min_containing_hex_side(positions, outer_center, outer_angle_deg):
    """
    Compute the side length of the smallest regular hexagon with given center and angle
    that contains all 12 unit hexagons at 'positions'.
    """
    # Get all vertices of inner hexagons
    n = len(positions)
    all_verts = np.empty((n * 6, 2))
    for i in range(n):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0)
        all_verts[i*6:(i+1)*6] = verts
    
    # Shift vertices by outer center
    all_verts_shifted = all_verts - outer_center
    
    # Outer hexagon normals
    a_rad = np.radians(outer_angle_deg)
    # Normals are at angle theta + 30 + 60k
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Dot products
    dots = all_verts_shifted @ normals.T
    max_dot = np.max(dots)
    
    # Side length s = (2/sqrt(3)) * max_dot
    s = (2.0 / np.sqrt(3.0)) * max_dot
    
    return s

def _objective(params):
    """
    Objective function.
    params: (27,) array.
    [0:24] -> 12 inner hexagon centers (x1, y1, ..., x12, y12)
    [24:26] -> outer hexagon center (cx, cy)
    [26]   -> outer hexagon angle (deg)
    """
    positions = params[:24].reshape(12, 2)
    outer_center = params[24:26]
    outer_angle = params[26]
    
    # Check separation
    if not _check_separation(positions):
        # Penalty for overlap. 
        # We can estimate overlap by summing distances between centers that are too close,
        # but a constant penalty is simpler and works well with Nelder-Mead.
        # A better penalty encourages separation.
        # Let's use a large constant.
        return 1000.0
    
    s = _min_containing_hex_side(positions, outer_center, outer_angle)
    return s

def _hex_lattice_positions(spacing_scale=1.0):
    """Generate a hexagonal lattice arrangement of 12 unit hexagons."""
    s = spacing_scale
    # Distance between centers of adjacent hexagons in a perfect lattice is 2 * apothem = 2 * (sqrt(3)/2) * side = sqrt(3) * side.
    # Wait, side=1. Apothem = sqrt(3)/2. Distance = 2 * apothem = sqrt(3) ~ 1.732.
    # My previous code used 2.5. That's too far.
    # Let's use the correct distance for touching hexagons.
    # Distance d = sqrt(3) * side.
    d = np.sqrt(3) * 1.0
    
    # Hexagonal lattice vectors
    v1 = np.array([d, 0])
    v2 = np.array([d/2, d * np.sqrt(3)/2])
    
    positions = []
    # Center
    positions.append(np.array([0.0, 0.0]))
    # 6 neighbors
    positions.append(v1)
    positions.append(-v1)
    positions.append(v2)
    positions.append(-v2)
    positions.append(v1 + v2)
    positions.append(-v1 - v2)
    # Next ring (6 more)
    positions.append(2 * v1)
    positions.append(-2 * v1)
    positions.append(2 * v2)
    positions.append(-2 * v2)
    positions.append(2 * v1 + v2) # Wait, 2v1+v2 is not a neighbor of the ring 1 in the same way?
    # Let's just generate a grid of indices and filter.
    
    # Simpler: just use the 12 closest points in a hexagonal lattice.
    # Indices (q, r) in axial coordinates.
    # Center (0,0).
    # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Distance squared in Cartesian:
    # Let's just hardcode the 12 positions for a compact cluster.
    # 1 center, 6 around it, 5 more to make 12?
    # A regular hexagon of side 2 has 19 cells.
    # A regular hexagon of side 1 has 7 cells.
    # We need 12 cells.
    # 7 (center + ring 1) + 5 from ring 2.
    
    pos = []
    # Ring 0
    pos.append([0.0, 0.0])
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        pos.append([d * np.cos(angle), d * np.sin(angle)])
    # Ring 2 (select 5)
    # The 6 neighbors of Ring 1 are at distance 2d.
    # Let's pick 5 of them.
    # Actually, a better initial guess might be a rectangular grid or just random.
    # The lattice above is valid and disjoint.
    
    # Let's use the 12 points:
    # (0,0)
    # (d, 0), (-d, 0)
    # (d/2, d*sqrt(3)/2), (-d/2, -d*sqrt(3)/2)
    # (-d/2, d*sqrt(3)/2), (d/2, -d*sqrt(3)/2)
    # (3d/2, d*sqrt(3)/2)
    # (-3d/2, d*sqrt(3)/2)
    # (3d/2, -d*sqrt(3)/2)
    # (-3d/2, -d*sqrt(3)/2)
    # (0, 2*d*sqrt(3)/2) = (0, d*sqrt(3))
    
    h = d
    v = d * np.sqrt(3)/2
    
    positions = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [-h, 0.0],
        [h/2, v],
        [-h/2, v],
        [h/2, -v],
        [-h/2, -v],
        [3*h/2, v],
        [-3*h/2, v],
        [3*h/2, -v],
        [-3*h/2, -v],
        [0.0, 2*v],
    ])
    return positions

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    # Initial configurations
    inits = []
    
    # 1. Hexagonal lattice (compact)
    base_lattice = _hex_lattice_positions(1.0)
    inits.append(base_lattice)
    
    # 2. Perturbed lattice
    for _ in range(5):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.2)
        
    # 3. Random
    for _ in range(5):
        inits.append(np.random.randn(12, 2) * 2.0)
        
    # 4. Circular
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for r in [1.5, 2.0]:
        pos = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        inits.append(pos)

    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Try a few initial angles and centers for the outer hexagon
        # Outer center should be near the centroid of the inner hexagons
        centroid = np.mean(init_pos, axis=0)
        
        for init_angle in [0, 30, 60, 90, 120, 150]:
            if time.time() > deadline - 20:
                break
                
            # Construct initial params
            init_params = np.zeros(27)
            init_params[:24] = init_pos.flatten()
            init_params[24] = centroid[0]
            init_params[25] = centroid[1]
            init_params[26] = init_angle
            
            # Run Nelder-Mead
            try:
                res = minimize(
                    _objective, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                continue
                
            if res.fun < best_s and res.fun < 100: # Check if it's a valid solution
                best_s = res.fun
                best_params = res.x.copy()
                
    # Fallback
    if best_params is None or best_s > 100:
        best_pos = _hex_lattice_positions(1.0)
        centroid = np.mean(best_pos, axis=0)
        best_angle = 0
        all_verts = np.empty((12*6, 2))
        for i in range(12):
            all_verts[i*6:(i+1)*6] = _hex_vertices(best_pos[i,0], best_pos[i,1], 1.0, 0)
        # Compute s
        a_rad = np.radians(best_angle)
        normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        dots = (all_verts - centroid) @ normals.T
        max_dot = np.max(dots)
        best_s = (2.0 / np.sqrt(3.0)) * max_dot
        best_params = np.zeros(27)
        best_params[:24] = best_pos.flatten()
        best_params[24] = centroid[0]
        best_params[25] = centroid[1]
        best_params[26] = best_angle

    # Extract results
    final_positions = best_params[:24].reshape(12, 2)
    final_center = best_params[24:26]
    final_angle = best_params[26]
    
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([final_center[0], final_center[1], final_angle])
    
    return inner_hex_data, outer_hex_data, best_s
