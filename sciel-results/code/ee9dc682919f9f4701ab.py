# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _compute_min_s_for_rotation(vertices, rot_deg):
    """
    Given all vertices of inner hexagons (N, 2), find minimum side length s
    of a regular hexagon (rotated by rot_deg) that contains all vertices.
    Uses linear programming with variables [ox, oy, s].
    """
    rot = np.radians(rot_deg)
    n_pts = len(vertices)
    n_constraints = 6 * n_pts
    A = np.zeros((n_constraints, 3))
    b = np.zeros(n_constraints)
    
    # Precompute the 6 normal directions
    normals = []
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals.append((np.cos(angle), np.sin(angle)))
    
    idx = 0
    for ci, si in normals:
        A[idx:idx+n_pts, 0] = -ci
        A[idx:idx+n_pts, 1] = -si
        A[idx:idx+n_pts, 2] = -np.sqrt(3) / 2
        b[idx:idx+n_pts] = -(ci * vertices[:, 0] + si * vertices[:, 1])
        idx += n_pts
    
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        return res.x[2], (res.x[0], res.x[1])
    else:
        return np.inf, None


def _build_cluster(centers, angle_deg=0.0):
    """
    Build all 72 vertices of the 12 hexagons given their centers and a common rotation.
    Returns (72, 2) array of vertices.
    """
    a = np.radians(angle_deg)
    unit_angles = np.radians(60 * np.arange(6))
    unit_verts = np.column_stack([np.cos(unit_angles), np.sin(unit_angles)])
    
    cos_a, sin_a = np.cos(a), np.sin(a)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rot_unit_verts = unit_verts @ rot_matrix.T
    
    all_verts = np.zeros((72, 2))
    for j in range(12):
        all_verts[j*6:(j+1)*6] = centers[j] + rot_unit_verts
    
    return all_verts


def _validate_packing(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Precompute inner hexagon vertices
    inner_verts_list = []
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        # Check unit edges
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k + 1) % 6]
            if abs(np.linalg.norm(v2 - v1) - 1) > 1e-8:
                return False
        inner_verts_list.append(verts)
    
    # Check separation between inner hexagons using separating axis test
    for i in range(12):
        for j in range(i + 1, 12):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            
            # Get all edge normals from both hexagons
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
            
            separated = False
            for n in normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                if max_i + eps < min_j or max_j + eps < min_i:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check all inner vertices are inside outer hexagon
    for i in range(12):
        verts = inner_verts_list[i]
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def _smooth_min_s(vertices, rot_deg, beta=50.0):
    """
    Smooth surrogate for min s using log-sum-exp.
    For each of the 6 half-planes, compute the required (ox, oy, s) and take a soft-max.
    Returns (s_smooth, ox, oy) as a smooth approximation.
    """
    rot = np.radians(rot_deg)
    n_pts = len(vertices)
    
    # For each of 6 directions, the constraint is:
    # ci*ox + si*oy + sqrt(3)/2 * s >= ci*vx + si*vy for all points
    # So s >= (ci*vx + si*vy - ci*ox - si*oy) / (sqrt(3)/2)
    # The minimum s for a given (ox, oy) is max over all (i, j) of that expression.
    # We use log-sum-exp as a smooth max.
    
    normals = []
    for i in range(6):
        angle = rot + np.radians(60 * i + 30)
        normals.append((np.cos(angle), np.sin(angle)))
    
    # For a fixed (ox, oy), s_min = max over all 6*n_pts constraints
    # We'll optimize over (ox, oy) to minimize the smooth s.
    
    def smooth_s(oo):
        ox, oy = oo
        vals = np.zeros(6 * n_pts)
        idx = 0
        for ci, si in normals:
            vals[idx:idx+n_pts] = (ci * vertices[:, 0] + si * vertices[:, 1] - ci * ox - si * oy) / (np.sqrt(3)/2)
            idx += n_pts
        vals = np.maximum(vals, 0)
        # log-sum-exp for smooth max
        m = vals.max()
        if m > 1e-10:
            s_val = m + np.log(np.sum(np.exp(vals - m))) / (6 * n_pts)
        else:
            s_val = 0.0
        return s_val
    
    # Optimize over (ox, oy)
    from scipy.optimize import minimize as _min
    res = _min(smooth_s, x0=[0, 0], method='Nelder-Mead', 
               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    return res.fun, (res.x[0], res.x[1])


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Treat the 12 hexagons as a rigid honeycomb cluster. The internal geometry is
    fixed (hexagonal lattice with slight expansion to ensure separation). The only degrees of
    freedom are:
    1. Global rotation of the cluster (which also rotates each hexagon)
    2. The outer hexagon's orientation
    3. The outer hexagon's center offset
    
    For each (cluster_rotation, outer_rotation) pair, we analytically compute the minimal
    enclosing outer hexagon via linear programming.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Define 12 hexagon centers in a hexagonal lattice pattern
    # Flower pattern: center + 6 neighbors + 5 from next ring
    # Slight expansion to ensure strict separation (gap > 1e-6)
    eps = 1e-3
    scale = 1.0 + eps
    
    centers_base = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [0, 2 * sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [0, -2 * sqrt3],
    ]) * scale
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # Phase 1: Coarse search over cluster rotation and outer rotation
    n_theta = 61  # 0 to 60 deg in 1 deg steps
    n_phi = 61    # 0 to 60 deg in 1 deg steps
    
    theta_vals = np.linspace(0, 60, n_theta)
    phi_vals = np.linspace(0, 60, n_phi)
    
    for theta_deg in theta_vals:
        if time.time() > deadline - 80:
            break
        
        cluster_verts = _build_cluster(centers_base, theta_deg)
        
        for phi_deg in phi_vals:
            if time.time() > deadline - 80:
                break
            
            s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
            if s < best_s:
                inner_hex_data = np.column_stack([centers_base, np.full(12, theta_deg)])
                outer_hex_data = np.array([center[0], center[1], phi_deg])
                if _validate_packing(inner_hex_data, outer_hex_data, s):
                    best_s = s
                    best_inner = inner_hex_data.copy()
                    best_outer = outer_hex_data.copy()
    
    # Phase 2: Fine grid search around best (theta, phi)
    if best_inner is not None and time.time() < deadline - 150:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        # 0.1 deg steps
        theta_fine = np.linspace(best_theta - 3, best_theta + 3, 61)
        phi_fine = np.linspace(best_phi - 3, best_phi + 3, 61)
        
        for theta_deg in theta_fine:
            if time.time() > deadline - 80:
                break
            
            cluster_verts = _build_cluster(centers_base, theta_deg)
            
            for phi_deg in phi_fine:
                if time.time() > deadline - 80:
                    break
                
                s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([centers_base, np.full(12, theta_deg)])
                    outer_hex_data = np.array([center[0], center[1], phi_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data.copy()
                        best_outer = outer_hex_data.copy()
    
    # Phase 3: Alternative cluster configurations
    if time.time() < deadline - 200:
        alt_centers = np.array([
            [0, 0],
            [2, 0],
            [4, 0],
            [1, sqrt3],
            [3, sqrt3],
            [-1, sqrt3],
            [-3, sqrt3],
            [-1, -sqrt3],
            [1, -sqrt3],
            [3, -sqrt3],
            [-3, -sqrt3],
            [0, 2 * sqrt3],
        ]) * scale
        
        for theta_deg in np.linspace(0, 60, 31):
            if time.time() > deadline - 100:
                break
            
            cluster_verts = _build_cluster(alt_centers, theta_deg)
            
            for phi_deg in np.linspace(0, 60, 31):
                if time.time() > deadline - 100:
                    break
                
                s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                if s < best_s:
                    inner_hex_data = np.column_stack([alt_centers, np.full(12, theta_deg)])
                    outer_hex_data = np.array([center[0], center[1], phi_deg])
                    if _validate_packing(inner_hex_data, outer_hex_data, s):
                        best_s = s
                        best_inner = inner_hex_data.copy()
                        best_outer = outer_hex_data.copy()
    
    # Phase 4: Compressed lattice
    if time.time() < deadline - 150:
        for compress in [0.998, 0.995, 0.99]:
            if time.time() > deadline - 80:
                break
            
            centers_comp = centers_base * (scale * compress)
            
            for theta_deg in np.linspace(0, 60, 31):
                if time.time() > deadline - 80:
                    break
                
                cluster_verts = _build_cluster(centers_comp, theta_deg)
                
                for phi_deg in np.linspace(0, 60, 31):
                    if time.time() > deadline - 80:
                        break
                    
                    s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                    if s < best_s:
                        inner_hex_data = np.column_stack([centers_comp, np.full(12, theta_deg)])
                        outer_hex_data = np.array([center[0], center[1], phi_deg])
                        if _validate_packing(inner_hex_data, outer_hex_data, s):
                            best_s = s
                            best_inner = inner_hex_data.copy()
                            best_outer = outer_hex_data.copy()
    
    # Phase 5: High-precision polish using smooth surrogate + LP verification
    if best_inner is not None and time.time() < deadline - 100:
        best_theta = best_inner[0, 2]
        best_phi = best_outer[2]
        
        # Ultra-fine grid: 0.01 deg steps in a small window
        for radius in [1.0, 0.5, 0.25]:
            if time.time() > deadline - 60:
                break
            
            theta_ultra = np.linspace(best_theta - radius, best_theta + radius, 41)
            phi_ultra = np.linspace(best_phi - radius, best_phi + radius, 41)
            
            for theta_deg in theta_ultra:
                if time.time() > deadline - 60:
                    break
                
                cluster_verts = _build_cluster(centers_base, theta_deg)
                
                for phi_deg in phi_ultra:
                    if time.time() > deadline - 60:
                        break
                    
                    s, center = _compute_min_s_for_rotation(cluster_verts, phi_deg)
                    if s < best_s:
                        inner_hex_data = np.column_stack([centers_base, np.full(12, theta_deg)])
                        outer_hex_data = np.array([center[0], center[1], phi_deg])
                        if _validate_packing(inner_hex_data, outer_hex_data, s):
                            best_s = s
                            best_inner = inner_hex_data.copy()
                            best_outer = outer_hex_data.copy()
        
        # Try to improve with Nelder-Mead on (theta, phi) using LP as objective
        if time.time() < deadline - 60:
            def objective(params):
                theta_d, phi_d = params
                cluster_v = _build_cluster(centers_base, theta_d)
                s_val, _ = _compute_min_s_for_rotation(cluster_v, phi_d)
                return s_val
            
            x0 = [best_theta, best_phi]
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_s:
                theta_opt, phi_opt = res.x
                cluster_verts = _build_cluster(centers_base, theta_opt)
                s_opt, center_opt = _compute_min_s_for_rotation(cluster_verts, phi_opt)
                inner_hex_data = np.column_stack([centers_base, np.full(12, theta_opt)])
                outer_hex_data = np.array([center_opt[0], center_opt[1], phi_opt])
                if _validate_packing(inner_hex_data, outer_hex_data, s_opt):
                    best_s = s_opt
                    best_inner = inner_hex_data.copy()
                    best_outer = outer_hex_data.copy()
    
    # If no valid packing was found, use a default
    if best_inner is None:
        centers = centers_base
        inner_hex_data = np.column_stack([centers, np.zeros(12)])
        s, center = _compute_min_s_for_rotation(_build_cluster(centers, 0), 0)
        outer_hex_data = np.array([center[0], center[1], 0])
        return inner_hex_data, outer_hex_data, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
