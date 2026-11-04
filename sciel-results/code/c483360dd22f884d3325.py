import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_vertices_batch(centers, angles_deg):
    """
    Vectorized: get vertices of multiple unit regular hexagons.
    centers: (N, 2) array, angles_deg: (N,) array
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0  # (6,)
    angle_rad = np.deg2rad(angles_deg)  # (N,)
    angles = angle_rad[:, np.newaxis] + base_angles[np.newaxis, :]
    cos_a = np.cos(angles)  # (N, 6)
    sin_a = np.sin(angles)  # (N, 6)
    cx = centers[:, 0:1, np.newaxis]  # (N, 1)
    cy = centers[:, 1:2, np.newaxis]  # (N, 1)
    x = cx + cos_a  # (N, 6)
    y = cy + sin_a  # (N, 6)
    return np.stack([x, y], axis=-1)  # (N, 6, 2)

def _compute_max_proj(centers, angles, outer_angle_deg):
    """
    Compute max projection of all inner hexagon vertices onto the 6 outer normals.
    Returns max_proj (float).
    """
    theta_rad = np.deg2rad(outer_angle_deg)
    normals = np.array([[np.cos(theta_rad + np.deg2rad(30 + k * 60)),
                         np.sin(theta_rad + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    centers_arr = np.array(centers)
    angles_arr = np.array(angles)
    verts = _hex_vertices_batch(centers_arr, angles_arr)  # (N, 6, 2)
    projs = np.einsum('nij,jk->nik', verts, normals.T)  # (N, 6, 6)
    max_proj = np.max(projs)
    return max_proj

def _min_outer_side(inner_centers, inner_angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length S of the outer hexagon (centered at 0,0, rotated by outer_angle_deg)
    that contains all inner hexagons.
    """
    max_proj = _compute_max_proj(inner_centers, inner_angles, outer_angle_deg)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _objective_full(x):
    """
    Full objective including outer angle.
    x = [x1,y1,a1, x2,y2,a2, ..., outer_angle_rad]
    """
    n = (len(x) - 1) // 3
    centers = [(x[3*i], x[3*i+1]) for i in range(n)]
    angles_deg = [np.rad2deg(x[3*i+2]) for i in range(n)]
    outer_ang_deg = np.rad2deg(x[-1])
    return _min_outer_side(centers, angles_deg, outer_ang_deg)

def _build_hex_lattice_config(n_hex=11):
    """
    Build a hexagonal lattice configuration with proper spacing.
    Uses axial coordinates for hexagonal grid.
    """
    # Axial coordinate to cartesian conversion for hex grid
    # For unit hexagons, distance between centers in same row = sqrt(3)
    # Distance between rows = 1.5, horizontal offset = sqrt(3)/2
    
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Try multiple lattice topologies
    topologies = []
    
    # Star pattern: center + 6 neighbors + 4 more
    star = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2), (-1,2)]
    topologies.append(star)
    
    # Another star variant
    star2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
             (2,0), (2,-1), (1,-2), (0,-2)]
    topologies.append(star2)
    
    # Linear-ish
    linear = [(0,0), (1,0), (2,0), (3,0), (4,0),
              (-1,1), (0,1), (1,1), (2,1), (3,1),
              (-1,-1)]
    topologies.append(linear)
    
    # Compact cluster
    compact = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1),
               (1,-1), (2,0), (2,-1), (1,-2), (0,-2)]
    topologies.append(compact)
    
    # Diagonal
    diag = [(0,0), (1,0), (2,0), (3,0), (4,0),
            (1,1), (2,1), (3,1), (4,1), (5,1),
            (0,-1)]
    topologies.append(diag)
    
    configs = []
    for topo in topologies:
        centers = []
        for i, j in topo:
            c = i * A + j * B
            centers.append(tuple(c))
        configs.append(centers)
    
    return configs

def _simulated_annealing(x0, max_iters=5000, T0=1.0, T_end=0.01, cooling=0.999):
    """
    Simulated annealing to minimize the objective function.
    x0: initial state [x1,y1,a1, ..., outer_angle]
    Returns: (best_x, best_f)
    """
    best_x = np.array(x0, dtype=float)
    best_f = _objective_full(best_x)
    current_x = best_x.copy()
    current_f = best_f
    
    T = T0
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        # Perturb: random small changes
        new_x = current_x.copy()
        for i in range(len(new_x)):
            if i % 3 == 2:  # angle
                new_x[i] += rng.normal(0, 0.1)
            else:  # position
                new_x[i] += rng.normal(0, 0.2)
        
        new_f = _objective_full(new_x)
        
        # Accept with probability
        if new_f < current_f:
            current_x = new_x
            current_f = new_f
            if new_f < best_f:
                best_x = new_x.copy()
                best_f = new_f
        else:
            delta = new_f - current_f
            if delta < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta / max(T, 1e-10))
            if rng.random() < accept_prob:
                current_x = new_x
                current_f = new_f
        
        T = max(T_end, T * cooling)
        
        # Check time
        if time.time() > _sa_deadline:
            break
    
    return best_x, best_f

def _local_optimize(x0, max_iter=300):
    """
    Local optimization using Nelder-Mead with tight tolerances.
    """
    try:
        result = minimize(_objective_full, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        return result.x, result.fun
    except Exception:
        return x0, _objective_full(x0)

_sa_deadline = 140.0  # Set in construct()

def construct():
    global _sa_deadline
    start_time = time.time()
    deadline = 140.0
    _sa_deadline = deadline
    
    # Generate initial configurations from hex lattice
    lattice_configs = _build_hex_lattice_config(N)
    
    best_S = float('inf')
    best_config = None
    
    # Evaluate initial configs with different rotations
    for centers in lattice_configs:
        if time.time() > deadline - 80:
            break
        for rot_deg in range(0, 180, 30):
            if time.time() > deadline - 80:
                break
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            rotated_centers = []
            for (x, y) in centers:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated_centers.append((rx, ry))
            
            for inner_ang in [0, 30, 60]:
                S = _min_outer_side(rotated_centers, [inner_ang]*N, outer_angle_deg=0.0)
                if S < best_S:
                    best_S = S
                    best_config = (rotated_centers, [inner_ang]*N, 0.0)
    
    # Also try a "flower" configuration: center + 6 around + 4 more
    if time.time() < deadline - 80:
        flower_centers = [(0.0, 0.0)]
        d1 = np.sqrt(3)
        for k in range(6):
            ang = k * np.pi / 3
            flower_centers.append((d1 * np.cos(ang), d1 * np.sin(ang)))
        # Add 4 more at larger radius
        d2 = 2 * np.sqrt(3)
        for k in range(4):
            ang = k * np.pi / 2 + np.pi / 6
            flower_centers.append((d2 * np.cos(ang), d2 * np.sin(ang)))
        
        for rot_deg in range(0, 180, 30):
            rot_rad = np.deg2rad(rot_deg)
            cos_r = np.cos(rot_rad)
            sin_r = np.sin(rot_rad)
            rotated_centers = []
            for (x, y) in flower_centers:
                rx = x * cos_r - y * sin_r
                ry = x * sin_r + y * cos_r
                rotated_centers.append((rx, ry))
            
            for inner_ang in [0, 30, 60]:
                S = _min_outer_side(rotated_centers, [inner_ang]*N, outer_angle_deg=0.0)
                if S < best_S:
                    best_S = S
                    best_config = (rotated_centers, [inner_ang]*N, 0.0)
    
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Phase 1: Local optimization on best config
    centers, angles, outer_ang = best_config
    x0 = []
    for (cx, cy), ang in zip(centers, angles):
        x0.extend([cx, cy, np.deg2rad(ang)])
    x0.append(np.deg2rad(outer_ang))
    
    if time.time() < deadline - 60:
        x_opt, f_opt = _local_optimize(x0, max_iter=400)
        if f_opt < best_S:
            best_S = f_opt
            opt_centers = [(x_opt[3*i], x_opt[3*i+1]) for i in range(N)]
            opt_angles = [np.rad2deg(x_opt[3*i+2]) for i in range(N)]
            opt_outer = np.rad2deg(x_opt[-1])
            best_config = (opt_centers, opt_angles, opt_outer)
    
    # Phase 2: Simulated annealing from multiple starting points
    if time.time() < deadline - 30:
        # Try SA from best config
        x0_sa = []
        centers, angles, outer_ang = best_config
        for (cx, cy), ang in zip(centers, angles):
            x0_sa.extend([cx, cy, np.deg2rad(ang)])
        x0_sa.append(np.deg2rad(outer_ang))
        
        _sa_deadline = deadline - 5
        x_sa, f_sa = _simulated_annealing(x0_sa, max_iters=8000, T0=0.5, T_end=0.001, cooling=0.9995)
        if f_sa < best_S:
            best_S = f_sa
            opt_centers = [(x_sa[3*i], x_sa[3*i+1]) for i in range(N)]
            opt_angles = [np.rad2deg(x_sa[3*i+2]) for i in range(N)]
            opt_outer = np.rad2deg(x_sa[-1])
            best_config = (opt_centers, opt_angles, opt_outer)
        
        # Try SA from other good configs
        if time.time() < deadline - 20:
            for centers_try in lattice_configs[:3]:
                if time.time() > deadline - 15:
                    break
                x0_try = []
                for (cx, cy) in centers_try:
                    x0_try.extend([cx, cy, 0.0])
                x0_try.append(0.0)
                
                _sa_deadline = deadline - 5
                x_sa2, f_sa2 = _simulated_annealing(x0_try, max_iters=4000, T0=0.3, T_end=0.001, cooling=0.9995)
                if f_sa2 < best_S:
                    best_S = f_sa2
                    opt_centers = [(x_sa2[3*i], x_sa2[3*i+1]) for i in range(N)]
                    opt_angles = [np.rad2deg(x_sa2[3*i+2]) for i in range(N)]
                    opt_outer = np.rad2deg(x_sa2[-1])
                    best_config = (opt_centers, opt_angles, opt_outer)
    
    # Phase 3: Final local polish
    if time.time() < deadline - 5:
        centers, angles, outer_ang = best_config
        x0_final = []
        for (cx, cy), ang in zip(centers, angles):
            x0_final.extend([cx, cy, np.deg2rad(ang)])
        x0_final.append(np.deg2rad(outer_ang))
        
        x_final, f_final = _local_optimize(x0_final, max_iter=600)
        if f_final < best_S:
            best_S = f_final
            opt_centers = [(x_final[3*i], x_final[3*i+1]) for i in range(N)]
            opt_angles = [np.rad2deg(x_final[3*i+2]) for i in range(N)]
            opt_outer = np.rad2deg(x_final[-1])
            best_config = (opt_centers, opt_angles, opt_outer)
    
    centers, angles, outer_ang = best_config
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), best_S, outer_ang

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
