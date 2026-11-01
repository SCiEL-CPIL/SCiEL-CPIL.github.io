import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def sat_min_separation(c1, c2):
    """
    Compute minimum separation between two hexagons using SAT.
    Returns positive if separated, negative if overlapping.
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            
    return min_sep

def smooth_min_separation(c1, c2, beta=50.0):
    """
    Compute a smooth approximation of minimum separation using soft-min.
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
            
    seps = []
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        seps.append(sep)
        
    seps_arr = np.array(seps)
    min_val = np.min(seps_arr)
    exps = np.exp(beta * (seps_arr - min_val))
    soft_min = min_val - np.log(np.sum(exps)) / beta
    
    return soft_min

def smooth_required_outer_side(centers, outer_center, outer_angle, beta=30.0):
    """
    Compute a smooth approximation of the required outer side length.
    """
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    d_values = []
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                d_values.append(d)
                
    d_arr = np.array(d_values)
    min_d = np.min(d_arr)
    exps = np.exp(beta * (d_arr - min_d))
    soft_max = min_d + np.log(np.sum(exps)) / beta
    
    return soft_max / (np.sqrt(3) / 2)

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
                
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
        
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < -1e-9:
                return False
    return True

def smooth_objective(params, beta=50.0):
    """
    Smooth objective function for L-BFGS-B optimization.
    params: flat array of length 3*N + 3 = [x1,y1,a1, ..., ox,oy,oa]
    """
    n = N
    centers = []
    for i in range(n):
        centers.append([params[3*i], params[3*i+1], params[3*i+2]])
    outer_center = [params[3*n], params[3*n+1]]
    outer_angle = params[3*n+2]
    
    # Smooth required side
    smooth_side = smooth_required_outer_side(centers, outer_center, outer_angle, beta)
    
    # Smooth overlap penalty
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep = smooth_min_separation(centers[i], centers[j], beta)
            if sep < 0:
                overlap_pen += (-sep) ** 2
                
    return smooth_side + 10.0 * overlap_pen

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    return centers

def initial_config_random(seed):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def lbfghs_polish(centers, outer_center, outer_angle, time_limit=20.0):
    """
    Use L-BFGS-B to polish a configuration.
    """
    start_time = time.time()
    n = N
    
    params = np.zeros(3 * n + 3)
    for i in range(n):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    params[3*n] = outer_center[0]
    params[3*n+1] = outer_center[1]
    params[3*n+2] = outer_angle
    
    beta = 100.0
    
    def objective_with_grad(p):
        obj = smooth_objective(p, beta=beta)
        grad = np.zeros_like(p)
        h = 1e-5
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += h
            p_minus = p.copy()
            p_minus[i] -= h
            grad[i] = (smooth_objective(p_plus, beta=beta) - smooth_objective(p_minus, beta=beta)) / (2 * h)
        return obj, grad

    try:
        result = minimize(
            objective_with_grad,
            params,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
        )
    except Exception:
        return centers, outer_center, outer_angle, outer_angle

    if time.time() - start_time > time_limit:
        return centers, outer_center, outer_angle, outer_angle

    new_params = result.x
    new_centers = []
    for i in range(n):
        new_centers.append([new_params[3*i], new_params[3*i+1], new_params[3*i+2]])
    new_outer_center = [new_params[3*n], new_params[3*n+1]]
    new_outer_angle = new_params[3*n+2]
    
    actual_side = required_outer_side(new_centers, new_outer_center, new_outer_angle)
    
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            sep, _ = sat_min_separation(new_centers[i], new_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        return centers, outer_center, outer_angle, outer_angle

    orig_side = required_outer_side(centers, outer_center, outer_angle)
    
    if actual_side < orig_side:
        return new_centers, new_outer_center, actual_side, new_outer_angle
    else:
        return centers, outer_center, outer_angle, outer_angle

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        initial_config_hexagonal(),
        initial_config_random(1),
        initial_config_random(2),
        initial_config_random(3),
        initial_config_random(4),
        initial_config_random(5)
    ]
    
    for config in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        centers = [list(c) for c in config]
        outer_center = [0.0, 0.0]
        outer_angle = 0.0
        
        # Initial side estimate
        init_side = required_outer_side(centers, outer_center, outer_angle)
        
        # Run L-BFGS-B
        polished_centers, polished_outer_center, polished_side, polished_angle = lbfghs_polish(
            centers, outer_center, outer_angle,
            time_limit=min(20, DEADLINE - (time.time() - start_time) - 5)
        )
        
        if check_validity(polished_centers, tuple(polished_outer_center), polished_side, polished_angle):
            if polished_side < best_score:
                best_score = polished_side
                best_result = (polished_centers, tuple(polished_outer_center), polished_side, polished_angle)
                
    if best_result is not None:
        return best_result
        
    # Fallback
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0
