"""Hexagonal packing of unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices_local(s=1.0):
    """Vertices of a unit regular hexagon centered at origin, flat-top orientation."""
    angles = np.radians(np.arange(6) * 60.0)
    return np.column_stack([s * np.cos(angles), s * np.sin(angles)])

def _hex_vertices(center, angle_deg, s=1.0):
    """Vertices of a hexagon with given center, rotation, and side length."""
    local = _hex_vertices_local(s)
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return local @ R.T + np.array(center)

def _point_in_hexagon(p, center, side, angle_deg):
    """Check if point p is inside hexagon with given center, side, angle."""
    theta = np.radians(angle_deg)
    R_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                      [np.sin(-theta), np.cos(-theta)]])
    p_local = R_inv @ (np.array(p) - np.array(center))
    x, y = p_local
    s_val = side
    sqrt3 = np.sqrt(3.0)
    if x > s_val + 1e-9:
        return False
    if x < -s_val - 1e-9:
        return False
    if sqrt3 * y - x > s_val * sqrt3 + 1e-9:
        return False
    if sqrt3 * y + x > s_val * sqrt3 + 1e-9:
        return False
    if -sqrt3 * y - x > s_val * sqrt3 + 1e-9:
        return False
    if -sqrt3 * y + x > s_val * sqrt3 + 1e-9:
        return False
    return True

def _hex_in_hexagon(center, angle_deg, outer_center, outer_side, outer_angle):
    """Check if a unit hexagon is inside the outer hexagon."""
    verts = _hex_vertices(center, angle_deg, 1.0)
    for v in verts:
        if not _point_in_hexagon(v, outer_center, outer_side, outer_angle):
            return False
    return True

def _hex_overlap(c1, c2, tol=1e-8):
    """Check if two unit hexagons overlap using center distance heuristic."""
    d = np.linalg.norm(np.array(c1) - np.array(c2))
    return d < np.sqrt(3.0) - tol

def _compute_min_hexagon_side(positions, angle=0.0):
    """Compute minimum side length for hexagon containing all inner hexagons."""
    pts = np.array(positions)
    center = np.mean(pts, axis=0)
    
    max_dist = 0
    for pos in positions:
        verts = _hex_vertices(pos, angle, 1.0)
        for v in verts:
            d = np.linalg.norm(v - center)
            max_dist = max(max_dist, d)
    
    return center, max_dist

def construct():
    """Construct N unit hexagons inside a regular hexagon, minimizing outer side length."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_config = None
    
    # Generate hexagonal lattice positions
    v1 = np.array([np.sqrt(3.0), 0.0])
    v2 = np.array([np.sqrt(3.0)/2.0, 1.5])
    
    candidates = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            pos = i * v1 + j * v2
            candidates.append(pos)
    
    candidates.sort(key=lambda p: np.linalg.norm(p))
    positions = [tuple(p) for p in candidates[:N]]
    
    # Initial config
    inner = [[p[0], p[1], 0.0] for p in positions]
    c, s = _compute_min_hexagon_side(positions)
    configs = [(inner, c.tolist(), s, 0.0)]
    
    # Try optimization
    def objective(params):
        positions_arr = np.array(params[:2*N]).reshape(N, 2)
        outer_c = params[2*N:2*N+2]
        outer_s = params[2*N+2]
        outer_ang = params[2*N+3]
        
        if outer_s <= 0:
            return 1e10
        
        for i in range(N):
            if not _hex_in_hexagon(positions_arr[i], 0.0, outer_c, outer_s, outer_ang):
                return 1e10
        for i in range(N):
            for j in range(i+1, N):
                if _hex_overlap(positions_arr[i], positions_arr[j]):
                    return 1e10
        
        return outer_s
    
    for config_idx, (inner_c, c, s, ang) in enumerate(configs):
        if time.time() > deadline:
            break
        
        x0 = np.array([inner_c[i][0] for i in range(N)] + 
                      [inner_c[i][1] for i in range(N)] + 
                      [c[0], c[1], s, ang])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_side and res.fun < 1e9:
                best_side = res.fun
                positions_opt = res.x[:2*N].reshape(N, 2)
                outer_c_opt = res.x[2*N:2*N+2]
                outer_s_opt = res.x[2*N+2]
                outer_ang_opt = res.x[2*N+3]
                best_config = ([[p[0], p[1], 0.0] for p in positions_opt], 
                              outer_c_opt.tolist(), outer_s_opt, outer_ang_opt)
        except Exception:
            pass
    
    if best_config is None or best_side >= 1e9:
        inner = [[i * np.sqrt(3.0), 0.0, 0.0] for i in range(N)]
        c, s = _compute_min_hexagon_side([[p[0], p[1]] for p in inner])
        return inner, c.tolist(), s, 0.0
    
    return best_config
# EVOLVE-BLOCK-END
