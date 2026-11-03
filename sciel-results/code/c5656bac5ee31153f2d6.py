import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def hexagon_vertices(cx, cy, s, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append((cx + s * np.cos(a), cy + s * np.sin(a)))
    return np.array(verts)

def axial_to_cartesian(q, r, size=1.0):
    x = size * 3/2 * q
    y = size * np.sqrt(3) * (r + q/2)
    return x, y

def generate_arrangement(pattern_type=0):
    if pattern_type == 0:
        positions_axial = [(0,0)]
        neighbors = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
        positions_axial += neighbors
        positions_axial += [(2,0), (2,-1), (-1,2), (-2,1)]
    elif pattern_type == 1:
        positions_axial = [(0,0)]
        neighbors = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
        positions_axial += neighbors
        positions_axial += [(2,0), (1,1), (-1,1), (-2,0)]
    elif pattern_type == 2:
        positions_axial = [(0,0)]
        neighbors = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
        positions_axial += neighbors
        positions_axial += [(2,0), (3,0), (3,-1), (2,-1)]
    else:
        positions_axial = [(0,0)]
        neighbors = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
        positions_axial += neighbors
        positions_axial += [(2,0), (0,-2), (-2,0), (0,2)]
    positions = []
    for (q, r) in positions_axial:
        x, y = axial_to_cartesian(q, r)
        positions.append((x, y))
    return positions

def compute_required_side(new_xs, new_ys, oa_rad):
    max_proj = 0.0
    n = len(new_xs)
    for i in range(n):
        ix, iy = new_xs[i], new_ys[i]
        for k in range(6):
            va = k * np.pi / 3
            vx = ix + np.cos(va)
            vy = iy + np.sin(va)
            for m in range(6):
                na = oa_rad + np.pi/6 + m * np.pi/3
                nx, ny = np.cos(na), np.sin(na)
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    return max_proj / (np.sqrt(3) / 2)

def optimize_arrangement(pattern_type, oa_init, max_iter=200):
    positions = generate_arrangement(pattern_type)
    n = len(positions)
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    
    def objective(params):
        rot = np.radians(params[0])
        tx, ty = params[1], params[2]
        oa = params[3]
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        new_xs = xs * cos_r - ys * sin_r + tx
        new_ys = xs * sin_r + ys * cos_r + ty
        oa_rad = np.radians(oa)
        return compute_required_side(new_xs, new_ys, oa_rad)
    
    x0 = [0.0, 0.0, 0.0, oa_init]
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7})
    
    best_rot = result.x[0]
    best_tx = result.x[1]
    best_ty = result.x[2]
    best_oa = result.x[3]
    best_s = result.fun
    
    cos_r, sin_r = np.cos(np.radians(best_rot)), np.sin(np.radians(best_rot))
    final_positions = []
    for (px, py) in positions:
        fx = px * cos_r - py * sin_r + best_tx
        fy = px * sin_r + py * cos_r + best_ty
        final_positions.append((fx, fy, 0.0))
    
    return final_positions, (0.0, 0.0), best_s, best_oa

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_score = float('inf')
    
    patterns = [0, 1, 2, 3]
    outer_angles = [0.0, 30.0, 15.0]
    
    for pattern in patterns:
        for oa_init in outer_angles:
            if time.time() > deadline:
                break
            try:
                positions, center, side, angle = optimize_arrangement(
                    pattern_type=pattern, outer_angle=oa_init, max_iter=150
                )
                if side < best_score:
                    best_score = side
                    best_result = (positions, center, side, angle)
            except Exception:
                pass
        if time.time() > deadline:
            break
    
    if best_result is None:
        positions = generate_arrangement(0)
        return [(x, y, 0.0) for (x, y) in positions], (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
