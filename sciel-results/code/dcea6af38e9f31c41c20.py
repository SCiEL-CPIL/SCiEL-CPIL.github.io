import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon that contains all points in verts.
    Uses LP to find optimal center and inradius.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _fast_enclosing_side(verts, outer_angle_deg=0.0):
    """
    Fast, differentiable approximation of the enclosing hexagon side length.
    Uses the centroid of vertices as the center.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    
    # Smooth max of absolute projections
    # max_i |p_ik| approximated by (1/beta) * log(sum(exp(beta * |p_ik|)))
    abs_proj = np.abs(projections)
    beta = 50.0
    max_proj = np.max(abs_proj, axis=0)
    # Use log-sum-exp for smoothness
    shifted = abs_proj - max_proj[:, np.newaxis]
    log_sum = np.log(np.sum(np.exp(beta * shifted), axis=0))
    smooth_max = max_proj + log_sum / beta
    
    r = np.max(smooth_max)
    side = r * 2.0 / _SQRT3
    return side

def _hex_distance(p1, a1, p2, a2):
    """Minimum distance between two unit hexagons at positions p1, p2 with angles a1, a2 (radians)."""
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])
    
    def support(a, theta):
        vals = np.cos(a + np.arange(6) * math.pi / 3.0 - theta)
        return np.max(vals)
    
    h1 = support(a1, theta)
    h2 = support(a2, theta + math.pi)
    
    return d - h1 - h2

def _objective_fast(params, positions, outer_angle_deg=0.0):
    """
    Fast objective: Approximate enclosing side + smooth overlap penalty.
    """
    n = len(positions)
    new_positions = []
    new_angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        new_positions.append((x, y))
        new_angles.append(a)
    
    # Overlap penalty
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = _hex_distance(new_positions[i], new_angles[i], new_positions[j], new_angles[j])
            if dist < 0:
                # Quadratic penalty for overlap
                overlap_penalty += dist * dist * 100.0
    
    # Enclosing side
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(new_positions[i][0], new_positions[i][1], math.degrees(new_angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    side = _fast_enclosing_side(all_vertices, outer_angle_deg)
    
    return side + overlap_penalty

def _objective_exact(params, positions, outer_angle_deg=0.0):
    """
    Exact objective: LP enclosing side + hard overlap penalty.
    """
    n = len(positions)
    new_positions = []
    new_angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        new_positions.append((x, y))
        new_angles.append(a)
    
    # Check for overlaps
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dist = _hex_distance(new_positions[i], new_angles[i], new_positions[j], new_angles[j])
            if dist < min_dist:
                min_dist = dist
    
    if min_dist < 0:
        return 1000.0 - min_dist * 100
    
    # Compute enclosing hexagon
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(new_positions[i][0], new_positions[i][1], math.degrees(new_angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: random lattice points
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def construct():
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None
    
    # Try a few layouts
    num_layouts = 3
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0 # Initial angle for all
        x0 = []
        for pos in positions:
            x0.append(pos[0])
            x0.append(pos[1])
            x0.append(math.radians(inner_angle))
        
        # Phase 1: Fast optimization
        if time.time() < _DEADLINE - 20.0:
            try:
                res = minimize(_objective_fast, x0, args=([], 0.0),
                              method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
                
                if res.fun < 100:
                    new_positions = []
                    new_angles = []
                    for i in range(N):
                        new_positions.append((res.x[3*i], res.x[3*i+1]))
                        new_angles.append(math.degrees(res.x[3*i+2]))
                    
                    all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, 0.0)
                    
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_params = res.x[:]
            except Exception:
                pass
        
        # Phase 2: Exact optimization polish
        if time.time() < _DEADLINE - 5.0 and best_params is not None:
            try:
                res = minimize(_objective_exact, best_params, args=([], 0.0),
                              method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < best_side and res.fun < 100:
                    new_positions = []
                    new_angles = []
                    for i in range(N):
                        new_positions.append((res.x[3*i], res.x[3*i+1]))
                        new_angles.append(math.degrees(res.x[3*i+2]))
                    
                    all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, 0.0)
                    
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_params = res.x[:]
            except Exception:
                pass
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
