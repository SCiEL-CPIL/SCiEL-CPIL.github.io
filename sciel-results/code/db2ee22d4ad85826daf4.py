"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 148.0  # Safety margin below 150s
EPS = 1e-9
SQRT3 = math.sqrt(3)
APOTHEM_UNIT = SQRT3 / 2.0  # apothem of unit hexagon (side=1)

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _get_axes(angle_deg):
    """Precompute edge normals for a regular hexagon with given rotation."""
    a = math.radians(angle_deg)
    axes = []
    for k in range(6):
        na = a + math.radians(60 * k + 30)
        axes.append((math.cos(na), math.sin(na)))
    return axes

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _separating_axis_check(vertsA, vertsB, axesA, axesB):
    """Check if two convex polygons overlap using SAT."""
    for axis in axesA:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + EPS or maxB <= minA + EPS:
            return False
    for axis in axesB:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + EPS or maxB <= minA + EPS:
            return False
    return True

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        na = a + math.radians(60 * k + 30)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / APOTHEM_UNIT
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = []
    axes_list = []
    for p in centers_angles:
        polys.append(_hex_vertices(p[0], p[1], p[2]))
        axes_list.append(_get_axes(p[2]))
        
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j], axes_list[i], axes_list[j]):
                return False
    return True

def _min_overlap_depth(vertsA, vertsB, axesA, axesB):
    """Compute minimum overlap depth between two convex polygons via SAT."""
    min_depth = float('inf')
    all_axes = axesA + axesB
    for axis in all_axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap <= EPS:
            return 0.0
        if overlap < min_depth:
            min_depth = overlap
    return min_depth

def _smooth_objective(v, beta_outer=5.0, beta_collision=5.0):
    """
    Smooth surrogate objective for N parameters: 3*N for hexagons + 2 for center + 1 for outer angle.
    v: flat vector of length 3*N + 3
    """
    # Unpack
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    angle_deg = v[3*N+2]
    
    cx, cy = center
    a = math.radians(angle_deg)
    
    # Compute outer hexagon normals
    normals = []
    for k in range(6):
        na = a + math.radians(60 * k + 30)
        normals.append((math.cos(na), math.sin(na)))
    
    # 1. Containment: smooth max of projections
    proj_vals = []
    all_verts = []
    all_axes = []
    
    for p in cfg:
        verts = _hex_vertices(p[0], p[1], p[2])
        all_verts.append(verts)
        all_axes.append(_get_axes(p[2]))
        
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / APOTHEM_UNIT
                proj_vals.append(val)
                
    if proj_vals:
        max_val = max(proj_vals)
        if max_val > 1e-10:
            sum_exp = 0.0
            for val in proj_vals:
                sum_exp += math.exp(beta_outer * (val - max_val))
            smooth_max = max_val + math.log(sum_exp) / beta_outer
        else:
            smooth_max = max_val
    else:
        smooth_max = 0.0

    # 2. Collision penalty: smooth sum of overlap depths
    collision_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _min_overlap_depth(all_verts[i], all_verts[j], all_axes[i], all_axes[j])
            if depth > 0:
                # Soft penalty: depth^2 is smooth
                collision_penalty += depth * depth
                
    return smooth_max + 10.0 * collision_penalty

def _smooth_grad(v, beta_outer=5.0, beta_collision=5.0):
    """Numerical gradient of smooth objective."""
    h = 1e-6
    grad = np.zeros(len(v))
    f0 = _smooth_objective(v, beta_outer, beta_collision)
    for i in range(len(v)):
        vp = v.copy()
        vp[i] += h
        fp = _smooth_objective(vp, beta_outer, beta_collision)
        grad[i] = (fp - f0) / h
    return grad

def construct():
    t0 = time.time()
    random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice."""
        pts = []
        row_height = SQRT3 * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring():
        """12 hexagons in a ring."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
        
    def make_random():
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-30, 30)])
        return pts
    
    def flatten(cfg, center, angle):
        v = []
        for p in cfg:
            v.extend([p[0], p[1], p[2]])
        v.extend([center[0], center[1], angle])
        return np.array(v)
        
    def unflatten(v):
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        angle = v[3*N+2]
        return cfg, center, angle

    # 1. Generate initial configurations
    inits = [make_hex_lattice(), make_ring(), make_random()]
    outer_angles = [0.0, 15.0, 30.0]
    
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    for init in inits:
        for oa in outer_angles:
            s = full_score(init, (0.0, 0.0), oa)
            if s is not None and s < best_score:
                best_score = s
                best_cfg = [row[:] for row in init]
                best_center = [0.0, 0.0]
                best_angle = oa
                
    if best_cfg is None:
        best_cfg = make_hex_lattice()
        best_score = 100.0
        best_center = [0.0, 0.0]
        best_angle = 0.0

    # 2. Simulated Annealing phase (corrected)
    T = 1.0
    T_end = 0.01
    alpha_sa = 0.999
    
    # Reserve time for optimization and coordinate descent
    opt_budget = 50.0
    cd_budget = 30.0
    sa_deadline = t0 + (TIME_LIMIT - opt_budget - cd_budget - 5.0)
    
    cur_cfg = [row[:] for row in best_cfg]
    cur_center = best_center[:]
    cur_angle = best_angle
    cur_score = best_score
    
    # Store best state
    best_sa_cfg = [row[:] for row in best_cfg]
    best_sa_center = best_center[:]
    best_sa_angle = best_angle
    best_sa_score = best_score
    
    while time.time() < sa_deadline:
        r = random.random()
        
        # Save state before perturbation
        old_cfg = [row[:] for row in cur_cfg]
        old_center = cur_center[:]
        old_angle = cur_angle
        
        if r < 0.5:
            idx = random.randrange(N)
            step = 0.2 * (1.0 + T)
            cur_cfg[idx][0] += random.gauss(0, step)
            cur_cfg[idx][1] += random.gauss(0, step)
        elif r < 0.7:
            idx = random.randrange(N)
            cur_cfg[idx][2] += random.gauss(0, 5.0 + 5.0 * T)
        elif r < 0.85:
            step = 0.1 * (1.0 + T)
            cur_center[0] += random.gauss(0, step)
            cur_center[1] += random.gauss(0, step)
        else:
            cur_angle += random.gauss(0, 5.0 + 5.0 * T)
            
        new_score = full_score(cur_cfg, tuple(cur_center), cur_angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_sa_score:
                    best_sa_score = new_score
                    best_sa_cfg = [row[:] for row in cur_cfg]
                    best_sa_center = cur_center[:]
                    best_sa_angle = cur_angle
            else:
                # Revert
                cur_cfg = old_cfg
                cur_center = old_center
                cur_angle = old_angle
        else:
            # Revert
            cur_cfg = old_cfg
            cur_center = old_center
            cur_angle = old_angle
            
        T = max(T_end, T * alpha_sa)
    
    # Update global best if SA found something better
    if best_sa_score < best_score:
        best_score = best_sa_score
        best_cfg = best_sa_cfg
        best_center = best_sa_center
        best_angle = best_sa_angle

    # 3. L-BFGS-B on smooth objective
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > opt_budget + 10:
        try:
            v0 = flatten(best_cfg, best_center, best_angle)
            n_params = len(v0)
            
            # Run L-BFGS-B with time budget
            opt_deadline = time.time() + opt_budget
            
            # Use a callback to check time
            def time_check_callback(xk):
                return time.time() > opt_deadline
            
            res = minimize(
                _smooth_objective,
                v0,
                method='L-BFGS-B',
                jac=_smooth_grad,
                options={'maxiter': 10000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 50000,
                         'callback': time_check_callback}
            )
            
            if res.success or hasattr(res, 'x'):
                v_opt = res.x if hasattr(res, 'x') else v0
                cfg_opt, center_opt, angle_opt = unflatten(v_opt)
                score_opt = full_score(cfg_opt, tuple(center_opt), angle_opt)
                if score_opt is not None and score_opt < best_score:
                    best_score = score_opt
                    best_cfg = cfg_opt
                    best_center = center_opt
                    best_angle = angle_opt
        except Exception:
            pass

    # 4. Coordinate Descent Polish on Exact Objective
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > cd_budget + 5:
        try:
            cfg_cd = [row[:] for row in best_cfg]
            center_cd = best_center[:]
            angle_cd = best_angle
            
            def get_score():
                s = full_score(cfg_cd, tuple(center_cd), angle_cd)
                if s is None:
                    return float('inf')
                return s
                
            cur_score = get_score()
            best_cfg_cd = [row[:] for row in cfg_cd]
            best_center_cd = center_cd[:]
            best_angle_cd = angle_cd
            best_score_cd = cur_score
            
            n_params = 3 * N + 3
            steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            cd_deadline = time.time() + cd_budget
            for iteration in range(200):
                if time.time() > cd_deadline:
                    break
                    
                improved = False
                for p in range(n_params):
                    if time.time() > cd_deadline:
                        break
                        
                    if p < 3 * N:
                        idx = p // 3
                        param_type = p % 3
                        old_val = cfg_cd[idx][param_type]
                    elif p < 3 * N + 2:
                        idx = p - 3 * N
                        old_val = center_cd[idx]
                    else:
                        old_val = angle_cd
                        
                    best_local_val = old_val
                    best_local_score = cur_score
                    
                    for step in steps:
                        for direction in [-1, 1]:
                            new_val = old_val + direction * step
                            
                            if p < 3 * N:
                                idx = p // 3
                                param_type = p % 3
                                cfg_cd[idx][param_type] = new_val
                            elif p < 3 * N + 2:
                                idx = p - 3 * N
                                center_cd[idx] = new_val
                            else:
                                angle_cd = new_val
                                
                            new_score = get_score()
                            if new_score < best_local_score:
                                best_local_score = new_score
                                best_local_val = new_val
                                
                            # Restore
                            if p < 3 * N:
                                idx = p // 3
                                param_type = p % 3
                                cfg_cd[idx][param_type] = old_val
                            elif p < 3 * N + 2:
                                idx = p - 3 * N
                                center_cd[idx] = old_val
                            else:
                                angle_cd = old_val
                                
                    if best_local_score < cur_score - 1e-12:
                        if p < 3 * N:
                            idx = p // 3
                            param_type = p % 3
                            cfg_cd[idx][param_type] = best_local_val
                        elif p < 3 * N + 2:
                            idx = p - 3 * N
                            center_cd[idx] = best_local_val
                        else:
                            angle_cd = best_local_val
                            
                        cur_score = best_local_score
                        improved = True
                        
                        if cur_score < best_score_cd:
                            best_score_cd = cur_score
                            best_cfg_cd = [row[:] for row in cfg_cd]
                            best_center_cd = center_cd[:]
                            best_angle_cd = angle_cd
                            
                if not improved:
                    break
                    
            if best_score_cd < best_score:
                best_score = best_score_cd
                best_cfg = best_cfg_cd
                best_center = best_center_cd
                best_angle = best_angle_cd
        except Exception:
            pass

    # Final safety check
    if best_cfg is None or not _is_valid(best_cfg):
        best_cfg = make_hex_lattice()
        best_center = [0.0, 0.0]
        best_angle = 0.0
        best_score = _outer_hex_side(best_cfg, tuple(best_center), best_angle)

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
