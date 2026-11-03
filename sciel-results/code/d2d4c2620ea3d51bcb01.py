import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
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
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _build_positions(layout_idx):
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
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        positions = []
        positions.append((0, 0))
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        positions = []
        y1 = -0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        positions = []
        y1 = -0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        positions = []
        y0 = -3.0*scale
        positions.append((0, y0))
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 3.0*scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _hex_distance_exact(p1, a1, p2, a2):
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

def _objective_smooth(params, outer_angle_deg, beta=50.0):
    n = len(params) // 3
    positions = []
    angles = []
    for i in range(n):
        x, y, a_deg = params[3*i], params[3*i+1], params[3*i+2]
        positions.append((x, y))
        angles.append(math.radians(a_deg))
    
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(positions[i][0], positions[i][1], math.degrees(angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    base_angle = math.radians(outer_angle_deg + 30)
    normals = np.column_stack([np.cos(base_angle + np.arange(6) * math.pi / 3.0),
                                np.sin(base_angle + np.arange(6) * math.pi / 3.0)])
    
    M = len(all_vertices)
    A_rows = []
    b_vals = []
    for i in range(M):
        vi = all_vertices[i]
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
            r = res.x[2]
            side = r * 2.0 / _SQRT3
        else:
            side = 100.0
    except:
        side = 100.0
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            p1 = np.array([params[3*i], params[3*i+1]])
            p2 = np.array([params[3*j], params[3*j+1]])
            a1 = angles[i]
            a2 = angles[j]
            dist = _hex_distance_exact(p1, a1, p2, a2)
            if dist < 0:
                penalty += (-dist) ** 2 * 100.0
            else:
                if dist < 0.1:
                    penalty += (0.1 - dist) ** 2 * 10.0
    
    return side + penalty

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Phase 1: Initial Sweep with finer angle resolution
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 20.0:
            break
        
        inner_angle = 30.0
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        # Try outer angles in steps of 0.5 for better resolution
        for outer_angle_deg in np.arange(0, 30.0, 0.5):
            if time.time() > _DEADLINE - 20.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle

    # Phase 2: Random Hill Climbing / Simulated Annealing with larger steps
    if best_inner is not None and time.time() < _DEADLINE - 15.0:
        current_inner = [list(x) for x in best_inner]
        current_side = best_side
        current_center = best_center
        current_outer_angle = best_outer_angle
        
        T = 0.2
        T_min = 0.001
        alpha = 0.999
        
        for iter in range(5000):
            if time.time() > _DEADLINE - 15.0:
                break
            
            idx = np.random.randint(N)
            
            old_x, old_y, old_a = current_inner[idx]
            
            dx = np.random.normal(0, 0.15)
            dy = np.random.normal(0, 0.15)
            da = np.random.normal(0, 0.15)
            
            new_x = old_x + dx
            new_y = old_y + dy
            new_a_deg = old_a + math.degrees(da)
            
            overlap = False
            p1_new = np.array([new_x, new_y])
            a1_new = math.radians(new_a_deg)
            
            for j in range(N):
                if j == idx:
                    continue
                p2 = np.array([current_inner[j][0], current_inner[j][1]])
                a2 = math.radians(current_inner[j][2])
                
                d = np.linalg.norm(p1_new - p2)
                if d < _SQRT3:
                    overlap = True
                    break
                if d > 2.0:
                    continue
                
                dist = _hex_distance_exact(p1_new, a1_new, p2, a2)
                if dist < 0:
                    overlap = True
                    break
            
            if overlap:
                continue
            
            current_inner[idx] = [new_x, new_y, new_a_deg]
            
            all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in current_inner])
            res = _min_enclosing_hex_side_lp(all_vertices, current_outer_angle)
            
            if res:
                cx, cy, new_side = res
                new_center = (cx, cy)
            else:
                new_side = float('inf')
                new_center = current_center
            
            if new_side < current_side:
                current_side = new_side
                current_center = new_center
            else:
                delta = new_side - current_side
                if np.random.rand() < math.exp(-delta / T):
                    current_side = new_side
                    current_center = new_center
                else:
                    current_inner[idx] = [old_x, old_y, old_a]
            
            T *= alpha
            if T < T_min:
                T = T_min

        if current_side < best_side:
            best_side = current_side
            best_inner = current_inner
            best_center = current_center

    # Phase 3: Gradient-based refinement using L-BFGS-B with multiple outer angles
    if best_inner is not None and time.time() < _DEADLINE - 8.0:
        x0 = []
        for i in range(N):
            x0.append(best_inner[i][0])
            x0.append(best_inner[i][1])
            x0.append(best_inner[i][2])
        x0 = np.array(x0)
        
        # Try more outer angles for gradient step
        for oa in [best_outer_angle, best_outer_angle - 3, best_outer_angle + 3, 
                   best_outer_angle - 1, best_outer_angle + 1]:
            if time.time() > _DEADLINE - 8.0:
                break
            
            def obj(params):
                return _objective_smooth(params, oa, beta=50.0)
            
            try:
                res_opt = minimize(obj, x0, method='L-BFGS-B', 
                                   options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res_opt.success or res_opt.fun < best_side:
                    new_params = res_opt.x
                    new_inner = []
                    for i in range(N):
                        new_inner.append([new_params[3*i], new_params[3*i+1], new_params[3*i+2]])
                    
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            p1 = np.array([new_inner[i][0], new_inner[i][1]])
                            p2 = np.array([new_inner[j][0], new_inner[j][1]])
                            a1 = math.radians(new_inner[i][2])
                            a2 = math.radians(new_inner[j][2])
                            dist = _hex_distance_exact(p1, a1, p2, a2)
                            if dist < -1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in new_inner])
                        res_lp = _min_enclosing_hex_side_lp(all_vertices, oa)
                        if res_lp:
                            cx, cy, side = res_lp
                            if side < best_side:
                                best_side = side
                                best_inner = new_inner
                                best_center = (cx, cy)
                                best_outer_angle = oa
                                x0 = new_params
            except Exception:
                pass

    # Phase 4: Final Polish with very fine outer angle sweep
    if best_inner is not None and time.time() < _DEADLINE - 5.0:
        all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in best_inner])
        
        best_polish_side = best_side
        best_polish_center = best_center
        best_polish_angle = best_outer_angle
        
        for outer_angle_deg in np.arange(best_outer_angle - 3, best_outer_angle + 3, 0.05):
            if time.time() > _DEADLINE - 5.0:
                break
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_polish_side:
                    best_polish_side = side
                    best_polish_center = (cx, cy)
                    best_polish_angle = outer_angle_deg
        
        best_side = best_polish_side
        best_center = best_polish_center
        best_outer_angle = best_polish_angle

    # Phase 5: Second round of gradient refinement from the polished result
    if best_inner is not None and time.time() < _DEADLINE - 3.0:
        x0 = []
        for i in range(N):
            x0.append(best_inner[i][0])
            x0.append(best_inner[i][1])
            x0.append(best_inner[i][2])
        x0 = np.array(x0)
        
        oa = best_outer_angle
        def obj2(params):
            return _objective_smooth(params, oa, beta=100.0)
        
        try:
            res_opt2 = minimize(obj2, x0, method='L-BFGS-B', 
                                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_opt2.success or res_opt2.fun < best_side:
                new_params = res_opt2.x
                new_inner = []
                for i in range(N):
                    new_inner.append([new_params[3*i], new_params[3*i+1], new_params[3*i+2]])
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        p1 = np.array([new_inner[i][0], new_inner[i][1]])
                        p2 = np.array([new_inner[j][0], new_inner[j][1]])
                        a1 = math.radians(new_inner[i][2])
                        a2 = math.radians(new_inner[j][2])
                        dist = _hex_distance_exact(p1, a1, p2, a2)
                        if dist < -1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in new_inner])
                    res_lp = _min_enclosing_hex_side_lp(all_vertices, oa)
                    if res_lp:
                        cx, cy, side = res_lp
                        if side < best_side:
                            best_side = side
                            best_inner = new_inner
                            best_center = (cx, cy)
        except Exception:
            pass

    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
