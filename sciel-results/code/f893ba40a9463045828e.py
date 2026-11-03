import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 142.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _hex_support_vectorized(angle1_deg, angle2_deg, theta_arr):
    """Vectorized support function for two hexagons at given angles for array of directions."""
    a1 = math.radians(angle1_deg)
    a2 = math.radians(angle2_deg)
    pi3 = math.pi / 3.0
    
    def _support_single(a0, theta):
        diff = (theta - a0) % (2 * math.pi)
        k = np.round(diff / pi3)
        d1 = np.abs(diff - k * pi3)
        d2 = np.abs(diff - (k - 1) * pi3)
        d3 = np.abs(diff - (k + 1) * pi3)
        min_d = np.minimum(np.minimum(d1, d2), d3)
        return np.cos(min_d)
    
    s1 = _support_single(a1, theta_arr)
    s2 = _support_single(a2, theta_arr)
    return s1, s2


def _hex_distance_fast(d, angle1, angle2):
    """Fast minimum distance between two unit hexagons. Returns negative if overlapping."""
    a1 = math.radians(angle1)
    a2 = math.radians(angle2)
    dx, dy = d[0], d[1]
    
    pi3 = math.pi / 3.0
    pi6 = math.pi / 6.0
    
    # All 12 edge normals
    normals1 = a1 + pi6 + np.arange(6) * pi3
    normals2 = a2 + pi6 + np.arange(6) * pi3
    all_normals = np.concatenate([normals1, normals2])
    
    # Compute projections
    cos_n = np.cos(all_normals)
    sin_n = np.sin(all_normals)
    proj = np.abs(dx * cos_n + dy * sin_n)
    
    # Compute supports
    s1, s2 = _hex_support_vectorized(angle1, angle2, all_normals)
    
    sep = proj - s1 - s2
    return float(np.min(sep))





def _enclosing_hex_side(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon with given orientation containing all vertices."""
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
            return side
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
    return side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
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
        # Row-based: 3-4-4 pattern
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
        # 4-3-4 pattern
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
        # 2-4-2-3 pattern
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
        # 3-5-3 pattern
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
        # Center + 6 ring + 4 more
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
        # 2-3-3-3 pattern
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
        # 4-4-3 pattern
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
        # 5-6 pattern
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
        # 6-5 pattern
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
        # Diamond-ish
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
        # 3-3-5 pattern
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
        # Fallback: compact grid
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]


def _objective_fast(params, inner_angles):
    """Objective: enclosing hex side length + penalty for overlaps. Optimized."""
    n_pos = N * 2
    positions = [(params[i], params[i+1]) for i in range(0, n_pos, 2)]
    outer_angle = params[-1]
    
    # Get all vertices
    all_vertices = []
    for idx, (px, py) in enumerate(positions):
        ia = inner_angles[idx] if idx < len(inner_angles) else 0.0
        verts = _hex_vertices(px, py, ia)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Enclosing hex side
    side = _enclosing_hex_side(all_vertices, outer_angle)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            ia_i = inner_angles[i] if i < len(inner_angles) else 0.0
            ia_j = inner_angles[j] if j < len(inner_angles) else 0.0
            di = np.array([positions[j][0] - positions[i][0], positions[j][1] - positions[i][1]])
            dist = _hex_distance_fast(di, ia_i, ia_j)
            if dist < 0:
                # Stronger penalty to prevent overlap
                penalty += dist * dist * 10000.0
    
    return side + penalty


def _optimize_layout(positions, inner_angle, outer_angles, inner_angle_candidates=None):
    """Optimize a layout by trying different outer angles and refining positions."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    
    if inner_angle_candidates is None:
        inner_angle_candidates = [inner_angle]
    
    # Try different inner angle configurations
    for ia in inner_angle_candidates:
        for outer_angle in outer_angles:
            all_vertices = np.vstack([_hex_vertices(px, py, ia) for px, py in positions])
            side = _enclosing_hex_side(all_vertices, outer_angle)
            
            if side < best_side:
                best_side = side
                best_pos = positions
                best_outer = outer_angle
    
    # Try to optimize positions with Nelder-Mead
    if time.time() < _DEADLINE - 15.0:
        x0 = np.array([positions[i][0] for i in range(N)] + 
                      [positions[i][1] for i in range(N)] + [best_outer])
        
        def obj(params):
            return _objective_fast(params, [inner_angle]*N)
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res.x[-1]
        except Exception:
            pass
    
    return best_side, best_pos, best_outer


def _is_valid(inner_config):
    """Check if the configuration has no overlaps."""
    for i in range(N):
        for j in range(i+1, N):
            p1 = np.array([inner_config[i][0], inner_config[i][1]])
            p2 = np.array([inner_config[j][0], inner_config[j][1]])
            d = p2 - p1
            dist = _hex_distance_fast(d, inner_config[i][2], inner_config[j][2])
            if dist < -1e-6: # Small tolerance for numerical error
                return False
    return True

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    best_valid_side = float('inf')
    best_valid_inner = None
    best_valid_center = (0.0, 0.0)
    best_valid_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Outer angles to try
    outer_angles = list(np.arange(0, 30.0, 1.0))
    
    # Inner angle candidates: allow rotation of inner hexagons
    inner_angle_candidates = [0.0, 15.0, 30.0, 45.0, 60.0]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # First pass: try discrete outer angles with multiple inner angles
        best_local_side = float('inf')
        best_local_pos = positions
        best_local_angle = 0.0
        best_local_inner_angle = 0.0
        
        for ia in inner_angle_candidates:
            for outer_angle_deg in outer_angles:
                all_vertices = np.vstack([_hex_vertices(px, py, ia) for px, py in positions])
                side = _enclosing_hex_side(all_vertices, outer_angle_deg)
                if side < best_local_side:
                    best_local_side = side
                    best_local_pos = positions
                    best_local_angle = outer_angle_deg
                    best_local_inner_angle = ia
        
        # Second pass: refine with local optimization
        if time.time() < _DEADLINE - 15.0:
            refined_side, refined_pos, refined_angle = _optimize_layout(
                best_local_pos, best_local_inner_angle, 
                list(np.arange(best_local_angle - 5, best_local_angle + 5, 0.5)))
            
            if refined_side < best_local_side:
                best_local_side = refined_side
                best_local_pos = refined_pos
                best_local_angle = refined_angle
        
        # Check validity of the local best
        local_inner = [[pos[0], pos[1], best_local_inner_angle] for pos in best_local_pos]
        if _is_valid(local_inner):
            if best_local_side < best_valid_side:
                best_valid_side = best_local_side
                best_valid_inner = local_inner
                best_valid_outer_angle = best_local_angle
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = local_inner
            best_center = (0.0, 0.0)  # Will be recomputed
            best_outer_angle = best_local_angle
    
    # Use the best valid configuration if available
    if best_valid_inner is not None:
        best_inner = best_valid_inner
        best_side = best_valid_side
        best_outer_angle = best_valid_outer_angle
    
    # Compute proper center for the best configuration
    if best_inner is not None:
        all_vertices = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
        # Find center via LP
        base_angle = math.radians(best_outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
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
                best_center = (res.x[0], res.x[1])
                # Only update side if it's consistent with validity (it should be)
                # The LP finds the minimal enclosing hex, so this side is accurate for this geometry
                best_side = res.x[2] * 2.0 / _SQRT3
        except Exception:
            pass
        
        # Final high-precision polish: tighten the LP solution
        # We only do this if we have time and the current solution is valid
        if time.time() < _DEADLINE - 5.0 and best_inner is not None and _is_valid(best_inner):
            # Use the LP center as starting point and do a final Nelder-Mead pass
            x0 = np.array([p[0] for p in best_inner] + 
                          [p[1] for p in best_inner] + [best_outer_angle])
            
            def final_obj(params):
                n_pos = N * 2
                positions = [(params[i], params[i+1]) for i in range(0, n_pos, 2)]
                oa = params[-1]
                all_v = np.vstack([_hex_vertices(px, py, p[2]) for (px, py), p in zip(positions, best_inner)])
                side = _enclosing_hex_side(all_v, oa)
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        di = np.array([positions[j][0]-positions[i][0], positions[j][1]-positions[i][1]])
                        d = _hex_distance_fast(di, best_inner[i][2], best_inner[j][2])
                        if d < 0:
                            penalty += d*d*10000.0
                return side + penalty
            
            try:
                res2 = minimize(final_obj, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                
                # Check if the optimized result is valid
                candidate_inner = [[res2.x[i], res2.x[N+i], best_inner[i][2]] for i in range(N)]
                candidate_outer_angle = res2.x[-1]
                
                if _is_valid(candidate_inner) and res2.fun < best_side:
                    best_side = res2.fun
                    best_inner = candidate_inner
                    best_outer_angle = candidate_outer_angle
                    # Recompute center
                    all_v2 = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
                    try:
                        res3 = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
                        if res3.status == 0:
                            best_center = (res3.x[0], res3.x[1])
                            best_side = res3.x[2] * 2.0 / _SQRT3
                    except Exception:
                        pass
            except Exception:
                pass
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final safety check: if somehow invalid, return a known valid sparse layout
    if not _is_valid(best_inner):
        # Fallback to a very sparse layout that is guaranteed valid
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
