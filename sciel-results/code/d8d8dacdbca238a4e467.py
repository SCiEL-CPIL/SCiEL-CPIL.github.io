# ... (imports)

def _hexagon_overlap(v1, v2):
    # v1, v2 are (6,2) arrays
    min_overlap_depth = float('inf')
    for i in range(6):
        for poly_idx in range(2):
            if poly_idx == 0:
                poly = v1
            else:
                poly = v2
            p1 = poly[i]
            p2 = poly[(i+1)%6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9: continue
            normal = normal / norm_len
            
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return 0.0
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap_depth:
                min_overlap_depth = overlap
                
    return min_overlap_depth

def _objective(params):
    n_params = 3 * N + 1
    if len(params) != n_params:
        return 1e6
    
    positions = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        th = params[3*i+2]
        positions.append((x, y))
        angles.append(th)
    
    phi_outer = params[-1]
    
    verts_list = []
    for i in range(N):
        v = _hex_vertices(positions[i][0], positions[i][1], angles[i])
        verts_list.append(v)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _hexagon_overlap(verts_list[i], verts_list[j])
            if d > 0:
                penalty += d * d
    
    if penalty > 1.0:
        return 100.0 + penalty
    
    all_verts = np.vstack(verts_list)
    res = _min_enclosing_hex_side_lp(all_verts, phi_outer)
    if res is None:
        return 1e6
    
    cx, cy, side = res
    return side + 100.0 * penalty

def _optimize_layout(initial_positions, initial_angles, initial_outer_angle):
    init_params = []
    for i in range(N):
        init_params.append(initial_positions[i][0])
        init_params.append(initial_positions[i][1])
        init_params.append(initial_angles[i])
    init_params.append(initial_outer_angle)
    
    try:
        res = minimize(_objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 100, 'maxfev': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success:
            best_params = res.x
            side = _objective(best_params)
            if side < 100: # Valid solution
                inner = []
                for i in range(N):
                    x = best_params[3*i]
                    y = best_params[3*i+1]
                    th = best_params[3*i+2]
                    inner.append([x, y, th])
                outer_angle = best_params[-1]
                verts_list = [_hex_vertices(inner[i][0], inner[i][1], inner[i][2]) for i in range(N)]
                all_verts = np.vstack(verts_list)
                res_lp = _min_enclosing_hex_side_lp(all_verts, outer_angle)
                if res_lp:
                    cx, cy, side_lp = res_lp
                    return inner, (cx, cy), side_lp, outer_angle
    except Exception:
        pass
    
    return None
