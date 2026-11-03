<<<<<<< SEARCH
def _nelder_mead_polish(positions, angles, max_iter=200, tol=1e-6):
    """
    Polish the inner positions using Nelder-Mead.
    Returns improved positions and angles.
    """
    from scipy.optimize import minimize
    
    # Flatten parameters: x0, y0, x1, y1, ..., angle0, angle1, ...
    x0 = []
    for px, py in positions:
        x0.extend([px, py])
    x0.extend(angles)
    
    def objective(params):
        n = len(positions)
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        ang = list(params[2*n:])
        
        # Check disjointness
        if not _check_disjoint(pos, ang):
            return 1e6
        
        all_verts = _compute_all_vertices(pos, ang)
        
        # Find best outer angle
        best_side = float('inf')
        for outer_angle in np.arange(0, 30.1, 2.0):
            res = _find_min_enclosing_hex_side_lp(all_verts, outer_angle)
            if res:
                side = res[2]
                if side < best_side:
                    best_side = side
        
        return best_side
    
    # Use Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if res.success:
            n = len(positions)
            new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            new_ang = list(res.x[2*n:])
            return new_pos, new_ang
    except:
        pass
    
    return positions, angles
=======
def _smooth_enclosing_hex_side(all_verts, outer_angle_deg):
    """
    Find minimum side length of enclosing regular hexagon with given angle.
    Returns (center_x, center_y, side_length) or (inf, 0, 0) if infeasible.
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in all_verts:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
    
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b,
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return float('inf'), 0.0, 0.0
    except:
        return float('inf'), 0.0, 0.0

def _nelder_mead_polish(positions, angles, max_iter=200, tol=1e-6):
    """
    Polish the inner positions using Nelder-Mead with smooth objective.
    Uses epigraph formulation with soft constraints.
    """
    n = len(positions)
    
    # Flatten parameters: x0, y0, x1, y1, ..., angle0, angle1, ...
    x0 = []
    for px, py in positions:
        x0.extend([px, py])
    x0.extend(angles)
    
    # Precompute best outer angle from initial config
    all_verts_init = _compute_all_vertices(positions, angles)
    best_init_angle = 0.0
    best_init_side = float('inf')
    for outer_angle in np.arange(0, 30.1, 1.0):
        res = _smooth_enclosing_hex_side(all_verts_init, outer_angle)
        if res[2] < best_init_side:
            best_init_side = res[2]
            best_init_angle = outer_angle
    
    def objective(params):
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        ang = list(params[2*n:])
        
        # Smooth disjointness penalty using minimum separation
        all_verts = []
        for i in range(n):
            px, py = pos[i]
            ang_i = ang[i]
            all_verts.append(_get_hex_vertices(px, py, ang_i))
        
        # Compute minimum separation between all pairs
        min_sep = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                sep = _hex_min_separation(all_verts[i], all_verts[j])
                if sep < min_sep:
                    min_sep = sep
        
        # Soft penalty for overlap (smooth)
        penalty = max(0, -min_sep) ** 2 * 100.0
        
        all_verts_flat = _compute_all_vertices(pos, ang)
        
        # Find best outer side using a few candidate angles
        best_side = float('inf')
        for outer_angle in [best_init_angle]:
            res = _smooth_enclosing_hex_side(all_verts_flat, outer_angle)
            if res[2] < best_side:
                best_side = res[2]
        
        return best_side + penalty
    
    # Use Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if res.success:
            new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            new_ang = list(res.x[2*n:])
            # Verify disjointness before accepting
            if _check_disjoint(new_pos, new_ang):
                return new_pos, new_ang
    except:
        pass
    
    return positions, angles

def _high_precision_polish(positions, angles, outer_angle_deg, max_iter=500):
    """
    Final high-precision polish using SLSQP with tight tolerances.
    Optimizes positions and angles simultaneously.
    """
    n = len(positions)
    
    x0 = []
    for px, py in positions:
        x0.extend([px, py])
    x0.extend(angles)
    
    def objective(params):
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        ang = list(params[2*n:])
        
        # Check disjointness
        if not _check_disjoint(pos, ang):
            return 1e10
        
        all_verts = _compute_all_vertices(pos, ang)
        res = _smooth_enclosing_hex_side(all_verts, outer_angle_deg)
        return res[2]
    
    try:
        res = minimize(objective, x0, method='SLSQP',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
        if res.success:
            new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            new_ang = list(res.x[2*n:])
            if _check_disjoint(new_pos, new_ang):
                return new_pos, new_ang
    except:
        pass
    
    return positions, angles
>>>>>>> REPLACE

