<<<<<<< SEARCH
def joint_optimize(init_inner, init_outer_center, init_outer_angle, deadline):
    """Joint optimization of inner positions/angles and outer hexagon."""
    def obj_joint_smooth(params, beta=80.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 500.0 * pen
    
    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 3000.0 * pen
    
    # Build initial params
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_outer_center[0]
    p0[N * 3 + 1] = init_outer_center[1]
    p0[N * 3 + 2] = init_outer_angle
    
    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val
    
    # Simulated annealing with adaptive step sizes
    T = 0.5
    T_min = 0.001
    cool = 0.9995
    max_iter = 12000
    np.random.seed(123)
    step_pos = 0.3
    step_ang = 12.0
    step_oc = 0.2
    step_oa = 5.0
    
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.60:
            # perturb one inner hexagon position
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, step_pos)
        elif r < 0.78:
            # perturb one inner hexagon angle
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, step_ang)
        elif r < 0.88:
            # perturb outer center
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, step_oc)
        else:
            # perturb outer angle
            nb[N * 3 + 2] += np.random.normal(0, step_oa)
        
        nb_val = obj_joint_smooth(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
        T *= cool
        if T < T_min:
            break
    
    # Extract best
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])
    
    return best_inner_j, best_oc_j, best_oa_j, best_params
=======
def joint_optimize(init_inner, init_outer_center, init_outer_angle, deadline):
    """Joint optimization of inner positions/angles and outer hexagon."""
    normals = _NORMALS
    
    def obj_joint_smooth(params, beta=80.0):
        # Fast inline computation
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        oa_rad = math.radians(oa)
        cos_o = math.cos(oa_rad)
        sin_o = math.sin(oa_rad)
        
        n_projs = N * 6
        P = np.empty(n_projs)
        idx = 0
        for i in range(N):
            ix = params[i * 3]
            iy = params[i * 3 + 1]
            iang = math.radians(params[i * 3 + 2])
            vx = _HEX_COS * math.cos(iang) - _HEX_SIN * math.sin(iang)
            vy = _HEX_COS * math.sin(iang) + _HEX_SIN * math.cos(iang)
            dx = ix - ocx
            dy = iy - ocy
            lx = cos_o * dx - sin_o * dy + cos_o * vx - sin_o * vy
            ly = sin_o * dx + cos_o * dy + sin_o * vx + cos_o * vy
            P[idx:idx+6] = lx * normals[:, 0] + ly * normals[:, 1]
            idx += 6
        
        m = P.max()
        if m < -1e6:
            smooth_max = 0.0
        else:
            smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
            if smooth_max < 0:
                smooth_max = 0.0
        side = 2.0 * smooth_max / SQRT3
        
        # Overlap penalty
        pen = 0.0
        min_dist = SQRT3 - 0.002
        for i in range(N):
            xi = params[i * 3]
            yi = params[i * 3 + 1]
            for j in range(i + 1, N):
                xj = params[j * 3]
                yj = params[j * 3 + 1]
                d = math.hypot(xi - xj, yi - yj)
                if d < min_dist:
                    pen += (min_dist - d) ** 2
        return side + 500.0 * pen
    
    def obj_joint_exact(params):
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        oa_rad = math.radians(oa)
        cos_o = math.cos(oa_rad)
        sin_o = math.sin(oa_rad)
        
        max_proj = -np.inf
        for i in range(N):
            ix = params[i * 3]
            iy = params[i * 3 + 1]
            iang = math.radians(params[i * 3 + 2])
            vx = _HEX_COS * math.cos(iang) - _HEX_SIN * math.sin(iang)
            vy = _HEX_COS * math.sin(iang) + _HEX_SIN * math.cos(iang)
            dx = ix - ocx
            dy = iy - ocy
            lx = cos_o * dx - sin_o * dy + cos_o * vx - sin_o * vy
            ly = sin_o * dx + cos_o * dy + sin_o * vx + cos_o * vy
            projs = lx * normals[:, 0] + ly * normals[:, 1]
            cur = projs.max()
            if cur > max_proj:
                max_proj = cur
        if max_proj < 0:
            max_proj = 0.0
        side = 2.0 * max_proj / SQRT3
        
        pen = 0.0
        min_dist = SQRT3 - 0.002
        for i in range(N):
            xi = params[i * 3]
            yi = params[i * 3 + 1]
            for j in range(i + 1, N):
                xj = params[j * 3]
                yj = params[j * 3 + 1]
                d = math.hypot(xi - xj, yi - yj)
                if d < min_dist:
                    pen += (min_dist - d) ** 2
        return side + 3000.0 * pen
    
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_outer_center[0]
    p0[N * 3 + 1] = init_outer_center[1]
    p0[N * 3 + 2] = init_outer_angle
    
    current = p0.copy()
    current_val = obj_joint_smooth(current)
    best_params = current.copy()
    best_val_j = current_val
    
    # Simulated annealing with adaptive step sizes - use time-based seed for diversity
    T = 0.5
    T_min = 0.001
    cool = 0.9995
    max_iter = 15000
    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
    step_pos = 0.3
    step_ang = 12.0
    step_oc = 0.2
    step_oa = 5.0
    
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        nb = current.copy()
        r = rng.random()
        if r < 0.60:
            idx = rng.randint(0, N)
            dim = rng.randint(0, 2)
            nb[idx * 3 + dim] += rng.normal(0, step_pos)
        elif r < 0.78:
            idx = rng.randint(0, N)
            nb[idx * 3 + 2] += rng.normal(0, step_ang)
        elif r < 0.88:
            dim = rng.randint(0, 2)
            nb[N * 3 + dim] += rng.normal(0, step_oc)
        else:
            nb[N * 3 + 2] += rng.normal(0, step_oa)
        
        nb_val = obj_joint_smooth(nb)
        if nb_val < current_val or rng.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
        T *= cool
        if T < T_min:
            break
    
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])
    
    return best_inner_j, best_oc_j, best_oa_j, best_params
>>>>>>> REPLACE

