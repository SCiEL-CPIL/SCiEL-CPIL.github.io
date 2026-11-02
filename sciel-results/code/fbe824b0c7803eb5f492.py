def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, S3/2])

    def bary_to_cart(bary_coords):
        u = bary_coords[0::2]
        v = bary_coords[1::2]
        w = 1.0 - u - v
        if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
            return None
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        s = u + v + w
        if s < 1e-12:
            return None
        u /= s
        v /= s
        w /= s
        return u[:, np.newaxis] * a + v[:, np.newaxis] * b + w[:, np.newaxis] * c

    # Initialize with a robust 11-point configuration
    candidates = []
    for i in range(5):
        for j in range(5):
            u = i / 4.0 * 0.9
            v = j / 4.0 * 0.9
            if u + v <= 1.0:
                candidates.append([u, v])
    
    if len(candidates) < N:
        while len(candidates) < N:
            candidates.append([0.5, 0.5])
            
    init_bary = np.array(candidates[:N]).flatten()
    points = bary_to_cart(init_bary)
    if points is None:
        # Fallback to a simple valid grid
        best_result = np.zeros((N, 2))
        for i in range(N):
            u, v = candidates[i]
            w = 1 - u - v
            best_result[i] = u * a + v * b + w * c
        return best_result

    current_score = compute_min_area(points)
    best_points = points.copy()
    best_score = current_score

    # Heuristic Local Search
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 10:
        # Find the minimum area triangle
        min_area = np.inf
        min_indices = (0, 1, 2)
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    area = 0.5 * abs((points[j][0]-points[i][0])*(points[k][1]-points[i][1]) - (points[k][0]-points[i][0])*(points[j][1]-points[i][1]))
                    if area < min_area:
                        min_area = area
                        min_indices = (i, j, k)
        
        # Pick a random vertex from the min triangle to optimize
        idx = rng.choice(min_indices)
        
        # Optimize point idx using Nelder-Mead
        def obj_single(x_bary):
            new_points = points.copy()
            u = x_bary[0]
            v = x_bary[1]
            w = 1 - u - v
            if u < 0 or v < 0 or w < 0:
                return 1e10
            new_points[idx] = u * a + v * b + w * c
            return -compute_min_area(new_points)
        
        # Current bary coords for idx
        p = points[idx]
        w_curr = p[1] / (S3/2)
        v_curr = p[0] - w_curr*0.5
        u_curr = 1 - v_curr - w_curr
        x0 = np.array([u_curr, v_curr])
        
        try:
            res = minimize(obj_single, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < -current_score: # Improvement
                new_bary = res.x
                u, v = new_bary
                w = 1 - u - v
                if u >= 0 and v >= 0 and w >= 0:
                    points[idx] = u * a + v * b + w * c
                    new_score = compute_min_area(points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = points.copy()
                    current_score = new_score
        except Exception:
            pass

    return best_points
