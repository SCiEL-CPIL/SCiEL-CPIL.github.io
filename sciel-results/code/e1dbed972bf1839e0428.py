import numpy as np
from scipy.optimize import minimize

N = 21

def _make_grid_layout(w, h, n):
    """Generate a grid layout with n circles in rectangle w x h."""
    aspect = w / h if h > 0 else 1.0
    # Find best grid dimensions
    best_c, best_r = 1, n
    best_min_dim = 0
    for c in range(1, n + 1):
        r = int(np.ceil(n / c))
        cw = w / c
        ch = h / r
        min_dim = min(cw, ch)
        if min_dim > best_min_dim:
            best_min_dim = min_dim
            best_c, best_r = c, r
    
    c, r = best_c, best_r
    cw = w / c
    ch = h / r
    
    positions = []
    radii = []
    for i in range(c):
        for j in range(r):
            if len(positions) >= n:
                break
            x = (i + 0.5) * cw
            y = (j + 0.5) * ch
            positions.append((x, y))
            r_val = 0.48 * min(cw, ch)
            radii.append(r_val)
    
    while len(positions) < n:
        positions.append((w/2, h/2))
        radii.append(0.05)
    
    return positions, radii

def _objective(x, w):
    h = 2.0 - w
    if w <= 0 or h <= 0:
        return 1e8
    radii_sum = 0.0
    for i in range(N):
        r = x[3 + 3*i]
        if r < 0:
            return 1e8
        radii_sum += r
    return -radii_sum

def _constraints_vec(x, w):
    h = 2.0 - w
    if w <= 0 or h <= 0:
        n_pairs = N*(N-1)//2
        return np.full(4*N + n_pairs, -1e6)
    
    cons = []
    # Wall constraints
    for i in range(N):
        xi = x[1 + 3*i]
        yi = x[2 + 3*i]
        ri = x[3 + 3*i]
        if ri < 0:
            cons.extend([-1e6, -1e6, -1e6, -1e6])
            continue
        cons.append(xi - ri)
        cons.append(w - xi - ri)
        cons.append(yi - ri)
        cons.append(h - yi - ri)
    
    # Pair constraints (use squared distance)
    for i in range(N):
        xi = x[1 + 3*i]
        yi = x[2 + 3*i]
        ri = x[3 + 3*i]
        for j in range(i+1, N):
            xj = x[1 + 3*j]
            yj = x[2 + 3*j]
            rj = x[3 + 3*j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            sum_r_sq = (ri + rj)**2
            cons.append(dist_sq - sum_r_sq)
    
    return np.array(cons)

def _validate(x, w):
    """Check if solution is valid (all constraints satisfied)."""
    h = 2.0 - w
    if w <= 0 or h <= 0:
        return False
    for i in range(N):
        xi = x[1 + 3*i]
        yi = x[2 + 3*i]
        ri = x[3 + 3*i]
        if ri < 0:
            return False
        if xi < ri - 1e-9 or xi > w - ri + 1e-9:
            return False
        if yi < ri - 1e-9 or yi > h - ri + 1e-9:
            return False
    for i in range(N):
        xi = x[1 + 3*i]
        yi = x[2 + 3*i]
        ri = x[3 + 3*i]
        for j in range(i+1, N):
            xj = x[1 + 3*j]
            yj = x[2 + 3*j]
            rj = x[3 + 3*j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _sum_radii(x):
    return sum(x[3 + 3*i] for i in range(N))

def construct():
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.5, 1.5, 0.7, 1.3, 0.8, 1.25, 0.6, 1.4, 0.9, 1.1]
    
    for w_init in aspect_ratios:
        if w_init <= 0 or w_init >= 2:
            continue
        h_init = 2.0 - w_init
        
        # Generate initial grid layout
        positions, radii = _make_grid_layout(w_init, h_init, N)
        
        # Build initial x vector
        x0 = [w_init]
        for i in range(N):
            x0.append(positions[i][0])
            x0.append(positions[i][1])
            x0.append(radii[i])
        x0 = np.array(x0)
        
        # Define objective and constraints as closures
        def obj(x):
            w = x[0]
            h = 2.0 - w
            if w <= 0 or h <= 0:
                return 1e8
            radii_sum = 0.0
            for i in range(N):
                r = x[3 + 3*i]
                if r < 0:
                    return 1e8
                radii_sum += r
            return -radii_sum
        
        def con_func(x):
            w = x[0]
            h = 2.0 - w
            if w <= 0 or h <= 0:
                n_pairs = N*(N-1)//2
                return np.full(4*N + n_pairs, -1e6)
            
            cons = []
            for i in range(N):
                xi = x[1 + 3*i]
                yi = x[2 + 3*i]
                ri = x[3 + 3*i]
                if ri < 0:
                    cons.extend([-1e6, -1e6, -1e6, -1e6])
                    continue
                cons.append(xi - ri)
                cons.append(w - xi - ri)
                cons.append(yi - ri)
                cons.append(h - yi - ri)
            
            for i in range(N):
                xi = x[1 + 3*i]
                yi = x[2 + 3*i]
                ri = x[3 + 3*i]
                for j in range(i+1, N):
                    xj = x[1 + 3*j]
                    yj = x[2 + 3*j]
                    rj = x[3 + 3*j]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r_sq = (ri + rj)**2
                    cons.append(dist_sq - sum_r_sq)
            
            return np.array(cons)
        
        bounds = [(1e-6, 2-1e-6)]
        for i in range(N):
            bounds.append((0, 2))
            bounds.append((0, 2))
            bounds.append((1e-6, 1.0))
        
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, 
                          constraints=con_func, 
                          options={'maxiter': 800, 'ftol': 1e-10, 'disp': False})
            
            if res.success:
                # Validate and get value
                w_opt = res.x[0]
                if _validate(res.x, w_opt):
                    val = _sum_radii(res.x)
                    if val > best_value:
                        best_value = val
                        best_result = res.x.copy()
                else:
                    # Try to fix small violations by slightly reducing radii
                    x_fixed = res.x.copy()
                    w_fix = x_fixed[0]
                    h_fix = 2.0 - w_fix
                    valid = True
                    for i in range(N):
                        idx_r = 3 + 3*i
                        x_idx = 1 + 3*i
                        y_idx = 2 + 3*i
                        ri = x_fixed[idx_r]
                        # Check walls
                        if x_fixed[x_idx] < ri:
                            x_fixed[idx_r] = max(0, x_fixed[x_idx] - 1e-6)
                        if w_fix - x_fixed[x_idx] < ri:
                            x_fixed[idx_r] = min(x_fixed[idx_r], max(0, w_fix - x_fixed[x_idx] - 1e-6))
                        if x_fixed[y_idx] < ri:
                            x_fixed[idx_r] = min(x_fixed[idx_r], max(0, x_fixed[y_idx] - 1e-6))
                        if h_fix - x_fixed[y_idx] < ri:
                            x_fixed[idx_r] = min(x_fixed[idx_r], max(0, h_fix - x_fixed[y_idx] - 1e-6))
                    
                    # Check pairs and reduce if needed
                    for i in range(N):
                        xi = x_fixed[1 + 3*i]
                        yi = x_fixed[2 + 3*i]
                        ri = x_fixed[3 + 3*i]
                        for j in range(i+1, N):
                            xj = x_fixed[1 + 3*j]
                            yj = x_fixed[2 + 3*j]
                            rj = x_fixed[3 + 3*j]
                            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                            if dist < ri + rj:
                                # Reduce both radii slightly
                                excess = ri + rj - dist
                                x_fixed[3 + 3*i] = max(1e-6, ri - excess/2)
                                x_fixed[3 + 3*j] = max(1e-6, rj - excess/2)
                    
                    if _validate(x_fixed, w_fix):
                        val = _sum_radii(x_fixed)
                        if val > best_value:
                            best_value = val
                            best_result = x_fixed.copy()
        except Exception:
            pass

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Extract result
    circles = []
    for i in range(N):
        x = best_result[1 + 3*i]
        y = best_result[2 + 3*i]
        r = best_result[3 + 3*i]
        circles.append([x, y, r])
    
    return np.array(circles)
