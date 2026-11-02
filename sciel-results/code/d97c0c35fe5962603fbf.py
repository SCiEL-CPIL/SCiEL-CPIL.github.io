<<<<<<< SEARCH
# EVOLVE-BLOCK-START

def _elapsed(global_start):
    return time.time() - global_start

def _remaining(global_start, deadline):
    return deadline - _elapsed(global_start)

def _softplus(x, beta=100.0):
    """
    Smooth approximation of max(0, x).
    Uses x/2 + sqrt(x^2 + 1/beta^2)/2 for stability and speed.
    """
    return 0.5 * (x + np.sqrt(x * x + 1.0 / (beta * beta)))

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _fix_boundaries(xyz):
    """Ensure circles are within boundaries by clipping radii and positions."""
    xyz_fixed = xyz.copy()
    for _ in range(50):
        if _validate(xyz_fixed):
            break
        # Reduce radii slightly
        xyz_fixed[:, 2] *= 0.95
        # Clip positions
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def _objective_penalty(params, beta=50.0, penalty_weight=100.0):
    """
    Objective function with smooth penalty for violations.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    b_left = - (x - r)
    b_right = - (1 - x - r)
    b_bottom = - (y - r)
    b_top = - (1 - y - r)
    b_rad = -r
    
    penalty += np.sum(_softplus(b_left, beta))
    penalty += np.sum(_softplus(b_right, beta))
    penalty += np.sum(_softplus(b_bottom, beta))
    penalty += np.sum(_softplus(b_top, beta))
    penalty += np.sum(_softplus(b_rad, beta))
    
    # Overlap penalties (all pairs)
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, beta)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_delaunay(params, beta=50.0, penalty_weight=100.0):
    """
    Objective function using Delaunay triangulation for overlap constraints.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    b_left = - (x - r)
    b_right = - (1 - x - r)
    b_bottom = - (y - r)
    b_top = - (1 - y - r)
    b_rad = -r
    
    penalty += np.sum(_softplus(b_left, beta))
    penalty += np.sum(_softplus(b_right, beta))
    penalty += np.sum(_softplus(b_bottom, beta))
    penalty += np.sum(_softplus(b_top, beta))
    penalty += np.sum(_softplus(b_rad, beta))
    
    # Get Delaunay edges
    centers = np.column_stack([x, y])
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
    except Exception:
        # Fallback to all pairs if Delaunay fails
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Overlap penalties for Delaunay neighbors
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        penalty += _softplus(overlap, beta)
    
    return -np.sum(r) + penalty_weight * penalty

def _get_contact_edges(centers, r):
    """Get edges for circles that are close enough to potentially overlap."""
    try:
        tree = cKDTree(centers)
        max_r = np.max(r)
        query_dist = 2.0 * max_r + 0.01
        pairs = tree.query_pairs(query_dist, output_type='set')
        return list(pairs)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _initial_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_corner_weighted():
    """Initial configuration with circles weighted towards corners."""
    positions = []
    # Corner circles
    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.12])
    
    # Edge midpoints
    edges = [[0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
    for ex, ey in edges:
        positions.append([ex, ey, 0.10])
    
    # Center
    positions.append([0.5, 0.5, 0.11])
    
    # Fill remaining with random positions
    np.random.seed(123)
    for _ in range(N - len(positions)):
        x = np.random.uniform(0.2, 0.8)
        y = np.random.uniform(0.2, 0.8)
        r = np.random.uniform(0.06, 0.10)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        positions.append([x, y, r])
    
    return np.array(positions[:N])

def _optimize_single(xyz0, global_start, deadline, max_iter=2000):
    """
    Optimize from a single initial configuration with adaptive penalty annealing.
    """
    params = xyz0.flatten()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Adaptive penalty annealing
    beta_start = 10.0
    beta_end = 200.0
    weight_start = 10.0
    weight_end = 500.0
    
    n_phases = 5
    for phase in range(n_phases):
        if _remaining(global_start, deadline) < 2.0:
            break
        
        t = phase / max(1, n_phases - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        weight = weight_start * (weight_end / weight_start) ** t
        
        try:
            result = minimize(_objective_delaunay, best_params, method='L-BFGS-B',
                              bounds=bounds, args=(beta, weight),
                              options={'maxiter': max_iter // n_phases, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < _objective_delaunay(best_params, beta, weight):
                best_params = result.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass
    
    # Final polish with full pairwise constraints
    if _remaining(global_start, deadline) > 3.0:
        try:
            result = minimize(_objective_penalty, best_params, method='L-BFGS-B',
                              bounds=bounds, args=(200.0, 500.0),
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < _objective_penalty(best_params, 200.0, 500.0):
                best_params = result.x
                best_score = np.sum(best_params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_initial_grid())
    initial_configs.append(_initial_hex())
    initial_configs.append(_initial_corner_weighted())
    
    # Random configurations
    np.random.seed(42)
    for trial in range(6):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.06, 0.14, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # Concentric rings pattern
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.12 * 1.8 * np.cos(angle)
        y = 0.5 + 0.12 * 1.8 * np.sin(angle)
        positions.append([x, y, 0.12 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.10 * 3.5 * np.cos(angle)
        y = 0.5 + 0.10 * 3.5 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(7):
        angle = 2 * np.pi * k / 7
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    initial_configs.append(np.array(positions[:N]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        if _remaining(global_start, deadline) < 5.0:
            break
        
        xyz_opt = _optimize_single(xyz0, global_start, deadline)
        
        # Fix if needed
        if not _validate(xyz_opt):
            xyz_opt = _fix_boundaries(xyz_opt)
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation and fix
    if not _validate(best_xyz):
        best_xyz = _fix_boundaries(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START

def _elapsed(global_start):
    return time.time() - global_start

def _remaining(global_start):
    return DEADLINE - _elapsed(global_start)

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _fix_config(xyz):
    """Ensure circles are within boundaries by clipping radii and positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.9
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _slsqp_objective(params):
    """Objective: maximize sum of radii (minimize negative)."""
    r = params[2::3]
    return -np.sum(r)

def _slsqp_constraints(params):
    """Constraints for SLSQP: boundary and non-overlap."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    constraints = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3] - p[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3] - p[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+1] - p[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i*3+1] - p[i*3+2]})
    
    # Non-overlap constraints: dist(i,j) - r_i - r_j >= 0
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    
    for i, j in edges:
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i, j=j: np.sqrt((p[i*3] - p[j*3])**2 + (p[i*3+1] - p[j*3+1])**2) - p[i*3+2] - p[j*3+2]
        })
    
    return constraints

def _slsqp_optimize(xyz0, global_start):
    """Optimize using SLSQP with constraints."""
    params = xyz0.flatten()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.01, 0.3))
    
    constraints = _slsqp_constraints(params)
    
    try:
        result = minimize(_slsqp_objective, params, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-8})
        if result.success or np.sum(result.x[2::3]) > np.sum(params[2::3]):
            return result.x.reshape(N, 3)
    except Exception:
        pass
    
    return xyz0

def _penalty_objective(params, beta=50.0):
    """Penalty-based objective for fallback."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    penalty += np.sum(np.maximum(0, r - x))
    penalty += np.sum(np.maximum(0, r - (1 - x)))
    penalty += np.sum(np.maximum(0, r - y))
    penalty += np.sum(np.maximum(0, r - (1 - y)))
    
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            penalty += max(0, overlap)
    
    return -np.sum(r) + 100.0 * penalty

def _penalty_optimize(xyz0, global_start):
    """Optimize using penalty method as fallback."""
    params = xyz0.flatten()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.01, 0.3))
    
    try:
        result = minimize(_penalty_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(50.0,),
                          options={'maxiter': 300, 'ftol': 1e-8})
        if np.sum(result.x[2::3]) > np.sum(params[2::3]):
            return result.x.reshape(N, 3)
    except Exception:
        pass
    
    return xyz0

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.85
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    positions = []
    # 5x5 grid with offset
    for row in range(5):
        for col in range(5):
            x = (col + 0.5) / 5
            y = (row + 0.5) / 5
            if row % 2 == 1:
                x += 0.1 / 5
            r = 0.5 / 5 * 0.9
            positions.append([x, y, r])
    return np.array(positions[:N])

def _initial_config_rings():
    """Concentric rings pattern."""
    positions = []
    positions.append([0.5, 0.5, 0.08])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.15 * np.cos(angle)
        y = 0.5 + 0.15 * np.sin(angle)
        positions.append([x, y, 0.07])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.28 * np.cos(angle)
        y = 0.5 + 0.28 * np.sin(angle)
        positions.append([x, y, 0.06])
    for k in range(7):
        angle = 2 * np.pi * k / 7
        x = 0.5 + 0.38 * np.cos(angle)
        y = 0.5 + 0.38 * np.sin(angle)
        positions.append([x, y, 0.05])
    return np.array(positions[:N])

def _shake_and_refine(xyz, global_start):
    """Perturb and refine to escape local minima."""
    best = xyz.copy()
    best_score = np.sum(best[:, 2])
    
    for trial in range(5):
        if _remaining(global_start) < 5.0:
            break
        
        perturbed = xyz.copy()
        perturbed[:, 0] += np.random.normal(0, 0.02, N)
        perturbed[:, 1] += np.random.normal(0, 0.02, N)
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        
        refined = _slsqp_optimize(perturbed, global_start)
        if not _validate(refined):
            refined = _fix_config(refined)
        
        score = np.sum(refined[:, 2])
        if score > best_score:
            best = refined
            best_score = score
    
    return best

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    np.random.seed(42)
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    initial_configs = [
        _initial_config_grid(),
        _initial_config_hex(),
        _initial_config_rings(),
    ]
    
    # Random configurations
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.04, 0.10, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        if _remaining(global_start) < 10.0:
            break
        
        # Try SLSQP first
        xyz_opt = _slsqp_optimize(xyz0, global_start)
        
        # If SLSQP didn't improve, try penalty method
        if np.sum(xyz_opt[:, 2]) <= np.sum(xyz0[:, 2]):
            xyz_opt = _penalty_optimize(xyz0, global_start)
        
        # Validate and fix if needed
        if not _validate(xyz_opt):
            xyz_opt = _fix_config(xyz_opt)
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Shake and refine to escape local minima
    if _remaining(global_start) > 15.0:
        best_xyz = _shake_and_refine(best_xyz, global_start)
        best_score = np.sum(best_xyz[:, 2])
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
