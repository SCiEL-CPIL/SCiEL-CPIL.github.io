"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: SLSQP constrained optimization + contact-graph local polish + simulated annealing restarts.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(200):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _sum_radii(pos, radii):
    return np.sum(radii)

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_grid():
    """Grid initialization with corner bias."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                radii[k] = 0.09
                k += 1
    if k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.01
        k += 1
    return pos, radii

def _init_corners():
    """Corner-biased initialization."""
    rng = np.random.RandomState(42)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02)]
        radii[i] = 0.17
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        if max_r < 0.01:
            for attempt in range(300):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
    return pos, radii

def _init_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _slsqp_solve(pos, radii, deadline=None):
    """
    SLSQP optimization: maximize sum of radii subject to boundary and non-overlap constraints.
    Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n_vars = 3 * N
    x0 = np.zeros(n_vars)
    for i in range(N):
        x0[3*i] = pos[i, 0]
        x0[3*i+1] = pos[i, 1]
        x0[3*i+2] = radii[i]
    
    # Objective: minimize -sum(radii)
    def objective(x):
        radii_vec = x[2::3]
        return -np.sum(radii_vec)
    
    # Constraints: boundary and non-overlap
    def make_boundary_constraint(i, wall):
        """wall: 0=left, 1=right, 2=bottom, 3=top"""
        def constraint(x):
            xi = x[3*i]
            yi = x[3*i+1]
            ri = x[3*i+2]
            if wall == 0:
                return xi - ri
            elif wall == 1:
                return 1.0 - xi - ri
            elif wall == 2:
                return yi - ri
            else:
                return 1.0 - yi - ri
        return constraint
    
    def make_overlap_constraint(i, j):
        def constraint(x):
            xi = x[3*i]
            yi = x[3*i+1]
            ri = x[3*i+2]
            xj = x[3*j]
            yj = x[3*j+1]
            rj = x[3*j+2]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - ri - rj
        return constraint
    
    constraints = []
    # Boundary constraints
    for i in range(N):
        for wall in range(4):
            constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, wall)})
    # Non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
    
    # Bounds: radii > 0, positions in [0,1]
    bounds = []
    for i in range(N):
        bounds.append((1e-5, 1.0))  # x
        bounds.append((1e-5, 1.0))  # y
        bounds.append((1e-5, 0.5))  # r
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
    
    if result.success:
        x_opt = result.x
        pos_opt = np.zeros((N, 2))
        radii_opt = np.zeros(N)
        for i in range(N):
            pos_opt[i, 0] = x_opt[3*i]
            pos_opt[i, 1] = x_opt[3*i+1]
            radii_opt[i] = x_opt[3*i+2]
        return pos_opt, radii_opt
    
    return pos, radii

def _local_search(pos, radii, deadline=None):
    """
    Local search: iteratively move one circle to maximize its radius given others fixed.
    """
    pos = pos.copy()
    radii = radii.copy()
    improved = True
    while improved and (deadline is None or time.time() < deadline - 2):
        improved = False
        for i in range(N):
            # Try to grow circle i
            x, y = pos[i]
            r = radii[i]
            
            # Compute max possible radius
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(N):
                if j == i:
                    continue
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - radii[j])
            
            if max_r > r * 1.001:
                radii[i] = max_r * 0.9999
                improved = True
            
            # Try small random nudges
            for _ in range(5):
                nx = x + np.random.uniform(-0.02, 0.02)
                ny = y + np.random.uniform(-0.02, 0.02)
                nx = np.clip(nx, 0.01, 0.99)
                ny = np.clip(ny, 0.01, 0.99)
                
                new_max_r = min(nx, 1.0 - nx, ny, 1.0 - ny)
                for j in range(N):
                    if j == i:
                        continue
                    dx = nx - pos[j, 0]
                    dy = ny - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    new_max_r = min(new_max_r, dist - radii[j])
                
                if new_max_r > radii[i] * 1.01:
                    pos[i] = [nx, ny]
                    radii[i] = new_max_r * 0.9999
                    improved = True
                    break
    
    return pos, radii

def _simulated_annealing(pos, radii, deadline=None):
    """
    Simulated annealing: perturb positions, then re-optimize radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _sum_radii(pos, radii)
    
    T = 0.05
    cooling = 0.995
    rng = np.random.RandomState(12345)
    
    for step in range(500):
        if deadline is not None and time.time() > deadline - 3:
            break
        
        # Perturb a random circle
        i = rng.randint(N)
        dx = rng.gaussian(0, T)
        dy = rng.gaussian(0, T)
        pos[i, 0] = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        pos[i, 1] = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        
        # Recompute radius for this circle
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(N):
            if j == i:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            max_r = min(max_r, dist - radii[j])
        radii[i] = max(max_r, 1e-5)
        
        score = _sum_radii(pos, radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
        
        T *= cooling
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate initializations
    inits = []
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_grid, rad_grid = _init_grid()
    inits.append((pos_grid, rad_grid))
    
    pos_cor, rad_cor = _init_corners()
    inits.append((pos_cor, rad_cor))
    
    pos_r1, rad_r1 = _init_random(100)
    inits.append((pos_r1, rad_r1))
    
    pos_r2, rad_r2 = _init_random(200)
    inits.append((pos_r2, rad_r2))
    
    best_pos = pos_hex.copy()
    best_radii = rad_hex.copy()
    best_score = _sum_radii(best_pos, best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Phase 1: SLSQP optimization
        pos_slsqp, rad_slsqp = _slsqp_solve(pos_init, rad_init, deadline=deadline)
        
        # Phase 2: Local search
        pos_ls, rad_ls = _local_search(pos_slsqp, rad_slsqp, deadline=deadline)
        
        # Phase 3: Post-process
        pos_pp, rad_pp = _post_process(pos_ls, rad_ls)
        
        score = _sum_radii(pos_pp, rad_pp)
        if _is_valid(pos_pp, rad_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = rad_pp.copy()
        
        # Phase 4: Simulated annealing refinement
        if time.time() < deadline - 5:
            pos_sa, rad_sa = _simulated_annealing(pos_pp, rad_pp, deadline=deadline)
            pos_pp2, rad_pp2 = _post_process(pos_sa, rad_sa)
            
            score2 = _sum_radii(pos_pp2, rad_pp2)
            if _is_valid(pos_pp2, rad_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = rad_pp2.copy()
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.09
                    k += 1
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
