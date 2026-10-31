"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(300):
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

def _compute_contact_graph(pos, radii, tol=1e-4):
    """Compute contact graph based on distance constraints."""
    n = len(radii)
    contacts = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = radii[i] + radii[j]
            if dist <= sum_r + tol:
                contacts[i, j] = True
                contacts[j, i] = True
    
    return contacts

def _solve_radii_fixed_pos(pos, radii_init, contacts):
    """
    Solve for optimal radii given fixed positions using SLSQP.
    Maximizes sum(radii) subject to:
    - r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    - r_i + r_j <= dist(i,j) for contact pairs
    """
    n = len(radii_init)
    
    def objective(x):
        return -np.sum(x)
    
    def jac_objective(x):
        return -np.ones(n)
    
    bounds = []
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        bounds.append((1e-6, max(max_r, 1e-6)))
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            if contacts[i, j]:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                
                def make_constraint(i, j, dist):
                    def constraint(x):
                        return dist - x[i] - x[j]
                    return constraint
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j, dist)})
    
    x0 = radii_init.copy()
    try:
        res = minimize(objective, x0, method='SLSQP', 
                      bounds=bounds, constraints=constraints,
                      jac=jac_objective,
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success or res.fun < objective(x0):
            return np.maximum(res.x, 1e-6)
    except Exception:
        pass
    
    return radii_init

def _optimize_positions_slsqp(pos, radii, max_iter=200):
    """
    Optimize positions given fixed radii using SLSQP with hard constraints.
    This ensures feasibility during optimization.
    """
    n = len(radii)
    x0 = pos.flatten().copy()
    
    bounds = []
    for i in range(n):
        bounds.append((radii[i], 1.0 - radii[i]))
        bounds.append((radii[i], 1.0 - radii[i]))
    
    def objective(x):
        pos = x.reshape(n, 2)
        # Penalize any overlap (should be zero at optimum)
        pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist:
                    viol = min_dist - np.sqrt(dist_sq)
                    pen += viol * viol
        return pen
    
    def jac_objective(x):
        pos = x.reshape(n, 2)
        grad = np.zeros(2 * n)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[i] + radii[j]
                
                if dist_sq < min_dist * min_dist and dist_sq > 1e-12:
                    dist = np.sqrt(dist_sq)
                    viol = min_dist - dist
                    grad[2*i] += 2 * viol * (-dx / dist)
                    grad[2*i+1] += 2 * viol * (-dy / dist)
                    grad[2*j] += 2 * viol * (dx / dist)
                    grad[2*j+1] += 2 * viol * (dy / dist)
        return grad
    
    # Hard constraints: dist(i,j) >= r_i + r_j
    def overlap_constraints(x):
        pos = x.reshape(n, 2)
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                cons.append(dist_sq - min_dist*min_dist)
        return np.array(cons)
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=[{'type': 'ineq', 'fun': overlap_constraints}],
                       jac=jac_objective,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if res.success or (hasattr(res, 'x') and res.fun < objective(x0)):
            return res.x.reshape(n, 2)
    except Exception:
        pass
    
    return pos

def _contact_graph_optimize(pos, radii, max_outer=20, max_inner=30):
    """
    Iterative contact graph optimization:
    1. Compute contact graph
    2. Solve for optimal radii given contacts (SLSQP)
    3. Optimize positions given radii (SLSQP)
    4. Repeat
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for outer in range(max_outer):
        # Compute contact graph
        contacts = _compute_contact_graph(pos, radii, tol=1e-3)
        
        # Solve for radii given fixed positions
        new_radii = _solve_radii_fixed_pos(pos, radii, contacts)
        
        # Check validity and update best
        pos_pp, radii_pp = _post_process(pos, new_radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Optimize positions given new radii
        pos = _optimize_positions_slsqp(pos, new_radii, max_iter=max_inner)
        
        # Update radii based on new positions
        radii = _solve_radii_fixed_pos(pos, new_radii, contacts)
        
        # Check convergence
        if outer > 0:
            pos_diff = np.max(np.abs(pos - best_pos))
            if pos_diff < 1e-7:
                break
    
    return best_pos, best_radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_hexagonal():
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
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
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

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
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
            for attempt in range(200):
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

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _slsqp_hard(pos, radii, deadline):
    """
    Optimize using SLSQP with explicit hard constraints.
    """
    n = len(radii)
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.99))
        bounds.append((1e-4, 0.99))
    for i in range(n):
        bounds.append((1e-4, 0.5))

    def boundary_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = []
        for i in range(n):
            cons.append(pos[i, 0] - radii[i])
            cons.append(1.0 - pos[i, 0] - radii[i])
            cons.append(pos[i, 1] - radii[i])
            cons.append(1.0 - pos[i, 1] - radii[i])
        return np.array(cons)

    def overlap_constraints(x):
        pos = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                cons.append(dist_sq - min_dist*min_dist)
        return np.array(cons)

    def objective(x):
        radii = x[2*n:]
        return -np.sum(radii)

    def jac_objective(x):
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        return grad
    
    try:
        res = minimize(objective, x, method='SLSQP', bounds=bounds,
                       constraints=[
                           {'type': 'ineq', 'fun': boundary_constraints},
                           {'type': 'ineq', 'fun': overlap_constraints}
                       ],
                       jac=jac_objective,
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if res.success or (hasattr(res, 'x') and res.fun < objective(x)):
            new_pos = res.x[:2*n].reshape(n, 2)
            new_radii = np.maximum(res.x[2*n:], 1e-4)
            return new_pos, new_radii
    except Exception:
        pass
    
    return pos, radii

def _try_radius_expansion(pos, radii, max_iter=100, step_size=0.005):
    """
    Try to expand radii of circles that are not touching any constraint.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            touching_boundary = False
            if abs(pos[i, 0] - radii[i]) < 1e-5 or abs(pos[i, 0] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            if abs(pos[i, 1] - radii[i]) < 1e-5 or abs(pos[i, 1] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            
            touching_circle = False
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + 1e-5:
                    touching_circle = True
                    break
            
            if not touching_boundary and not touching_circle:
                max_expansion = 0.01
                max_expansion = min(max_expansion, pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, pos[i, 1] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 1] - radii[i])
                
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    space = dist - radii[i] - radii[j]
                    max_expansion = min(max_expansion, space)
                
                if max_expansion > 1e-6:
                    expansion = min(step_size, max_expansion * 0.5)
                    radii[i] += expansion
                    improved = True
        
        if not improved:
            break
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    # Stage 1: Contact graph optimization for each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        try:
            pos_cg, rad_cg = _contact_graph_optimize(pos_init, rad_init, max_outer=15, max_inner=30)
            pos_pp, radii_pp = _post_process(pos_cg, rad_cg)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Stage 2: SLSQP polish on the best result
    if time.time() < deadline - 20:
        try:
            pos_slq, rad_slq = _slsqp_hard(best_pos, best_radii, deadline)
            pos_exp, rad_exp = _try_radius_expansion(pos_slq, rad_slq, max_iter=100, step_size=0.005)
            pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Stage 3: Another contact graph optimization from the polished result
    if time.time() < deadline - 15:
        try:
            pos_cg2, rad_cg2 = _contact_graph_optimize(best_pos, best_radii, max_outer=10, max_inner=20)
            pos_pp, radii_pp = _post_process(pos_cg2, rad_cg2)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Stage 4: Final SLSQP polish
    if time.time() < deadline - 10:
        try:
            pos_slq2, rad_slq2 = _slsqp_hard(best_pos, best_radii, deadline)
            pos_exp2, rad_exp2 = _try_radius_expansion(pos_slq2, rad_slq2, max_iter=50, step_size=0.002)
            pos_pp, radii_pp = _post_process(pos_exp2, rad_exp2)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Stage 5: Perturbation restarts
    if time.time() < deadline - 5:
        for seed in [42, 123, 456]:
            if time.time() > deadline - 3:
                break
            rng = np.random.RandomState(seed)
            pos_pert = best_pos + rng.normal(0, 0.005, (N, 2))
            pos_pert = np.clip(pos_pert, 0.01, 0.99)
            radii_pert = best_radii * 0.98
            
            try:
                pos_cg3, rad_cg3 = _contact_graph_optimize(pos_pert, radii_pert, max_outer=8, max_inner=15)
                pos_pp, radii_pp = _post_process(pos_cg3, rad_cg3)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Final radius expansion attempt
    if time.time() < deadline - 1:
        pos_exp, rad_exp = _try_radius_expansion(best_pos, best_radii, max_iter=100, step_size=0.001)
        pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
            best_pos = pos_pp
            best_radii = radii_pp
            best_score = np.sum(radii_pp)
            
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
