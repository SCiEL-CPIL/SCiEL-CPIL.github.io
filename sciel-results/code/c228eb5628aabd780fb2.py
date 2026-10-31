"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-6):
    """Strict validity check for final result."""
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

def _post_process(pos, radii, tol=1e-7):
    """Shrink radii slightly to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(100):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + tol:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    if sum_r > 1e-9:
                        k = (dist - tol) / (sum_r + 1e-12)
                    else:
                        k = 0.99
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _initial_greedy_best_fit():
    """Generate a tight initial packing using a greedy best-fit algorithm."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        for _ in range(15000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
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
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_hexagonal():
    """Generate a hexagonal lattice-like initial state."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            pos[k] = [x, y]
            radii[k] = r
            k += 1
    while k < N:
        if k == 25:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.05
        else:
            pos[k] = [0.1, 0.1]
            radii[k] = 0.05
        k += 1
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    """Generate a random valid-ish initial state."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _hard_constraints(x):
    """All hard inequality constraints (for SLSQP)."""
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    b_cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    
    o_cons_list = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            o_cons_list.append(dist_sq - sum_r * sum_r)
    o_cons = np.array(o_cons_list)
    
    return np.concatenate([b_cons, o_cons])

def _soft_penalty_objective(x, beta=120.0):
    """
    Smooth penalty-based objective using soft-min for constraint violations.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    b_cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    
    o_cons_list = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            o_cons_list.append(dist_sq - sum_r * sum_r)
    o_cons = np.array(o_cons_list)
    
    all_cons = np.concatenate([b_cons, o_cons])
    
    scaled = -beta * all_cons
    scaled = np.clip(scaled, -500, 500)
    penalty = -np.log(1.0 + np.exp(scaled)) / beta
    
    return -np.sum(radii) + penalty.sum()

def _contact_graph_solve(pos, radii, max_iter=40, tol=1e-10):
    """
    Contact-graph-aware optimization.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for it in range(max_iter):
        contacts = []
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + tol:
                    contacts.append((i, j))
        
        wall_contacts = []
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            if x - r < tol: wall_contacts.append((i, 'L'))
            if 1.0 - x - r < tol: wall_contacts.append((i, 'R'))
            if y - r < tol: wall_contacts.append((i, 'B'))
            if 1.0 - y - r < tol: wall_contacts.append((i, 'T'))
        
        def active_constraints(x):
            pos_x = x[:2*N].reshape(N, 2)
            rad_x = x[2*N:]
            cons = []
            for (i, j) in contacts:
                dx = pos_x[i,0] - pos_x[j,0]
                dy = pos_x[i,1] - pos_x[j,1]
                dist_sq = dx*dx + dy*dy
                sum_r = rad_x[i] + rad_x[j]
                cons.append(dist_sq - sum_r*sum_r)
            for (i, wall) in wall_contacts:
                if wall == 'L':
                    cons.append(pos_x[i,0] - rad_x[i])
                elif wall == 'R':
                    cons.append(1.0 - pos_x[i,0] - rad_x[i])
                elif wall == 'B':
                    cons.append(pos_x[i,1] - rad_x[i])
                elif wall == 'T':
                    cons.append(1.0 - pos_x[i,1] - rad_x[i])
            for i in range(N):
                for j in range(i+1, N):
                    if (i, j) not in contacts:
                        dx = pos_x[i,0] - pos_x[j,0]
                        dy = pos_x[i,1] - pos_x[j,1]
                        dist_sq = dx*dx + dy*dy
                        sum_r = rad_x[i] + rad_x[j]
                        cons.append(dist_sq - sum_r*sum_r)
            cons.extend(pos_x[:,0] - rad_x)
            cons.extend(1.0 - pos_x[:,0] - rad_x)
            cons.extend(pos_x[:,1] - rad_x)
            cons.extend(1.0 - pos_x[:,1] - rad_x)
            return np.array(cons)
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((1e-5, 0.5))
        
        x0 = np.concatenate([pos.flatten(), radii])
        
        try:
            result = minimize(
                lambda x: -np.sum(x[2*N:]),
                x0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': active_constraints},
                options={'maxiter': 80, 'ftol': 1e-13}
            )
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_new, radii_new = _post_process(pos_new, radii_new, tol=1e-9)
            if _is_valid(pos_new, radii_new, tol=1e-8):
                score = np.sum(radii_new)
                if score > best_score + 1e-12:
                    best_score = score
                    best_pos = pos_new.copy()
                    best_radii = radii_new.copy()
                    pos = pos_new
                    radii = radii_new
                else:
                    break
        except Exception:
            break
    
    return best_pos, best_radii

def _high_precision_polish(pos, radii, max_iters=300, time_limit=None, start_time=None):
    """
    High-precision polish with SLSQP and coordinate descent.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 30.0
    
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        x0 = np.concatenate([pos.flatten(), radii])
        try:
            result = minimize(
                lambda x: -np.sum(x[2*N:]),
                x0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': _hard_constraints},
                options={'maxiter': 150, 'ftol': 1e-14}
            )
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_new, radii_new = _post_process(pos_new, radii_new, tol=1e-9)
            if _is_valid(pos_new, radii_new, tol=1e-8):
                score = np.sum(radii_new)
                if score > best_score:
                    best_score = score
                    best_pos = pos_new.copy()
                    best_radii = radii_new.copy()
                    pos = pos_new
                    radii = radii_new
        except Exception:
            pass
        
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            r_lo = radii[i]
            r_hi = min(r_lo * 1.01 + 0.001, 
                       min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1]))
            if r_hi <= r_lo + 1e-12:
                continue
            for _ in range(40):
                r_mid = (r_lo + r_hi) / 2.0
                test_radii = radii.copy()
                test_radii[i] = r_mid
                test_pos, test_radii = _post_process(pos, test_radii, tol=1e-9)
                if _is_valid(test_pos, test_radii, tol=1e-8):
                    r_lo = r_mid
                    radii[i] = r_mid
                    improved = True
                else:
                    r_hi = r_mid
            if r_lo > radii[i] + 1e-12:
                radii[i] = r_lo
                improved = True
                pos, radii = _post_process(pos, radii, tol=1e-9)
        
        if not improved:
            break
    
    return best_pos, best_radii

def _soft_penalty_optimize(pos_init, radii_init, start_time, deadline):
    """
    Optimize using soft penalty objective with L-BFGS-B, then polish with SLSQP.
    """
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    pos_l, radii_l = pos_init, radii_init
    for beta in [80.0, 150.0, 300.0, 500.0]:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(
                lambda x, b=beta: _soft_penalty_objective(x, b), x0,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pos_try = result.x[:2*N].reshape(N, 2)
            radii_try = result.x[2*N:]
            pos_try, radii_try = _post_process(pos_try, radii_try)
            if _is_valid(pos_try, radii_try):
                score = np.sum(radii_try)
                score_cur = np.sum(radii_l)
                if score > score_cur:
                    pos_l, radii_l = pos_try, radii_try
        except Exception:
            pass
    
    if time.time() < deadline - 5:
        try:
            result2 = minimize(
                lambda x: -np.sum(x[2*N:]),
                np.concatenate([pos_l.flatten(), radii_l]),
                method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': _hard_constraints},
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            pos_s = result2.x[:2*N].reshape(N, 2)
            radii_s = result2.x[2*N:]
            pos_s, radii_s = _post_process(pos_s, radii_s)
            if _is_valid(pos_s, radii_s):
                return pos_s, radii_s
        except Exception:
            pass
    
    if _is_valid(pos_l, radii_l):
        return pos_l, radii_l
    return pos_init, radii_init

def _vectorized_refine(pos, radii, steps=600, step_size=0.025, rng=None):
    """
    Vectorized local search to refine the packing.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for step in range(steps):
        current_step = step_size * (0.997 ** step)
        delta_pos = rng.normal(0, current_step, (N, 2))
        delta_radii = rng.normal(0, current_step * 0.4, N)
        new_pos = pos + delta_pos
        new_radii = radii + delta_radii
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, 1)
        new_radii = np.maximum(new_radii, 1e-5)
        new_pos, new_radii = _post_process(new_pos, new_radii, tol=1e-8)
        if _is_valid(new_pos, new_radii, tol=1e-7):
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos = new_pos
                radii = new_radii
    return best_pos, best_radii

def _convex_relaxation_step(pos, radii, max_iter=40, tol=1e-10):
    """
    Convex relaxation step: linearize non-convex pairwise distance constraints.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for it in range(max_iter):
        dx = np.zeros((N, N))
        dy = np.zeros((N, N))
        dist = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                ddx = pos[i, 0] - pos[j, 0]
                ddy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(ddx*ddx + ddy*ddy)
                dx[i, j] = ddx
                dy[i, j] = ddy
                dist[i, j] = d
                dx[j, i] = -ddx
                dy[j, i] = -ddy
                dist[j, i] = d
        
        def lin_constraints(x):
            pos_x = x[:2*N].reshape(N, 2)
            rad_x = x[2*N:]
            
            b_cons = np.concatenate([
                pos_x[:, 0] - rad_x,
                1.0 - pos_x[:, 0] - rad_x,
                pos_x[:, 1] - rad_x,
                1.0 - pos_x[:, 1] - rad_x
            ])
            
            o_cons = []
            for i in range(N):
                for j in range(i+1, N):
                    d = dist[i, j]
                    if d < 1e-12:
                        ni_x = 1.0
                        ni_y = 0.0
                    else:
                        ni_x = dx[i, j] / d
                        ni_y = dy[i, j] / d
                    val = (pos_x[i, 0] - pos_x[j, 0]) * ni_x + \
                          (pos_x[i, 1] - pos_x[j, 1]) * ni_y - rad_x[i] - rad_x[j]
                    o_cons.append(val)
            
            return np.concatenate([b_cons, o_cons])
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((1e-5, 0.5))
        
        x0 = np.concatenate([pos.flatten(), radii])
        
        try:
            result = minimize(
                lambda x: -np.sum(x[2*N:]),
                x0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': lin_constraints},
                options={'maxiter': 100, 'ftol': 1e-13}
            )
            
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            
            pos_new, radii_new = _post_process(pos_new, radii_new, tol=1e-9)
            if _is_valid(pos_new, radii_new, tol=1e-8):
                score = np.sum(radii_new)
                if score > best_score + 1e-11:
                    best_score = score
                    best_pos = pos_new.copy()
                    best_radii = radii_new.copy()
                    pos = pos_new
                    radii = radii_new
                else:
                    break
        except Exception:
            break
    
    return best_pos, best_radii

def _force_directed_anneal(pos, radii, start_time, deadline, max_iter=2000):
    """
    Force-directed annealing: treat overlapping circles as repelling,
    walls as attracting, and gradually cool.
    """
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    T = 0.05
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        forces = np.zeros((N, 2))
        
        # Repulsion between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r and dist > 1e-12:
                    overlap = sum_r - dist
                    f = 10.0 * overlap / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                elif dist <= 1e-12:
                    forces[i, 0] += 0.01
                    forces[j, 0] -= 0.01
        
        # Wall repulsion
        for i in range(N):
            x, y = pos[i]
            r = radii[i]
            if x < r:
                forces[i, 0] += 10.0 * (r - x)
            if x > 1.0 - r:
                forces[i, 0] -= 10.0 * (x - (1.0 - r))
            if y < r:
                forces[i, 1] += 10.0 * (r - y)
            if y > 1.0 - r:
                forces[i, 1] -= 10.0 * (y - (1.0 - r))
        
        # Add noise
        noise = np.random.normal(0, T, (N, 2))
        
        # Update positions
        step = 0.01
        pos += step * (forces + noise)
        pos[:, 0] = np.clip(pos[:, 0], 0.01, 0.99)
        pos[:, 1] = np.clip(pos[:, 1], 0.01, 0.99)
        
        # Try to increase radii
        pos, radii = _post_process(pos, radii, tol=1e-8)
        
        score = np.sum(radii)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
        
        T *= cooling
    
    return best_pos, best_radii

def _hilbert_curve_init():
    """
    Generate initial positions using Hilbert curve ordering for spatial locality.
    """
    # Simple space-filling curve approximation: use a Z-order (Morton) curve
    # to place circles in a locality-preserving order
    n = N
    # Create a grid of candidate positions
    grid_size = 8
    candidates = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 0.5) / grid_size
            y = (j + 0.5) / grid_size
            candidates.append((x, y))
    
    # Sort by Z-order (Morton code)
    def morton_code(x, y, bits=4):
        code = 0
        for b in range(bits):
            code |= ((x >> b) & 1) << (2*b)
            code |= ((y >> b) & 1) << (2*b + 1)
        return code
    
    # Use a simpler approach: snake order
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    grid_n = 5
    cell_size = 1.0 / grid_n
    k = 0
    for row in range(grid_n):
        cols = range(grid_n) if row % 2 == 0 else range(grid_n-1, -1, -1)
        for col in cols:
            if k >= N:
                break
            x = (col + 0.5) * cell_size
            y = (row + 0.5) * cell_size
            pos[k] = [x, y]
            radii[k] = cell_size * 0.48
            k += 1
    pos, radii = _post_process(pos, radii)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    initializers = [
        _initial_greedy_best_fit,
        _initial_hexagonal,
        _hilbert_curve_init,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
        lambda: _initial_random(6),
        lambda: _initial_random(7),
        lambda: _initial_random(8),
        lambda: _initial_random(9),
        lambda: _initial_random(10),
    ]
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 25:
            break
        
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
        
        # Strategy A: SLSQP directly
        if time.time() < deadline - 20:
            try:
                x0 = np.concatenate([pos_init.flatten(), rad_init])
                result = minimize(
                    lambda x: -np.sum(x[2*N:]), x0,
                    method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': _hard_constraints},
                    options={'maxiter': 250, 'ftol': 1e-12}
                )
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Strategy B: Soft penalty L-BFGS-B then SLSQP polish
        if time.time() < deadline - 20:
            try:
                pos_b, radii_b = _soft_penalty_optimize(pos_init, rad_init, start_time, deadline)
                if _is_valid(pos_b, radii_b):
                    score = np.sum(radii_b)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_b.copy()
                        best_radii = radii_b.copy()
            except Exception:
                pass
        
        # Strategy C: Contact graph solve
        if time.time() < deadline - 20 and best_pos is not None:
            try:
                pos_c, radii_c = _contact_graph_solve(
                    best_pos, best_radii, max_iter=30
                )
                if _is_valid(pos_c, radii_c):
                    score_c = np.sum(radii_c)
                    if score_c > best_score:
                        best_score = score_c
                        best_pos = pos_c.copy()
                        best_radii = radii_c.copy()
            except Exception:
                pass
        
        # Strategy D: Convex relaxation step
        if time.time() < deadline - 20 and best_pos is not None:
            try:
                pos_d, radii_d = _convex_relaxation_step(
                    best_pos, best_radii, max_iter=30
                )
                if _is_valid(pos_d, radii_d):
                    score_d = np.sum(radii_d)
                    if score_d > best_score:
                        best_score = score_d
                        best_pos = pos_d.copy()
                        best_radii = radii_d.copy()
            except Exception:
                pass
        
        # Strategy E: Force-directed anneal
        if time.time() < deadline - 15 and best_pos is not None:
            try:
                pos_e, radii_e = _force_directed_anneal(
                    best_pos, best_radii, start_time, deadline, max_iter=500
                )
                if _is_valid(pos_e, radii_e):
                    score_e = np.sum(radii_e)
                    if score_e > best_score:
                        best_score = score_e
                        best_pos = pos_e.copy()
                        best_radii = radii_e.copy()
            except Exception:
                pass
        
        # Vectorized refinement on best so far
        if time.time() < deadline - 15 and best_pos is not None:
            try:
                pos_ref, rad_ref = _vectorized_refine(
                    best_pos, best_radii,
                    steps=300, step_size=0.02,
                    rng=np.random.RandomState(idx * 7 + 1)
                )
                if _is_valid(pos_ref, rad_ref):
                    score_ref = np.sum(rad_ref)
                    if score_ref > best_score:
                        best_score = score_ref
                        best_pos = pos_ref.copy()
                        best_radii = rad_ref.copy()
            except Exception:
                pass
    
    # Final high-precision polish with coordinate descent
    if best_pos is not None and time.time() < deadline - 10:
        try:
            pos_pol, rad_pol = _high_precision_polish(
                best_pos, best_radii,
                max_iters=200,
                time_limit=deadline - time.time() - 5,
                start_time=start_time
            )
            if _is_valid(pos_pol, rad_pol):
                score_pol = np.sum(rad_pol)
                if score_pol > best_score:
                    best_score = score_pol
                    best_pos = pos_pol.copy()
                    best_radii = rad_pol.copy()
        except Exception:
            pass
    
    # Final contact graph polish
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_cg, rad_cg = _contact_graph_solve(
                best_pos, best_radii, max_iter=20
            )
            if _is_valid(pos_cg, rad_cg):
                score_cg = np.sum(rad_cg)
                if score_cg > best_score:
                    best_score = score_cg
                    best_pos = pos_cg.copy()
                    best_radii = rad_cg.copy()
        except Exception:
            pass
    
    # Final convex relaxation polish
    if best_pos is not None and time.time() < deadline - 3:
        try:
            pos_cr, rad_cr = _convex_relaxation_step(
                best_pos, best_radii, max_iter=15
            )
            if _is_valid(pos_cr, rad_cr):
                score_cr = np.sum(rad_cr)
                if score_cr > best_score:
                    best_score = score_cr
                    best_pos = pos_cr.copy()
                    best_radii = rad_cr.copy()
        except Exception:
            pass
    
    # Final fallback
    if best_pos is None or not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
