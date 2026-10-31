"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(np.clip(beta * x, -50, 50))) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _barrier_objective(x, mu=1.0):
    """
    Interior-point barrier objective using log-barrier for constraints.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    obj = -np.sum(radii)
    
    eps = 1e-10
    d_left = np.maximum(pos[:, 0] - radii, eps)
    d_right = np.maximum(1.0 - pos[:, 0] - radii, eps)
    d_bottom = np.maximum(pos[:, 1] - radii, eps)
    d_top = np.maximum(1.0 - pos[:, 1] - radii, eps)
    
    obj += mu * (np.sum(np.log(d_left)) + np.sum(np.log(d_right)) + 
                 np.sum(np.log(d_bottom)) + np.sum(np.log(d_top)))
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-16)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    gap = dist - min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    gap_clipped = np.maximum(gap[mask], eps)
    obj += mu * np.sum(np.log(gap_clipped))
    
    return obj

def _barrier_gradient(x, mu=1.0):
    """
    Analytical gradient of the barrier objective - vectorized version.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    grad = np.zeros_like(x)
    grad[2*N:] = -1.0
    
    eps = 1e-10
    d_left = np.maximum(pos[:, 0] - radii, eps)
    d_right = np.maximum(1.0 - pos[:, 0] - radii, eps)
    d_bottom = np.maximum(pos[:, 1] - radii, eps)
    d_top = np.maximum(1.0 - pos[:, 1] - radii, eps)
    
    grad[:N:2] += mu / d_left
    grad[:N:2] -= mu / d_right
    grad[1::2] += mu / d_bottom
    grad[1::2] -= mu / d_top
    
    grad[2*N:] -= mu / d_left
    grad[2*N:] -= mu / d_right
    grad[2*N:] -= mu / d_bottom
    grad[2*N:] -= mu / d_top
    
    # Non-overlap barrier gradients - vectorized
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-16)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    gap = dist - min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    gap_clipped = np.maximum(gap, eps)
    inv_gap = 1.0 / gap_clipped
    coef = mu / (dist * gap_clipped)
    
    # For pair (i,j): d/dx_i log(gap) = dx_ij / (dist * gap)
    # grad[2i] += coef[i,j]*dx[i,j] - coef[j,i]*dx[j,i] = coef[i,j]*dx[i,j] + coef[i,j]*dx[i,j]
    # Since dx[j,i] = -dx[i,j] and coef[j,i] = coef[i,j] (symmetric)
    # Actually coef is symmetric, dx is antisymmetric
    # grad[2i] = sum_j coef[i,j]*dx[i,j] for j != i
    # But we only have upper triangle, so:
    # grad[2i] += coef[i,j]*dx[i,j] for j>i, and += coef[j,i]*dx[i,j] for j<i
    # Since coef[j,i] = coef[i,j] and dx[i,j] = -dx[j,i]:
    # grad[2i] = sum_{j>i} coef[i,j]*dx[i,j] + sum_{j<i} coef[j,i]*(-dx[j,i])
    #          = sum_{j>i} coef[i,j]*dx[i,j] + sum_{j<i} coef[i,j]*dx[i,j]
    #          = sum_{j != i} coef[i,j]*dx[i,j]
    
    # Vectorized: grad_pos_x[i] = sum_j coef[i,j] * dx[i,j]
    grad_pos_x = np.sum(coef * dx, axis=1)
    grad_pos_y = np.sum(coef * dy, axis=1)
    
    grad[:N:2] += grad_pos_x
    grad[1::2] += grad_pos_y
    
    # d/dr_i = -sum_j 1/gap_ij
    grad_r = -np.sum(inv_gap, axis=1)
    grad[2*N:] += grad_r
    
    return grad

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)

    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -50.0, 50.0)
    soft_max_neg = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -soft_max_neg[:, 0]
    boundary_pen = np.sum(_soft_max(radii - soft_min_d, beta))

    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol_raw = min_dist - dist
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))

    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

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

def _compute_energy(pos, radii, penalty_weight=50.0):
    """Compute full energy with boundary and overlap penalties."""
    n = len(radii)
    total = -np.sum(radii)
    
    # Boundary penalties
    for i in range(n):
        r = radii[i]
        x, y = pos[i]
        bv = (max(0.0, r - x) + max(0.0, x - (1.0 - r)) +
              max(0.0, r - y) + max(0.0, y - (1.0 - r)))
        total += penalty_weight * bv * bv
    
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            md = radii[i] + radii[j]
            viol = max(0.0, md * md - dist_sq)
            total += penalty_weight * viol * viol
    
    return total

def _simulated_annealing(pos, radii, n_steps=5000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """
    Simulated annealing with proper incremental energy tracking.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    pw = 50.0
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    current_energy = _compute_energy(pos, radii, pw)
    
    # Precompute pairwise violations for fast updates
    def _pairwise_violations(pos, radii):
        bv = np.zeros(n)
        for i in range(n):
            r = radii[i]
            x, y = pos[i]
            bv[i] = (max(0.0, r - x) + max(0.0, x - (1.0 - r)) +
                     max(0.0, r - y) + max(0.0, y - (1.0 - r)))
        
        ov = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                md = radii[i] + radii[j]
                ov[i, j] = max(0.0, md * md - dist_sq)
        return bv, ov
    
    bv, ov = _pairwise_violations(pos, radii)
    
    def _recompute_circle_terms(pos, radii, i):
        r = radii[i]
        x, y = pos[i]
        bv_i = (max(0.0, r - x) + max(0.0, x - (1.0 - r)) +
                max(0.0, r - y) + max(0.0, y - (1.0 - r)))
        
        ov_row = np.zeros(n)
        for j in range(n):
            if j == i:
                continue
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            md = r + radii[j]
            ov_row[j] = max(0.0, md * md - dist_sq)
        return bv_i, ov_row
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)
        
        new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        new_r = np.clip(radii[i] + dr, 0.01, 0.4)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        
        old_bv_i, old_ov_row = _recompute_circle_terms(pos, radii, i)
        pos[i, 0], pos[i, 1], radii[i] = new_x, new_y, new_r
        new_bv_i, new_ov_row = _recompute_circle_terms(pos, radii, i)
        
        delta_bv = pw * (new_bv_i * new_bv_i - old_bv_i * old_bv_i)
        delta_ov = pw * sum(new_ov_row[j] * new_ov_row[j] - old_ov_row[j] * old_ov_row[j] for j in range(n) if j != i)
        delta_r = -(new_r - old_r)
        delta_e = delta_bv + delta_ov + delta_r
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy += delta_e
            bv[i] = new_bv_i
            for j in range(n):
                if j != i:
                    ov[min(i, j), max(i, j)] = new_ov_row[j]
        else:
            pos[i, 0], pos[i, 1], radii[i] = old_x, old_y, old_r
        
        if step % 100 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_radius_growth(pos, radii):
    """
    Maximize sum of radii given fixed positions, subject to non-overlap and boundary constraints.
    This is a Linear Program.
    """
    c = -np.ones(N)
    
    A_ub_rows = []
    b_ub_rows = []
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(dist)
            
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(pos[i, 0])
        
        row = np.zeros(N)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0 - pos[i, 0])
        
        row = np.zeros(N)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(pos[i, 1])
        
        row = np.zeros(N)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0 - pos[i, 1])
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    bounds = [(0, None)] * N
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    return radii

def _slsqp_hard(pos, radii, deadline):
    """
    SLSQP optimization with hard constraints using scipy's constraint formulation.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    # Define constraints
    def boundary_constraints(x):
        pos_c = x[:2*N].reshape(N, 2)
        radii_c = np.maximum(x[2*N:], 1e-8)
        return np.concatenate([
            pos_c[:, 0] - radii_c,  # x >= r
            1.0 - pos_c[:, 0] - radii_c,  # x + r <= 1
            pos_c[:, 1] - radii_c,  # y >= r
            1.0 - pos_c[:, 1] - radii_c  # y + r <= 1
        ])
    
    def overlap_constraints(x):
        pos_c = x[:2*N].reshape(N, 2)
        radii_c = np.maximum(x[2*N:], 1e-8)
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos_c[i, 0] - pos_c[j, 0]
                dy = pos_c[i, 1] - pos_c[j, 1]
                dist = np.sqrt(dx * dx + dy * dy + 1e-16)
                cons.append(dist - radii_c[i] - radii_c[j])
        return np.array(cons)
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    def objective(x):
        radii_c = np.maximum(x[2*N:], 1e-8)
        return -np.sum(radii_c)
    
    try:
        result = minimize(objective, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.success:
            pos_out = result.x[:2*N].reshape(N, 2)
            radii_out = np.maximum(result.x[2*N:], 1e-4)
            return pos_out, radii_out
    except Exception:
        pass
        
    return pos, radii

def _polish(pos, radii, deadline):
    """High-precision polishing with multiple optimization passes."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    # Interior-point barrier optimization with adaptive mu
    if time.time() < deadline - 15:
        mu = 0.5
        for _ in range(20):
            if time.time() > deadline - 12:
                break
            try:
                result = minimize(_barrier_objective, x, jac=_barrier_gradient,
                                  method='L-BFGS-B', bounds=bounds,
                                  args=(mu,),
                                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                if result.fun < _barrier_objective(x, mu) - 1e-10:
                    x = result.x
            except Exception:
                break
            mu *= 0.5
    
    # SLSQP with hard constraints
    if time.time() < deadline - 10:
        try:
            pos_slsqp, radii_slsqp = _slsqp_hard(x[:2*N].reshape(N, 2), np.maximum(x[2*N:], 1e-4), deadline)
            if np.sum(radii_slsqp) > np.sum(np.maximum(x[2*N:], 1e-4)):
                x = np.concatenate([pos_slsqp.flatten(), radii_slsqp])
        except Exception:
            pass
    
    # L-BFGS-B with penalty
    if time.time() < deadline - 8:
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.fun < _objective(x, 5000.0) - 1e-12:
                x = result.x
        except Exception:
            pass
    
    # High-penalty SLSQP
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, x, method='SLSQP', bounds=bounds,
                              args=(20000.0,),
                              options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
            if result.fun < _objective(x, 20000.0) - 1e-13:
                x = result.x
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    # Radius growth step with fixed positions
    if time.time() < deadline - 2:
        radii_grown = _slsqp_radius_growth(pos_out, radii_out)
        if np.sum(radii_grown) > np.sum(radii_out):
            radii_out = radii_grown
    
    # Final micro-polish
    if time.time() < deadline - 1:
        x2 = np.concatenate([pos_out.flatten(), radii_out])
        try:
            result = minimize(_objective, x2, method='SLSQP', bounds=bounds,
                              args=(50000.0,),
                              options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < _objective(x2, 50000.0) - 1e-14:
                x2 = result.x
                pos_out = x2[:2*N].reshape(N, 2)
                radii_out = np.maximum(x2[2*N:], 1e-4)
        except Exception:
            pass
            
    return pos_out, radii_out

def _jitter_and_expand(pos, radii, n_jitters=10, jitter_scale=0.02, deadline=None):
    """
    Jitter positions randomly, then expand radii to maximum possible.
    This decouples topology search from geometry optimization.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    rng = np.random.RandomState(42)
    
    for _ in range(n_jitters):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Jitter positions
        pos_j = pos.copy()
        pos_j += rng.normal(0, jitter_scale, (N, 2))
        pos_j = np.clip(pos_j, 0.01, 0.99)
        
        # Expand radii to maximum
        radii_exp = _slsqp_radius_growth(pos_j, np.ones(N) * 0.01)
        
        # Post-process
        pos_pp, radii_pp = _post_process(pos_j, radii_exp)
        
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
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
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: Interior-point barrier optimization
        if time.time() < deadline - 12:
            mu = 1.0
            for _ in range(15):
                if time.time() > deadline - 10:
                    break
                try:
                    result = minimize(_barrier_objective, current_x, jac=_barrier_gradient,
                                      method='L-BFGS-B', bounds=bounds,
                                      args=(mu,),
                                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                    if result.fun < _barrier_objective(current_x, mu) - 1e-10:
                        current_x = result.x
                except Exception:
                    break
                mu *= 0.5
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 2b: Smooth objective (backup)
        if time.time() < deadline - 8:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 3: Simulated annealing
        if time.time() < deadline - 15:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=3000, t0=0.04, t_end=0.0003,
                    seed=idx * 100, deadline=deadline
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Stage 4: Jitter-and-expand
        if time.time() < deadline - 10:
            try:
                pos_j, rad_j = _jitter_and_expand(
                    best_pos, best_radii,
                    n_jitters=15, jitter_scale=0.015,
                    deadline=deadline
                )
                pos_pp, radii_pp = _post_process(pos_j, rad_j)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
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
