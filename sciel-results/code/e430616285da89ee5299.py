"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(radii <= 0):
        return False
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

def _post_process(pos, radii, max_iter=500):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    for _ in range(max_iter):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-6)
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
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
        if valid:
            break
    return pos, radii

def _max_radius_at(x, y, pos, radii, idx=None):
    """Max radius a circle at (x,y) can have given existing circles."""
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    if idx is not None:
        pos = np.delete(pos, idx, axis=0)
        radii = np.delete(radii, idx)
    for j in range(len(radii)):
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, dist - radii[j])
    return max(0.0, max_r)

def _greedy_init(seed=42, n_trials=8000):
    """Greedy best-fit initialization with random sampling."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Start with a large central circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(n_trials):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            r = _max_radius_at(x, y, pos[:i], radii[:i])
            if r > best_r and r > 0.005:
                best_r = r
                best_x = x
                best_y = y
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98
    
    return pos, radii

def _hex_init():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.115
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = r + offset + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    while k < N:
        for attempt in range(200):
            rng = np.random.RandomState(k * 7 + 13 + attempt)
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            r = _max_radius_at(x, y, pos[:k], radii[:k])
            if r > 0.015:
                pos[k] = [x, y]
                radii[k] = r * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _corner_init(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.20
    
    edges = [
        (0.5, 0.06), (0.5, 0.94),
        (0.06, 0.5), (0.94, 0.5),
        (0.3, 0.06), (0.7, 0.06),
        (0.3, 0.94), (0.7, 0.94),
        (0.06, 0.3), (0.06, 0.7),
        (0.94, 0.3), (0.94, 0.7),
    ]
    for i in range(4, 16):
        ex, ey = edges[i - 4]
        pos[i] = [ex + rng.uniform(-0.02, 0.02), ey + rng.uniform(-0.02, 0.02)]
        r = _max_radius_at(pos[i, 0], pos[i, 1], pos[:i], radii[:i])
        radii[i] = max(r * 0.95, 0.02)
    
    for i in range(16, N):
        for attempt in range(300):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            r = _max_radius_at(x, y, pos[:i], radii[:i])
            if r > 0.012:
                pos[i] = [x, y]
                radii[i] = r * 0.95
                break
        else:
            pos[i] = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
            radii[i] = 0.01
    
    return pos, radii

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.03, 0.10, N)
    return _post_process(pos, radii)

def _smooth_objective(x, penalty_weight=1000.0, beta=50.0):
    """
    Smooth penalty objective for L-BFGS-B.
    Uses softplus to smooth max(0, violation) terms.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    
    # Boundary violations (smooth)
    bv_left = _soft_max(radii - pos[:, 0], beta)
    bv_right = _soft_max(pos[:, 0] - (1.0 - radii), beta)
    bv_bottom = _soft_max(radii - pos[:, 1], beta)
    bv_top = _soft_max(pos[:, 1] - (1.0 - radii), beta)
    boundary_pen = np.sum(bv_left * bv_left + bv_right * bv_right + bv_bottom * bv_bottom + bv_top * bv_top)
    
    # Overlap violations (smooth)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    ov = _soft_max(min_dist[mask] - dist[mask], beta)
    overlap_pen = np.sum(ov * ov)
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _smooth_optimize(x0, deadline, penalty_weight=1000.0, beta=50.0, max_iter=500):
    """L-BFGS-B with smooth penalty objective."""
    x = x0.copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.001, 0.45))
    
    try:
        result = minimize(_smooth_objective, x, method='L-BFGS-B', bounds=bounds,
                          args=(penalty_weight, beta),
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if result.fun < _smooth_objective(x, penalty_weight, beta):
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2 * N].reshape(N, 2)
    radii_out = np.maximum(x[2 * N:], 1e-4)
    return pos_out, radii_out

def _slsqp_optimize(x0, deadline, max_iter=200):
    """SLSQP with hard constraints using Euclidean distance for overlaps."""
    x = x0.copy()
    n = N
    
    def constraint_boundary(x):
        pos = x[:2 * n].reshape(n, 2)
        radii = x[2 * n:]
        c1 = pos[:, 0] - radii
        c2 = 1.0 - pos[:, 0] - radii
        c3 = pos[:, 1] - radii
        c4 = 1.0 - pos[:, 1] - radii
        return np.concatenate([c1, c2, c3, c4])
    
    def constraint_overlap(x):
        pos = x[:2 * n].reshape(n, 2)
        radii = x[2 * n:]
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy + 1e-16)
                min_dist = radii[i] + radii[j]
                constraints.append(dist - min_dist)
        return np.array(constraints)
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.001, 0.45))
    
    def neg_sum_radii(x):
        return -np.sum(x[2 * n:])
    
    try:
        result = minimize(neg_sum_radii, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'maxls': 50})
        if result.fun < neg_sum_radii(x):
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2 * n].reshape(n, 2)
    radii_out = np.maximum(x[2 * n:], 1e-4)
    return pos_out, radii_out

def _polish(pos, radii, deadline):
    """
    High-precision polish: alternate between SLSQP and smooth L-BFGS-B
    with progressively tighter tolerances.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Phase 1: Smooth L-BFGS-B with increasing penalty
    for pw in [500.0, 2000.0, 5000.0, 10000.0]:
        if time.time() > deadline - 5:
            break
        x = np.concatenate([pos.flatten(), radii])
        pos_s, radii_s = _smooth_optimize(x, deadline, penalty_weight=pw, beta=100.0, max_iter=400)
        pos_pp, radii_pp = _post_process(pos_s, radii_s)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
                pos = best_pos
                radii = best_radii
    
    # Phase 2: SLSQP polish with tight tolerance and more iterations
    if time.time() < deadline - 4:
        x = np.concatenate([pos.flatten(), radii])
        pos_h, radii_h = _slsqp_optimize(x, deadline, max_iter=500)
        pos_pp, radii_pp = _post_process(pos_h, radii_h)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Phase 3: Another smooth pass from the polished result with very high penalty
    if time.time() < deadline - 3:
        x = np.concatenate([best_pos.flatten(), best_radii])
        pos_s, radii_s = _smooth_optimize(x, deadline, penalty_weight=20000.0, beta=200.0, max_iter=300)
        pos_pp, radii_pp = _post_process(pos_s, radii_s)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Phase 4: Final SLSQP pass from the best result
    if time.time() < deadline - 2:
        x = np.concatenate([best_pos.flatten(), best_radii])
        pos_h, radii_h = _slsqp_optimize(x, deadline, max_iter=400)
        pos_pp, radii_pp = _post_process(pos_h, radii_h)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _anneal(pos, radii, deadline, n_steps=5000, t0=0.02, t_end=0.0003, seed=0):
    """Simulated annealing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    def energy(p, r):
        bv = np.maximum(0.0, r - p[:, 0]) + np.maximum(0.0, p[:, 0] - (1.0 - r)) + \
             np.maximum(0.0, r - p[:, 1]) + np.maximum(0.0, p[:, 1] - (1.0 - r))
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        ov = np.maximum(0.0, min_dist[mask] - np.sqrt(np.maximum(dist_sq[mask], 0)))
        return -np.sum(r) + 100.0 * (np.sum(bv * bv) + np.sum(ov * ov))
    
    current_e = energy(pos, radii)
    
    for step in range(n_steps):
        if time.time() > deadline - 3:
            break
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.015)
        dy = rng.normal(0, temp * 0.015)
        dr = rng.normal(0, temp * 0.008)
        
        new_x = np.clip(pos[i, 0] + dx, 0.005, 0.995)
        new_y = np.clip(pos[i, 1] + dy, 0.005, 0.995)
        new_r = np.clip(radii[i] + dr, 0.005, 0.4)
        
        old_vals = (pos[i, 0], pos[i, 1], radii[i])
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        new_e = energy(pos, radii)
        delta = new_e - current_e
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-8)):
            current_e = new_e
        else:
            pos[i, 0], pos[i, 1], radii[i] = old_vals
        
        if step % 200 == 0:
            pp, rr = _post_process(pos, radii)
            if _is_valid(pp, rr):
                s = np.sum(rr)
                if s > best_score:
                    best_score = s
                    best_pos = pp.copy()
                    best_radii = rr.copy()
    
    return best_pos, best_radii

def _local_optimize(pos, radii, deadline, max_rounds=20000):
    """
    Iterative local optimization with efficient violation tracking.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    rng = np.random.RandomState(12345)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    def compute_violations(p, r):
        bv = np.maximum(0.0, r - p[:, 0]) + np.maximum(0.0, p[:, 0] - (1.0 - r)) + \
             np.maximum(0.0, r - p[:, 1]) + np.maximum(0.0, p[:, 1] - (1.0 - r))
        dx = p[:, 0, np.newaxis] - p[np.newaxis, :, 0]
        dy = p[:, 1, np.newaxis] - p[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        ov = np.maximum(0.0, min_dist[mask] - np.sqrt(np.maximum(dist_sq[mask], 0)))
        return np.sum(bv), np.sum(ov)
    
    # Cache current violation
    bv_cur, ov_cur = compute_violations(pos, radii)
    cur_penalty = 50.0
    cur_score = np.sum(radii) - cur_penalty * (bv_cur + ov_cur)
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 5:
            break
        
        # Try to grow the smallest circle
        min_idx = np.argmin(radii)
        max_r_here = _max_radius_at(pos[min_idx, 0], pos[min_idx, 1], pos, radii, idx=min_idx)
        if max_r_here > radii[min_idx] * 1.001:
            new_r = radii[min_idx] * 1.01
            if new_r <= max_r_here * 0.999:
                old_r = radii[min_idx]
                radii[min_idx] = new_r
                bv_new, ov_new = compute_violations(pos, radii)
                new_score = np.sum(radii) - cur_penalty * (bv_new + ov_new)
                if new_score > cur_score:
                    bv_cur, ov_cur = bv_new, ov_new
                    cur_score = new_score
                    if np.sum(radii) > best_score:
                        best_score = np.sum(radii)
                        best_pos = pos.copy()
                        best_radii = radii.copy()
                else:
                    radii[min_idx] = old_r
                continue
        
        # Random perturbation
        i = rng.randint(n)
        scale = 0.01 * (1.0 - round_num / max(max_rounds, 1) * 0.5)
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        dr = rng.normal(0, scale * 0.5)
        
        old_vals = (pos[i, 0], pos[i, 1], radii[i])
        pos[i, 0] = np.clip(pos[i, 0] + dx, 0.0, 1.0)
        pos[i, 1] = np.clip(pos[i, 1] + dy, 0.0, 1.0)
        radii[i] = np.clip(radii[i] + dr, 0.001, 0.45)
        
        bv_new, ov_new = compute_violations(pos, radii)
        new_score = np.sum(radii) - cur_penalty * (bv_new + ov_new)
        
        if new_score > cur_score:
            bv_cur, ov_cur = bv_new, ov_new
            cur_score = new_score
            if np.sum(radii) > best_score:
                best_score = np.sum(radii)
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            pos[i, 0], pos[i, 1], radii[i] = old_vals
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    
    # Generate diverse initializations
    inits = []
    pos_g, rad_g = _greedy_init(seed=42, n_trials=6000)
    inits.append((pos_g, rad_g))
    
    pos_h, rad_h = _hex_init()
    inits.append((pos_h, rad_h))
    
    pos_c, rad_c = _corner_init(seed=7)
    inits.append((pos_c, rad_c))
    
    pos_r1, rad_r1 = _random_init(100)
    inits.append((pos_r1, rad_r1))
    
    pos_r2, rad_r2 = _random_init(200)
    inits.append((pos_r2, rad_r2))
    
    # Pick the best initial solution
    best_pos = inits[0][0].copy()
    best_radii = inits[0][1].copy()
    best_score = np.sum(best_radii)
    
    for pos_i, rad_i in inits:
        pp, rr = _post_process(pos_i, rad_i)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()
    
    # Optimize from each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Smooth penalty-based optimization with increasing penalties
        for pw in [200.0, 1000.0, 3000.0, 8000.0]:
            if time.time() > deadline - 15:
                break
            pos_opt, radii_opt = _smooth_optimize(current_x, deadline, penalty_weight=pw, beta=50.0, max_iter=400)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    current_x = np.concatenate([pos_pp.flatten(), radii_pp])
        
        # SLSQP with hard constraints - more iterations
        if time.time() < deadline - 12:
            try:
                pos_hc, rad_hc = _slsqp_optimize(current_x, deadline, max_iter=300)
                pos_pp, radii_pp = _post_process(pos_hc, rad_hc)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Local optimization
        if time.time() < deadline - 10:
            pos_lo, rad_lo = _local_optimize(best_pos, best_radii, deadline, max_rounds=15000)
            pos_pp, radii_pp = _post_process(pos_lo, rad_lo)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Simulated annealing with more steps
        if time.time() < deadline - 8:
            try:
                pos_sa, rad_sa = _anneal(current_x[:2*N].reshape(N, 2), 
                                          np.maximum(current_x[2*N:], 1e-4),
                                          deadline, n_steps=4000, t0=0.015, t_end=0.0001,
                                          seed=idx * 100)
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
        if np.sum(rad_pol) > best_score:
            best_score = np.sum(rad_pol)
            best_pos = pos_pol.copy()
            best_radii = rad_pol.copy()
    
    # Final local optimization
    if time.time() < deadline - 5:
        pos_lo, rad_lo = _local_optimize(best_pos, best_radii, deadline, max_rounds=20000)
        pos_pp, radii_pp = _post_process(pos_lo, rad_lo)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Final validity check
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
