"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

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

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    Uses a soft-min (log-sum-exp) surrogate for the distance to the boundary
    and a smooth max(0, required - dist) for overlaps, which is more
    differentiable than the squared-penalty objective.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)

    # Smooth boundary: soft-min of the four wall distances
    d = np.stack([pos[:, 0], 1.0 - pos[:, 0], pos[:, 1], 1.0 - pos[:, 1]], axis=1)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -50.0, 50.0)
    soft_max_neg = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -soft_max_neg[:, 0]
    boundary_pen = np.sum(_soft_max(radii - soft_min_d, beta))

    # Smooth overlap: for each pair, required = r_i + r_j, dist = ||p_i - p_j||
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
    """Shrink radii to ensure strict validity (vectorized)."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(100):
        valid = True
        
        # Boundary: vectorized
        max_r_wall = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                                np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
        mask_b = radii > max_r_wall
        if np.any(mask_b):
            radii[mask_b] = np.maximum(max_r_wall[mask_b] * 0.9999, 1e-5)
            valid = False
        
        # Overlaps: vectorized
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_mask = (dist < sum_r) & mask
        if np.any(overlap_mask):
            # For each overlapping pair, scale both radii
            ii, jj = np.where(overlap_mask)
            for i, j in zip(ii, jj):
                d = dist[i, j]
                sr = sum_r[i, j]
                k = d / (sr + 1e-12) * 0.9999
                if k < 1.0:
                    radii[i] = max(radii[i] * k, 1e-5)
                    radii[j] = max(radii[j] * k, 1e-5)
            valid = False
        
        if valid:
            break
            
    return pos, radii

def _exact_radius_expand(pos, radii_init, max_iter=200):
    """
    Given fixed positions, find the maximum feasible radii for all circles
    using an iterative water-filling approach. This ensures no overlaps
    and maximizes the sum of radii for the given positions.
    """
    pos = pos.copy()
    radii = np.maximum(radii_init, 1e-6).copy()
    n = len(radii)
    
    # Precompute distances
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    wall_r = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                        np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    
    # Iterative expansion
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            # Max radius for circle i given all others
            max_r = wall_r[i]
            for j in range(n):
                if j == i:
                    continue
                available = dist[i, j] - radii[j]
                if available < max_r:
                    max_r = available
            
            if max_r < 1e-6:
                max_r = 1e-6
                
            if abs(radii[i] - max_r) > 1e-12:
                radii[i] = max_r
                changed = True
        
        if not changed:
            break
            
    return radii

def _jitter_and_expand(pos, radii, n_iter=100, deadline=None, seed=0):
    """
    Stochastic search: Jitter positions, then expand radii to maximum.
    This decouples topology search (positions) from geometry optimization (radii).
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    n = len(radii)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline - 1.0:
            break
            
        # Jitter: perturb positions slightly
        # Scale jitter by current radius to maintain relative spacing
        jitter_scale = np.mean(radii) * 0.1
        pos_jittered = pos + rng.normal(0, jitter_scale, (n, 2))
        
        # Clamp to valid bounds (roughly)
        pos_jittered[:, 0] = np.clip(pos_jittered[:, 0], 0.01, 0.99)
        pos_jittered[:, 1] = np.clip(pos_jittered[:, 1], 0.01, 0.99)
        
        # Expand: find max radii for these positions
        radii_new = _exact_radius_expand(pos_jittered, radii)
        
        # Check validity and score
        if _is_valid(pos_jittered, radii_new):
            score = np.sum(radii_new)
            if score > best_score:
                best_score = score
                best_pos = pos_jittered.copy()
                best_radii = radii_new.copy()
                pos = best_pos.copy() # Accept and continue from here
                radii = best_radii.copy()
                
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

def _incremental_energy(pos, radii, changed_idx, penalty_weight=50.0):
    """
    Fast incremental energy computation for a single changed circle.
    Only recomputes terms involving the changed circle.
    """
    n = len(radii)
    x, y, r = pos[changed_idx, 0], pos[changed_idx, 1], radii[changed_idx]
    
    # Boundary penalty for changed circle
    bv = np.maximum(0.0, r - x) + np.maximum(0.0, x - (1.0 - r)) + \
         np.maximum(0.0, r - y) + np.maximum(0.0, y - (1.0 - r))
    boundary_pen = bv * bv
    
    # Overlap penalty with all other circles
    overlap_pen = 0.0
    for j in range(n):
        if j == changed_idx:
            continue
        dx = x - pos[j, 0]
        dy = y - pos[j, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = r + radii[j]
        viol = max(0.0, min_dist * min_dist - dist_sq)
        overlap_pen += viol * viol
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _full_energy(pos, radii, penalty_weight=50.0):
    """Full energy computation (same as _objective but returns penalty only)."""
    n = len(radii)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return penalty_weight * (boundary_pen + overlap_pen)

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0, deadline=None):
    """
    Simulated annealing with full-energy evaluation (N=26 is small enough).
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Current energy (penalty-based)
    current_penalty = _full_energy(pos, radii, 50.0)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Perturb: move one circle slightly
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.02)
        dy = rng.normal(0, temp * 0.02)
        dr = rng.normal(0, temp * 0.01)
        
        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr
        
        # Clamp
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)
        
        # Try the move
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        new_energy = _full_energy(pos, radii, 50.0)
        
        delta_e = new_energy - current_penalty
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_penalty = new_energy
        else:
            # Reject
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        
        # Periodically check validity and update best
        if step % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    # Final check
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_constraints(pos, radii):
    """Generate constraint functions for SLSQP with hard non-overlap constraints."""
    n = len(radii)
    constraints = []
    
    # Boundary constraints: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*n+i] + x[2*n+j])**2
            })
    
    return constraints

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP hard constraints and exact radius expansion."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
        
    # SLSQP with hard constraints
    try:
        cons = _slsqp_constraints(pos, radii)
        result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x, method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success or result.fun < -np.sum(np.maximum(x[2*N:], 1e-6)) + 1e-10:
            x = result.x
    except Exception:
        pass
    
    # Exact radius expansion on the polished positions
    if time.time() < deadline - 1:
        try:
            pos_q = x[:2*N].reshape(N, 2)
            rad_q = np.maximum(x[2*N:], 1e-6)
            rad_opt = _exact_radius_expand(pos_q, rad_q)
            # Verify validity
            pos_pp, rad_pp = _post_process(pos_q, rad_opt)
            if _is_valid(pos_pp, rad_pp) and np.sum(rad_pp) > np.sum(np.maximum(x[2*N:], 1e-6)):
                x = np.concatenate([pos_pp.flatten(), rad_pp])
        except Exception:
            pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 1e-4)
    
    return pos_out, radii_out

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
    
    # Use Jitter and Expand strategy for each initialization
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # 1. Jitter and Expand to find a good local topology
        pos_j, rad_j = _jitter_and_expand(pos_init, rad_init, n_iter=200, deadline=deadline, seed=idx*100)
        
        pos_pp, radii_pp = _post_process(pos_j, rad_j)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
            
        # 2. Polish the best found so far
        if time.time() < deadline - 5:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp2, radii_pp2 = _post_process(pos_pol, rad_pol)
            score2 = np.sum(radii_pp2)
            if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = radii_pp2.copy()
    
    # Final polish and validation
    if time.time() < deadline - 2:
        pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
        pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
        score = np.sum(radii_pp)
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
            
    # Final validation and exact radius squeeze
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # One last exact radius optimization on the final positions
    try:
        rad_final = _exact_radius_expand(best_pos, best_radii)
        pos_fv, rad_fv = _post_process(best_pos, rad_final)
        if _is_valid(pos_fv, rad_fv) and np.sum(rad_fv) > np.sum(best_radii):
            best_pos = pos_fv
            best_radii = rad_fv
    except Exception:
        pass
    
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
