"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Strict validity check."""
    n = len(radii)
    if np.any(radii < 0):
        return False
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
    
    for _ in range(500):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-14) * 0.99999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-6)
                        radii[j] = max(radii[j] * k, 1e-6)
                        valid = False
        if valid:
            break
    return pos, radii

def _lp_radius_optimize(pos):
    """Solve for maximum radii given fixed positions using LP."""
    n = len(pos)
    c = -np.ones(n)
    A_rows = []
    b_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(d)
    for i in range(n):
        w = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(w)
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0, None)] * n
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def _slsqp_constraints(n):
    """Generate constraint functions for SLSQP."""
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i]})
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 - (x[2*n+i] + x[2*n+j])**2
            })
    return constraints

def _objective(x, penalty_weight=1000.0):
    """Penalty-based objective for L-BFGS-B."""
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    bv = (np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) +
          np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii)))
    boundary_pen = np.sum(bv * bv)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0.0, min_dist * min_dist - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

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
        for _ in range(8000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r:
                    max_r = possible_r
            if max_r > best_r and max_r > 0.003:
                best_r = max_r
                best_x = x
                best_y = y
                found = True
        if not found or best_r < 0.003:
            best_x, best_y, best_r = 0.5, 0.5, 0.003
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.98
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
        for _ in range(200):
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
            if edge == 0: x, y = t, 0.08
            elif edge == 1: x, y = t, 0.92
            elif edge == 2: x, y = 0.08, t
            else: x, y = 0.92, t
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
        radii[i] = max(max_r, 0.01) * 0.96
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=30000, t_init=0.01, t_min=1e-7, cooling=0.9997, seed=0, deadline=None):
    """Simulated annealing with incremental energy evaluation."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    t = t_init
    step = 0
    while step < n_steps:
        if deadline is not None and time.time() > deadline:
            break
        i = rng.randint(n)
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        scale = 0.01 * t / t_init + 0.001
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        dr = rng.normal(0, scale * 0.5)
        new_x = np.clip(old_x + dx, 0.001, 0.999)
        new_y = np.clip(old_y + dy, 0.001, 0.999)
        new_r = np.clip(old_r + dr, 0.001, 0.4)
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        # Incremental energy
        new_bv = (np.maximum(0.0, new_r - new_x) + np.maximum(0.0, new_x - (1.0 - new_r)) +
                  np.maximum(0.0, new_r - new_y) + np.maximum(0.0, new_y - (1.0 - new_r)))
        new_overlap = 0.0
        for j in range(n):
            if j == i: continue
            dx3 = new_x - pos[j, 0]
            dy3 = new_y - pos[j, 1]
            dist_sq3 = dx3 * dx3 + dy3 * dy3
            min_dist3 = new_r + radii[j]
            viol3 = max(0.0, min_dist3 * min_dist3 - dist_sq3)
            new_overlap += viol3 * viol3
        new_contrib = 50.0 * (new_bv * new_bv + new_overlap)
        old_bv = (np.maximum(0.0, old_r - old_x) + np.maximum(0.0, old_x - (1.0 - old_r)) +
                  np.maximum(0.0, old_r - old_y) + np.maximum(0.0, old_y - (1.0 - old_r)))
        old_overlap = 0.0
        for j in range(n):
            if j == i: continue
            dx2 = old_x - pos[j, 0]
            dy2 = old_y - pos[j, 1]
            dist_sq2 = dx2 * dx2 + dy2 * dy2
            min_dist2 = old_r + radii[j]
            viol2 = max(0.0, min_dist2 * min_dist2 - dist_sq2)
            old_overlap += viol2 * viol2
        old_contrib = 50.0 * (old_bv * old_bv + old_overlap)
        delta_energy = new_contrib - old_contrib
        delta_score = new_r - old_r
        delta_total = delta_energy - delta_score
        if delta_total < 0 or rng.random() < np.exp(-delta_total / max(t, 1e-10)):
            pass
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        t *= cooling
        t = max(t, t_min)
        step += 1
        if step % 200 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    pos_pp, radii_pp = _post_process(pos, radii)
    if _is_valid(pos_pp, radii_pp):
        score = np.sum(radii_pp)
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    return best_pos, best_radii

def _polish(pos, radii, deadline):
    """High-precision polishing with SLSQP and LP-based local search."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    x = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.5))
    # SLSQP with hard constraints
    if time.time() < deadline - 15:
        try:
            cons = _slsqp_constraints(N)
            result = minimize(lambda x: -np.sum(np.maximum(x[2*N:], 1e-6)), x, method='SLSQP',
                              bounds=bounds, constraints=cons,
                              options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.success:
                x_new = result.x
                pos_slq = x_new[:2*N].reshape(N, 2)
                rad_slq = np.maximum(x_new[2*N:], 1e-5)
                pos_pp, radii_pp = _post_process(pos_slq, rad_slq)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass
    # LP-based local search
    if time.time() < deadline - 5:
        for it in range(200):
            if time.time() > deadline - 5:
                break
            new_radii = _lp_radius_optimize(best_pos)
            if new_radii is None:
                break
            new_radii_safe = np.maximum(new_radii * 0.999999, 1e-6)
            if _is_valid(best_pos, new_radii_safe):
                score = np.sum(new_radii_safe)
                if score > best_score:
                    best_score = score
                    best_radii = new_radii_safe.copy()
                else:
                    break
            else:
                break
            # Compute forces from tight constraints
            dx = best_pos[:, 0, np.newaxis] - best_pos[np.newaxis, :, 0]
            dy = best_pos[:, 1, np.newaxis] - best_pos[np.newaxis, :, 1]
            dists = np.sqrt(dx*dx + dy*dy)
            forces = np.zeros((N, 2))
            for i in range(N):
                for j in range(i+1, N):
                    sum_r = best_radii[i] + best_radii[j]
                    d = dists[i, j]
                    if d > 0 and sum_r > d * 0.93:
                        strength = (sum_r / d) ** 5
                        fx = strength * dx[i, j]
                        fy = strength * dy[i, j]
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            for i in range(N):
                if best_pos[i, 0] < best_radii[i] * 1.1: forces[i, 0] += 1.0
                if best_pos[i, 0] > 1.0 - best_radii[i] * 1.1: forces[i, 0] -= 1.0
                if best_pos[i, 1] < best_radii[i] * 1.1: forces[i, 1] += 1.0
                if best_pos[i, 1] > 1.0 - best_radii[i] * 1.1: forces[i, 1] -= 1.0
            lr = 0.003
            best_pos += lr * forces
            best_pos = np.clip(best_pos, 0.01, 0.99)
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    for i in range(4):
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
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        # L-BFGS-B with escalating penalties
        for p_w in [200.0, 1000.0, 5000.0]:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
                if result.fun < _objective(current_x, p_w) - 1e-12:
                    current_x = result.x
            except Exception:
                pass
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-5)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        # Simulated annealing
        if time.time() < deadline - 15:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-5),
                    n_steps=25000, t_init=0.008, t_min=1e-7, cooling=0.9998,
                    seed=idx * 200 + 50, deadline=deadline - 12
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish on best
    if time.time() < deadline - 10:
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
    
    # Second polish with perturbation
    if time.time() < deadline - 8:
        try:
            rng = np.random.RandomState(12345)
            perturbed_pos = best_pos + rng.normal(0, 0.002, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
            pos_pol2, rad_pol2 = _polish(perturbed_pos, best_radii.copy(), deadline)
            pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
            score2 = np.sum(radii_pp2)
            if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                best_score = score2
                best_pos = pos_pp2.copy()
                best_radii = radii_pp2.copy()
        except Exception:
            pass
    
    # Third polish with larger perturbation
    if time.time() < deadline - 6:
        try:
            rng = np.random.RandomState(99999)
            perturbed_pos = best_pos + rng.normal(0, 0.005, (N, 2))
            perturbed_pos = np.clip(perturbed_pos, 0.01, 0.99)
            pos_pol3, rad_pol3 = _polish(perturbed_pos, best_radii.copy(), deadline)
            pos_pp3, radii_pp3 = _post_process(pos_pol3, rad_pol3)
            score3 = np.sum(radii_pp3)
            if _is_valid(pos_pp3, radii_pp3) and score3 > best_score:
                best_score = score3
                best_pos = pos_pp3.copy()
                best_radii = radii_pp3.copy()
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
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
