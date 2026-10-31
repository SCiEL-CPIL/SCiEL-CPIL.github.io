"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
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
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii, max_iter=500):
    """Shrink radii to ensure strict validity, preserving as much radius as possible."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    for _ in range(max_iter):
        valid = True
        for i in range(n):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-6)
                valid = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    if radii[i] <= radii[j]:
                        max_i = dist - radii[j]
                        if max_i < radii[i]:
                            radii[i] = max(max_i * 0.99999, 1e-6)
                            valid = False
                    else:
                        max_j = dist - radii[i]
                        if max_j < radii[j]:
                            radii[j] = max(max_j * 0.99999, 1e-6)
                            valid = False
        if valid:
            break
    # Final tight pass
    for i in range(n):
        x, y = pos[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if radii[i] > max_r:
            radii[i] = max(max_r * 0.999999, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                k = dist / (sum_r + 1e-12) * 0.999999
                if k < 1.0:
                    radii[i] = max(radii[i] * k, 1e-6)
                    radii[j] = max(radii[j] * k, 1e-6)
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

def _contact_graph(pos, radii, tol=0.005):
    """Find pairs of circles that are close to touching."""
    n = len(radii)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= radii[i] + radii[j] + tol:
                contacts.append((i, j))
    return contacts

def _solve_qp_radii(pos, contacts, max_r=0.45):
    """
    Solve QP to maximize sum of radii given positions and contact constraints.
    Uses scipy.optimize.minimize with SLSQP for the QP subproblem.
    """
    n = len(pos)
    
    # Boundary max radii
    max_r_bounds = np.array([min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]) for i in range(n)])
    
    # Distances for contact pairs
    contact_dists = []
    for (i, j) in contacts:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        contact_dists.append(dist)
    
    def obj(r):
        return -np.sum(r)
    
    def jac(r):
        return -np.ones(n)
    
    constraints = []
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: max_r_bounds[i] - r[i]})
    # Contact constraints
    for k, (i, j) in enumerate(contacts):
        d = contact_dists[k]
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i, j=j, d=d: d - r[i] - r[j]})
    
    bounds = [(0.0, max_r_bounds[i]) for i in range(n)]
    
    # Initial guess: equal split of boundary limits
    r0 = np.minimum(max_r_bounds, 0.15)
    
    try:
        result = minimize(obj, r0, method='SLSQP', jac=jac, bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12})
        if result.success:
            return np.maximum(result.x, 0.0)
    except Exception:
        pass
    
    # Fallback: greedy with iterative relaxation
    radii = max_r_bounds.copy()
    for _ in range(100):
        changed = False
        for (i, j) in contacts:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if radii[i] + radii[j] > dist:
                total = radii[i] + radii[j]
                scale = dist / total * 0.99999
                radii[i] *= scale
                radii[j] *= scale
                changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)

def _expand_positions(pos, radii, deadline, n_steps=200):
    """
    Try to expand positions to create more room for larger circles.
    Use a force-based approach: push circles apart when they're close.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(pos)
    rng = np.random.RandomState(42)
    
    for step in range(n_steps):
        if time.time() > deadline - 10:
            break
        
        # Compute forces
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                desired = radii[i] + radii[j] + 0.01  # Small gap
                if dist < desired:
                    # Push apart
                    f = (desired - dist) / dist
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
        
        # Also push away from boundaries
        for i in range(n):
            if pos[i, 0] < radii[i] + 0.01:
                forces[i, 0] += 0.5
            if pos[i, 0] > 1.0 - radii[i] - 0.01:
                forces[i, 0] -= 0.5
            if pos[i, 1] < radii[i] + 0.01:
                forces[i, 1] += 0.5
            if pos[i, 1] > 1.0 - radii[i] - 0.01:
                forces[i, 1] -= 0.5
        
        # Apply forces with damping
        lr = 0.05 * (1.0 - step / n_steps)
        pos += lr * forces
        pos = np.clip(pos, 0.0, 1.0)
    
    return pos

def _greedy_init(seed=42, n_trials=8000):
    """Greedy best-fit initialization with random sampling."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
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

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _smooth_objective(x, penalty_weight=1000.0, beta=50.0):
    """Smooth penalty objective for L-BFGS-B."""
    pos = x[:2 * N].reshape(N, 2)
    radii = np.maximum(x[2 * N:], 1e-6)
    
    bv_left = _soft_max(radii - pos[:, 0], beta)
    bv_right = _soft_max(pos[:, 0] - (1.0 - radii), beta)
    bv_bottom = _soft_max(radii - pos[:, 1], beta)
    bv_top = _soft_max(pos[:, 1] - (1.0 - radii), beta)
    boundary_pen = np.sum(bv_left * bv_left + bv_right * bv_right + bv_bottom * bv_bottom + bv_top * bv_top)
    
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    ov = _soft_max(min_dist_sq[mask] - dist_sq[mask], beta)
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
    """SLSQP with hard constraints."""
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

def _qp_polish(pos, radii, deadline, n_iter=50):
    """
    QP-based polish: iterate between expanding positions and solving QP for radii.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for it in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Get contact graph with tighter tolerance for more constraints
        contacts = _contact_graph(pos, radii, tol=0.008)
        
        # Solve QP for optimal radii given positions
        new_radii = _solve_qp_radii(pos, contacts)
        
        # Check validity
        pp, rr = _post_process(pos, new_radii)
        if _is_valid(pp, rr):
            score = np.sum(rr)
            if score > best_score:
                best_score = score
                best_pos = pp.copy()
                best_radii = rr.copy()
        
        # Try to expand positions to create more room
        if time.time() > deadline - 8:
            break
        pos_expanded = _expand_positions(pos, best_radii, deadline, n_steps=25)
        pos = pos_expanded
        
        # Re-solve QP with new positions
        contacts = _contact_graph(pos, best_radii, tol=0.008)
        new_radii = _solve_qp_radii(pos, contacts)
        pp, rr = _post_process(pos, new_radii)
        if _is_valid(pp, rr):
            score = np.sum(rr)
            if score > best_score:
                best_score = score
                best_pos = pp.copy()
                best_radii = rr.copy()
    
    return best_pos, best_radii

def _high_precision_polish(pos, radii, deadline):
    """
    High-precision polish: multiple SLSQP passes with increasing iterations
    and tight tolerances, interleaved with smooth L-BFGS-B for escape.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Phase 1: SLSQP with moderate iterations
    if time.time() < deadline - 10:
        x = np.concatenate([pos.flatten(), radii])
        pos_h, radii_h = _slsqp_optimize(x, deadline, max_iter=350)
        pos_pp, radii_pp = _post_process(pos_h, radii_h)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Phase 2: Smooth L-BFGS-B with high penalty to refine
    if time.time() < deadline - 8:
        x = np.concatenate([best_pos.flatten(), best_radii])
        for pw in [5000.0, 15000.0, 40000.0]:
            if time.time() > deadline - 6:
                break
            pos_s, radii_s = _smooth_optimize(x, deadline, penalty_weight=pw, beta=100.0, max_iter=250)
            pos_pp, radii_pp = _post_process(pos_s, radii_s)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    x = np.concatenate([best_pos.flatten(), best_radii])
    
    # Phase 3: Final SLSQP with maximum iterations
    if time.time() < deadline - 5:
        x = np.concatenate([best_pos.flatten(), best_radii])
        pos_h, radii_h = _slsqp_optimize(x, deadline, max_iter=500)
        pos_pp, radii_pp = _post_process(pos_h, radii_h)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Phase 4: QP polish
    if time.time() < deadline - 4:
        pos_qp, radii_qp = _qp_polish(best_pos, best_radii, deadline, n_iter=25)
        if np.sum(radii_qp) > best_score:
            best_score = np.sum(radii_qp)
            best_pos = pos_qp.copy()
            best_radii = radii_qp.copy()
    
    # Phase 5: Final SLSQP squeeze
    if time.time() < deadline - 2:
        x = np.concatenate([best_pos.flatten(), best_radii])
        pos_h, radii_h = _slsqp_optimize(x, deadline, max_iter=300)
        pos_pp, radii_pp = _post_process(pos_h, radii_h)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate diverse initializations
    inits = []
    pos_g, rad_g = _greedy_init(seed=42, n_trials=8000)
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
    
    # Optimize from each initialization (limit to top 4 by initial score)
    init_scores = []
    for idx, (pos_init, rad_init) in enumerate(inits):
        pp, rr = _post_process(pos_init, rad_init)
        if _is_valid(pp, rr):
            init_scores.append((np.sum(rr), idx, pp, rr))
    init_scores.sort(key=lambda x: -x[0])
    
    # Process top initializations
    for rank, (score_init, idx, pos_init, rad_init) in enumerate(init_scores[:4]):
        if time.time() > deadline - 25:
            break
        
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Smooth penalty-based optimization with increasing penalties
        for pw in [500.0, 2000.0, 8000.0]:
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
        
        # SLSQP with hard constraints
        if time.time() < deadline - 12:
            try:
                pos_hc, rad_hc = _slsqp_optimize(current_x, deadline, max_iter=400)
                pos_pp, radii_pp = _post_process(pos_hc, rad_hc)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # QP polish
        if time.time() < deadline - 10:
            pos_qp, radii_qp = _qp_polish(best_pos, best_radii, deadline, n_iter=35)
            if np.sum(radii_qp) > best_score:
                best_score = np.sum(radii_qp)
                best_pos = pos_qp.copy()
                best_radii = radii_qp.copy()
    
    # High-precision final polish
    if time.time() < deadline - 15:
        pos_pol, rad_pol = _high_precision_polish(best_pos, best_radii, deadline)
        if np.sum(rad_pol) > best_score:
            best_score = np.sum(rad_pol)
            best_pos = pos_pol.copy()
            best_radii = rad_pol.copy()
    
    # Final QP polish
    if time.time() < deadline - 5:
        pos_qp, radii_qp = _qp_polish(best_pos, best_radii, deadline, n_iter=25)
        if np.sum(radii_qp) > best_score:
            best_score = np.sum(radii_qp)
            best_pos = pos_qp.copy()
            best_radii = radii_qp.copy()
    
    # Final validity check
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Try one more quick fix
        contacts = _contact_graph(best_pos, best_radii, tol=0.01)
        new_radii = _solve_qp_radii(best_pos, contacts)
        pp, rr = _post_process(best_pos, new_radii)
        if _is_valid(pp, rr):
            best_pos, best_radii = pp, rr
        else:
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
