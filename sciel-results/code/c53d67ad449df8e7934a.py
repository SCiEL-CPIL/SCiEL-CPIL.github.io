"""Packing of 50 equal circles in a unit square via soft-violation L-BFGS + SA + binary search."""
import numpy as np
import time

N = 50
HALF = 0.5


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _soft_min_radius(pts, beta=50.0):
    """Smooth soft-minimum surrogate for the radius using log-sum-exp."""
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    all_d = np.concatenate([d_left, d_right, d_bottom, d_top, pair_dists])
    
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = max_d - log_sum / beta
    
    return float(r_soft)


def _soft_min_radius_grad(pts, beta=50.0):
    """Gradient of the soft-minimum radius surrogate (analytic)."""
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    
    all_d = np.concatenate([d_left, d_right, d_bottom, d_top, pair_dists])
    
    max_d = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - max_d))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    grad[:, 0] += weights[:N]
    grad[:, 0] -= weights[N:2*N]
    grad[:, 1] += weights[2*N:3*N]
    grad[:, 1] -= weights[3*N:4*N]
    
    # Vectorized pairwise gradient
    i_idx = iu[0]
    j_idx = iu[1]
    w_pair = weights[4*N:]
    
    # Only process pairs with significant weight
    mask = w_pair > 1e-12
    if np.any(mask):
        im = i_idx[mask]
        jm = j_idx[mask]
        wm = w_pair[mask]
        
        dij = pts[im] - pts[jm]  # (M, 2)
        dists = np.linalg.norm(dij, axis=1)  # (M,)
        safe_dists = np.where(dists > 1e-15, dists, 1.0)
        unit = dij / safe_dists[:, None]  # (M, 2)
        coeff = wm / (2.0 * safe_dists)  # (M,)
        
        # Accumulate: grad[i] += coeff * unit, grad[j] -= coeff * unit
        np.add.at(grad, im, coeff[:, None] * unit)
        np.add.at(grad, jm, -coeff[:, None] * unit)
    
    return grad


def _soft_violation_obj(pts, r_target, beta=20.0):
    """Smooth soft-violation: sum of soft-max(0, 2r - dist) for pairs + walls.
    
    Uses softplus approximation: softmax(0, x) ≈ log(1 + exp(beta*x)) / beta
    """
    # Wall violations: r_target - d_wall
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_excess = np.concatenate([
        r_target - d_left,
        r_target - d_right,
        r_target - d_bottom,
        r_target - d_top
    ])
    
    # Pair violations: 2*r_target - dist
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_excess = 2.0 * r_target - dist[iu]
    
    all_excess = np.concatenate([wall_excess, pair_excess])
    
    # Softplus: log(1 + exp(beta * x)) / beta, stable version
    # For large positive x: ~x; for large negative x: ~exp(beta*x)/beta
    b = beta
    safe = np.where(b * all_excess > 50.0, b * all_excess, np.log1p(np.exp(b * all_excess)))
    return float(np.sum(safe) / b)


def _soft_violation_grad(pts, r_target, beta=20.0):
    """Analytic gradient of the soft-violation objective."""
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_excess = np.concatenate([
        r_target - d_left,
        r_target - d_right,
        r_target - d_bottom,
        r_target - d_top
    ])
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_excess = 2.0 * r_target - dist[iu]
    
    all_excess = np.concatenate([wall_excess, pair_excess])
    b = beta
    
    # sigmoid(b * x) = 1 / (1 + exp(-b*x)), stable
    bx = b * all_excess
    sig = np.where(bx > 0, 1.0 / (1.0 + np.exp(-bx)), np.exp(bx) / (1.0 + np.exp(bx)))
    
    grad = np.zeros((N, 2))
    
    # Wall gradients
    # d_left[i] = x_i + HALF, excess = r - d_left, d(excess)/d(x_i) = -1
    # d(softplus)/d(excess) = sig, so d(obj)/d(x_i) = -sig * (1/b) * b = -sig
    grad[:, 0] -= sig[:N]
    grad[:, 0] += sig[N:2*N]
    grad[:, 1] -= sig[2*N:3*N]
    grad[:, 1] += sig[3*N:4*N]
    
    # Pair gradients
    i_idx = iu[0]
    j_idx = iu[1]
    sig_pair = sig[4*N:]
    
    mask = sig_pair > 1e-12
    if np.any(mask):
        im = i_idx[mask]
        jm = j_idx[mask]
        sm = sig_pair[mask]
        
        dij = pts[im] - pts[jm]
        dists = np.linalg.norm(dij, axis=1)
        safe_dists = np.where(dists > 1e-15, dists, 1.0)
        unit = dij / safe_dists[:, None]
        
        # d(pair_excess)/d(pts[i]) = -unit (since excess = 2r - |dij|, d|dij|/d(pts[i]) = unit)
        # d(obj)/d(pts[i]) = sig * (-unit) ... wait let me re-derive
        # obj = sum softplus(b * excess) / b
        # d(obj)/d(pts[i]) = sum_j sig(b*excess_ij) * b * d(excess_ij)/d(pts[i]) / b
        #                  = sum_j sig(b*excess_ij) * d(excess_ij)/d(pts[i])
        # excess_ij = 2r - |pts[i]-pts[j]|
        # d(excess_ij)/d(pts[i]) = -(pts[i]-pts[j])/|pts[i]-pts[j]| = -unit
        # d(excess_ij)/d(pts[j]) = +(pts[i]-pts[j])/|pts[i]-pts[j]| = +unit
        
        np.add.at(grad, im, -sm[:, None] * unit)
        np.add.at(grad, jm, sm[:, None] * unit)
    
    return grad


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist)
    iu = np.triu_indices(N, k=1)
    pair_viol = pair_viol[iu]
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _contact_graph(pts, r_target, tolerance=1e-6):
    """Identify contact pairs: pairs where distance is within 2*r_target of contact."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Contact if dist is close to 2*r_target (within tolerance)
    contacts = np.abs(dist - 2.0 * r_target) < tolerance
    iu = np.triu_indices(N, k=1)
    contact_pairs = [(iu[0][k], iu[1][k]) for k in range(len(iu[0])) if contacts[iu[0][k], iu[1][k]]]
    return contact_pairs


def _wall_contacts(pts, r_target, tolerance=1e-6):
    """Identify wall contacts: points where distance to wall is close to r_target."""
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    
    wall_contacts = []
    for i in range(N):
        if abs(d_left[i] - r_target) < tolerance:
            wall_contacts.append((i, 'left'))
        if abs(d_right[i] - r_target) < tolerance:
            wall_contacts.append((i, 'right'))
        if abs(d_bottom[i] - r_target) < tolerance:
            wall_contacts.append((i, 'bottom'))
        if abs(d_top[i] - r_target) < tolerance:
            wall_contacts.append((i, 'top'))
    return wall_contacts


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=5000):
    """Simulated annealing to minimize violation for target radius r_target with adaptive step size."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    # Adaptive temperature schedule
    t_start = 0.05
    t_end = 0.0001
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    accept_count = 0
    recent_accepts = 0
    window = 200
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Adaptive step size: scale perturbation with current violation
        scale = max(temp, 0.005 * np.sqrt(max(current_viol, 1e-10)))
        
        # Random move: pick 1-3 circles and perturb
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, scale, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            accept_count += 1
            recent_accepts += 1
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
        
        # Adaptive restart: if stuck, increase temperature temporarily
        if step % window == 0 and step > 0:
            if recent_accepts < window * 0.1:
                # Stuck: boost temperature
                temp = t_start * 0.5
                recent_accepts = 0
        
        if step % window == 0:
            recent_accepts = 0
    
    return best_pts, best_viol


def _lbfgs_soft_violation(pts0, r_target, deadline, beta=20.0):
    """Use L-BFGS-B to minimize soft-violation for target radius (analytic gradient)."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _soft_violation_obj(p, r_target, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _soft_violation_grad(p, r_target, beta).ravel()
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(_violation(pts, r_target))


def _lbfgs_soft_min(pts0, deadline, beta_start=20.0, beta_end=200.0, n_iters=5):
    """Use L-BFGS-B to maximize the soft-minimum radius surrogate with increasing beta."""
    from scipy.optimize import minimize
    
    x = pts0.ravel().copy()
    best_x = x.copy()
    best_val = _soft_min_radius(pts0, beta_start)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iters - 1, 1))
        
        def neg_obj(xx, b=beta):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            p = xx.reshape(N, 2)
            return -_soft_min_radius(p, b)
        
        def neg_grad(xx, b=beta):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            p = xx.reshape(N, 2)
            return -_soft_min_radius_grad(p, b).ravel()
        
        try:
            res = minimize(neg_obj, x, method='L-BFGS-B',
                          jac=neg_grad,
                          bounds=[(-HALF, HALF)] * (2 * N),
                          options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
            x_new = np.clip(res.x, -HALF, HALF)
            val = _soft_min_radius(x_new.reshape(N, 2), beta)
            if val > best_val:
                best_val = val
                best_x = x_new.copy()
            x = x_new
        except Exception:
            break
    
    pts = best_x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _refine_radius(pts_start, deadline, rng):
    """Binary search on radius with soft-violation L-BFGS polish."""
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    r_lo = best_r
    r_hi = 0.071377103865  # Known best
    
    n_iters = 12
    for it in range(n_iters):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        remaining = deadline - time.time()
        
        # SA to get close to feasible
        sa_budget = min(remaining * 0.4, 30.0)
        n_sa = int(3000 * min(1.0, sa_budget / 15.0))
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_mid, time.time() + sa_budget, rng, n_steps=max(n_sa, 500))
        
        # L-BFGS soft-violation polish
        try:
            pts_lbfgs, viol_lbfgs = _lbfgs_soft_violation(pts_sa, r_mid, deadline - 2.0, beta=20.0)
        except Exception:
            pts_lbfgs = pts_sa
            viol_lbfgs = viol_sa
        
        r_achieved = _compute_r(pts_lbfgs)
        
        if viol_lbfgs < 1e-8 and r_achieved >= r_mid - 1e-9:
            r_lo = r_mid
            best_pts = pts_lbfgs.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def _high_precision_polish(pts, deadline):
    """Final high-precision polish using very high-beta soft-min L-BFGS."""
    from scipy.optimize import minimize
    
    best_pts = pts.copy()
    best_r = _compute_r(pts)
    
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline - 2.0:
            break
        
        x = best_pts.ravel().copy()
        
        def neg_obj(xx, b=beta):
            if time.time() > deadline - 1.0:
                raise RuntimeError("timeout")
            p = xx.reshape(N, 2)
            return -_soft_min_radius(p, b)
        
        def neg_grad(xx, b=beta):
            if time.time() > deadline - 1.0:
                raise RuntimeError("timeout")
            p = xx.reshape(N, 2)
            return -_soft_min_radius_grad(p, b).ravel()
        
        try:
            res = minimize(neg_obj, x, method='L-BFGS-B',
                          jac=neg_grad,
                          bounds=[(-HALF, HALF)] * (2 * N),
                          options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14})
            x_new = np.clip(res.x, -HALF, HALF)
            r_new = _compute_r(x_new.reshape(N, 2))
            if r_new > best_r:
                best_r = r_new
                best_pts = x_new.reshape(N, 2).copy()
        except Exception:
            break
    
    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s limit
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps):
        nonlocal best_pts, best_r
        if time.time() > deadline - 10.0:
            return
        
        # SA phase
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - 10.0, rng, n_steps=n_sa_steps)
        
        # L-BFGS soft-violation polish
        try:
            pts_lbfgs, viol_lbfgs = _lbfgs_soft_violation(pts_sa, r_target, deadline - 5.0, beta=20.0)
        except Exception:
            pts_lbfgs = pts_sa
            viol_lbfgs = viol_sa
        
        r = _compute_r(pts_lbfgs)
        if r > best_r:
            best_r = r
            best_pts = pts_lbfgs.copy()
    
    # Phase 1: Multi-start with moderate target
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        try_improve(init, 0.07, 3000)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search on radius with soft-violation polish
    if time.time() < deadline - 25.0:
        best_pts, best_r = _refine_radius(best_pts, deadline, rng)
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 10.0:
        try:
            best_pts, best_r = _high_precision_polish(best_pts, deadline - 3.0)
        except Exception:
            pass
    
    # Phase 4: One more binary search iteration if time permits
    if time.time() < deadline - 8.0:
        try:
            best_pts, best_r = _refine_radius(best_pts, deadline, rng)
        except Exception:
            pass
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            best_pts, best_r = _high_precision_polish(best_pts, deadline - 1.0)
        except Exception:
            pass
    
    return best_pts
