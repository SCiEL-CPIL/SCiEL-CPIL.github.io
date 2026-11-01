"""Packing of 50 equal circles in a unit square via soft-min optimization + SA + L-BFGS polish."""
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


def _soft_min_obj(pts, beta=50.0):
    """
    Soft-min objective: approximates -r (negate to maximize r).
    Uses log-sum-exp style soft-min over all constraints.
    Returns (objective, gradient) where objective = -soft_min.
    """
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    
    # All constraint values: pairwise/2 and wall distances
    iu = np.triu_indices(N, k=1)
    pair_vals = dist[iu] / 2.0
    all_vals = np.concatenate([pair_vals, d])
    
    # Soft-min via log-sum-exp: soft_min ≈ -log(sum(exp(-beta * x))) / beta
    # For numerical stability:
    m = np.min(all_vals)
    exp_args = -beta * (all_vals - m)
    exp_args = np.clip(exp_args, -700, 700)
    s = np.sum(np.exp(exp_args))
    soft_min_val = m - np.log(s) / beta
    
    # Gradient with respect to pts
    # d(soft_min)/d(pts_i) = sum_j w_j * d(constraint_j)/d(pts_i)
    # where w_j = exp(-beta * (x_j - m)) / s
    weights = np.exp(exp_args) / s
    
    grad = np.zeros((N, 2))
    
    # Pairwise contributions
    pair_weights = weights[:len(iu)]
    for k in range(len(iu)):
        i, j = iu[0][k], iu[1][k]
        d_ij = pts[i] - pts[j]
        dist_ij = dist[i, j]
        if dist_ij > 1e-12:
            # d(dist_ij/2)/d(pts_i) = (pts_i - pts_j) / (2 * dist_ij)
            w = pair_weights[k]
            grad[i] += w * d_ij / (2.0 * dist_ij)
            grad[j] -= w * d_ij / (2.0 * dist_ij)
    
    # Wall contributions
    wall_weights = weights[len(iu):]
    for i in range(N):
        w = wall_weights[i]
        # d(d_i)/d(pts_i)
        # d = min(HALF-x, HALF-y, HALF+x, HALF+y)
        dx = HALF - pts[i, 0]
        dy = HALF - pts[i, 1]
        dxp = HALF + pts[i, 0]
        dyp = HALF + pts[i, 1]
        dmin = min(dx, dy, dxp, dyp)
        if abs(dx - dmin) < 1e-10:
            grad[i, 0] -= w
        if abs(dy - dmin) < 1e-10:
            grad[i, 1] -= w
        if abs(dxp - dmin) < 1e-10:
            grad[i, 0] += w
        if abs(dyp - dmin) < 1e-10:
            grad[i, 1] += w
    
    return -soft_min_val, grad


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


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=3000):
    """Simulated annealing to minimize violation for target radius r_target."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    # Temperature schedule
    t_start = 0.03
    t_end = 0.00003
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        # Random move: pick 1-3 circles and perturb
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol


def _lbfgs_violation(pts0, r_target, deadline):
    """Use L-BFGS-B to minimize violation for target radius."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation(p, r_target)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        # Analytical gradient of violation
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        g = np.zeros_like(p)
        
        # Pairwise: d(max(0, 2r - dist))/d(p_i)
        # = -sign(2r - dist) * (p_i - p_j) / dist when 2r > dist
        iu = np.triu_indices(N, k=1)
        for k in range(len(iu)):
            i, j = iu[0][k], iu[1][k]
            d_ij = 2.0 * r_target - dist[i, j]
            if d_ij > 0 and dist[i, j] > 1e-12:
                d_ij = pts[i] - pts[j]
                dist_ij = dist[i, j]
                # d(2r - dist)/d(p_i) = -(p_i - p_j)/dist
                # d(max(0, 2r-dist))/d(p_i) = -(p_i-p_j)/dist when violated
                g[i] -= (p[i] - p[j]) / dist[i, j]
                g[j] += (p[i] - p[j]) / dist[i, j]
        
        # Wall: d(max(0, r - d_i))/d(p_i)
        for i in range(N):
            dx = HALF - p[i, 0]
            dy = HALF - p[i, 1]
            dxp = HALF + p[i, 0]
            dyp = HALF + p[i, 1]
            dmin = min(dx, dy, dxp, dyp)
            if r_target - dmin > 0:
                if abs(dx - dmin) < 1e-10:
                    g[i, 0] += 1.0
                if abs(dy - dmin) < 1e-10:
                    g[i, 1] += 1.0
                if abs(dxp - dmin) < 1e-10:
                    g[i, 0] -= 1.0
                if abs(dyp - dmin) < 1e-10:
                    g[i, 1] -= 1.0
        
        return g.ravel()
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts, float(res.fun)


def _lbfgs_soft_min(pts0, deadline, beta=50.0):
    """Use L-BFGS-B to maximize r via soft-min objective."""
    from scipy.optimize import minimize
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        val, _ = _soft_min_obj(p, beta)
        return val
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        _, g = _soft_min_obj(p, beta)
        return g.ravel()
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
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


def _binary_search_radius(pts_start, deadline, rng):
    """Binary search on radius with violation minimization."""
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    r_lo = best_r
    r_hi = 0.07138
    
    for _ in range(12):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to find feasible configuration at r_mid
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_mid, deadline - 5.0, rng, n_steps=3000)
        
        # Polish with L-BFGS
        pts_lbfgs, viol_lbfgs = _lbfgs_violation(pts_sa, r_mid, deadline - 5.0)
        
        # Also try soft-min polish
        if time.time() < deadline - 3.0:
            pts_soft = _lbfgs_soft_min(pts_lbfgs, deadline - 3.0, beta=80.0)
            r_soft = _compute_r(pts_soft)
            if r_soft > _compute_r(pts_lbfgs):
                pts_lbfgs = pts_soft
        
        # Check if feasible
        r_achieved = _compute_r(pts_lbfgs)
        
        if viol_lbfgs < 1e-7 and r_achieved >= r_mid - 1e-9:
            r_lo = r_mid
            best_pts = pts_lbfgs.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s limit
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps):
        nonlocal best_pts, best_r
        if time.time() > deadline - 10.0:
            return
        
        # SA phase
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - 10.0, rng, n_steps=n_sa_steps)
        
        # L-BFGS polish (violation-based)
        try:
            pts_lbfgs, viol_lbfgs = _lbfgs_violation(pts_sa, r_target, deadline - 10.0)
        except Exception:
            pts_lbfgs = pts_sa
            viol_lbfgs = viol_sa
        
        # Soft-min polish (directly maximizes r)
        if time.time() < deadline - 5.0:
            try:
                pts_soft = _lbfgs_soft_min(pts_lbfgs, deadline - 5.0, beta=60.0)
                r_soft = _compute_r(pts_soft)
                if r_soft > _compute_r(pts_lbfgs):
                    pts_lbfgs = pts_soft
            except Exception:
                pass
        
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
        if time.time() > deadline - 30.0:
            break
        try_improve(init, 0.07, 2500)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search on radius
    if time.time() < deadline - 20.0:
        best_pts, best_r = _binary_search_radius(best_pts, deadline, rng)
    
    # Phase 3: Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        r_target = best_r * 1.0005
        try_improve(best_pts, r_target, 3500)
    
    # Phase 4: Very final polish
    if time.time() < deadline - 2.0:
        r_target = best_r * 1.0002
        try_improve(best_pts, r_target, 2500)
    
    return best_pts
