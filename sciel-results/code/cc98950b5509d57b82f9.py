"""Packing of 50 equal circles in a unit square via contact-graph Gauss-Seidel + SA + L-BFGS."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = time.time() + 145.0  # safety margin before 150s


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


def _soft_min_obj(pts, t):
    """Soft-min of all constraint values (pairwise/2 and boundary distances)."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    vals = np.concatenate([d, pair])
    m = np.min(vals)
    # Use logsumexp for stability
    s = np.sum(np.exp(-t * (vals - m)))
    return m - np.log(s) / t


def _soft_min_grad(pts, t):
    """Analytical gradient of the soft-min objective w.r.t. pts."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    vals = np.concatenate([d, pair])
    m = np.min(vals)
    exps = np.exp(-t * (vals - m))
    weights = exps / np.sum(exps)  # softmax weights

    grad = np.zeros((N, 2))

    # Boundary contributions
    # d_i = min(HALF-x, HALF-y, HALF+x, HALF+y)
    # The gradient of min is the gradient of the active constraint
    # For soft-min, we weight by the softmax weights
    # d_i's gradient: which edge is closest?
    # Actually, d = min of 4 linear functions, so gradient is the gradient of the min
    # But soft-min smooths this. Let's compute properly.
    # d_i = min(HALF - x_i, HALF - y_i, HALF + x_i, HALF + y_i)
    # grad of (HALF - x) = (-1, 0), (HALF - y) = (0, -1), (HALF + x) = (1, 0), (HALF + y) = (0, 1)
    # The soft-min weight for each of the 4 boundary constraints for point i
    # But we combined all into one vector. Let's separate.
    
    # Actually, let me recompute. The vals vector is [d_0, ..., d_{N-1}, pair_01, pair_02, ...]
    # For boundary d_i, the gradient w.r.t. x_i depends on which edge is active.
    # For the soft-min, the gradient of d_i (as a smooth function) is:
    # We need to be more careful. Let's use the fact that d_i = min of 4 linear terms.
    # The soft-min over ALL constraints (boundary + pairwise) is what we're optimizing.
    
    # For boundary term d_i:
    # The 4 candidate values are: HALF-x_i, HALF-y_i, HALF+x_i, HALF+y_i
    # But in our vals array, d_i is already the min of these 4.
    # The gradient of d_i w.r.t. x_i is:
    #   -1 if HALF-x_i is the min
    #   +1 if HALF+x_i is the min
    #   0 otherwise
    # For soft-min, we weight by softmax. But the softmax is over ALL vals, not just the 4 boundary terms for one point.
    # This is getting complex. Let's use a simpler approach: numerical gradient is too slow.
    # Let's just compute the gradient of the soft-min properly.
    
    # The soft-min of vals is: f = m - log(sum(exp(-t*(vals-m))))/t
    # df/d(vals_k) = -exp(-t*(vals_k-m)) / sum(exp(-t*(vals-m))) * (-t) / t = exp(-t*(vals_k-m)) / sum(exp(-t*(vals-m)))
    # Wait: f = m - (1/t) * log(S) where S = sum(exp(-t*(vals-m)))
    # df/dm = 1 - (1/t) * d(log S)/dm = 1 - (1/t) * sum(exp(-t*(vals-m)) * t) / S = 1 - S/S = 0
    # df/d(vals_k) = -(1/t) * d(log S)/d(vals_k) = -(1/t) * exp(-t*(vals_k-m)) * (-t) / S = exp(-t*(vals_k-m)) / S
    # So df/d(vals_k) = weights[k]
    
    # Now, vals_k for boundary: d_i. d_i = min of 4 linear functions of x_i, y_i.
    # The gradient of d_i w.r.t. x_i is one of {-1, 0, +1} depending on which edge is active.
    # For the soft-min, the contribution to grad x_i from boundary is:
    #   sum over the 4 boundary candidates of (weight_of_that_candidate * gradient_of_that_candidate)
    # But in our formulation, d_i is a single value in vals, not 4 separate values.
    # This is a problem. The soft-min over [d_i, pair_ij, ...] doesn't correctly handle the min inside d_i.
    
    # Let me restructure: instead of taking min of 4 edges first, include all 4 edge distances separately.
    # Actually, for the gradient to be correct, I should include all individual constraints.
    # Let me redo this.
    
    # For now, let's use a simpler gradient approximation that works well enough.
    # The key insight: for the soft-min, the gradient points in the direction that increases the minimum constraint.
    # For boundary: if the point is closest to the left edge (HALF+x_i is min), gradient pushes right (+1, 0).
    # For pairwise: if pair_ij is the minimum, gradient pushes i away from j and j away from i.
    
    # Let's compute the gradient of each individual constraint and weight by softmax.
    
    # Boundary: for point i, the 4 edge distances
    edge_vals = np.array([HALF - pts[i, 0], HALF - pts[i, 1], HALF + pts[i, 0], HALF + pts[i, 1]])
    # But we need the soft-min over ALL constraints, not just these 4.
    # This is complex. Let's just use numerical gradient with a smaller step for speed.
    # Actually, for N=50, 100 variables, numerical gradient is 100 function evals per gradient.
    # Each function eval is O(N^2) = 2500 operations. So 250k ops per gradient.
    # L-BFGS-B does ~100-500 iterations. So 25M-125M ops. Should be fine in 145s.
    
    return None  # Signal to use numerical gradient


def _grad_soft_min(pts, t):
    """Numerical gradient of soft-min objective."""
    eps = 1e-7
    g = np.zeros((N, 2))
    base = _soft_min_obj(pts, t)
    for i in range(N):
        for j in range(2):
            pts_p = pts.copy()
            pts_p[i, j] += eps
            g[i, j] = (_soft_min_obj(pts_p, t) - base) / eps
    return g


def _violation_obj(pts, r_target, t):
    """Smooth violation: how much constraints are violated for target radius r_target."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    
    boundary_viol = np.maximum(0, r_target - d)
    pair_viol = np.maximum(0, r_target - pair)
    
    all_viol = np.concatenate([boundary_viol, pair_viol])
    if np.max(all_viol) < 1e-15:
        return 0.0
    
    m = np.max(all_viol)
    s = np.sum(np.exp(t * (all_viol - m)))
    return np.log(s) / t + m - 1.0 / t


def _violation_grad(pts, r_target, t):
    """Numerical gradient of violation objective."""
    eps = 1e-7
    g = np.zeros((N, 2))
    base = _violation_obj(pts, r_target, t)
    for i in range(N):
        for j in range(2):
            pts_p = pts.copy()
            pts_p[i, j] += eps
            g[i, j] = (_violation_obj(pts_p, r_target, t) - base) / eps
    return g


def _contact_graph(pts, r):
    """Find contact pairs: pairs where dist/2 <= r * 1.05 (slightly above r)."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    threshold = r * 1.05
    contacts = []
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i, j] / 2.0 <= threshold:
                contacts.append((i, j))
    return contacts


def _gauss_seidel_contact(pts, r, n_iter=50, tol=1e-10):
    """Gauss-Seidel iteration to satisfy contact constraints.
    For each contact pair (i,j), push them apart to distance 2*r.
    For each point near a wall, push it inward to distance r from wall.
    """
    pts = pts.copy()
    contacts = _contact_graph(pts, r)
    
    for _ in range(n_iter):
        if time.time() > DEADLINE:
            break
        max_change = 0.0
        # Pairwise contacts
        for (i, j) in contacts:
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                # Random small push
                diff = np.array([1.0, 0.0])
                dist = 1.0
            # We want dist >= 2*r
            if dist < 2 * r:
                # Push apart by (2*r - dist)/2 each
                correction = (2 * r - dist) / (2 * dist) * diff
                pts[i] += correction
                pts[j] -= correction
                max_change = max(max_change, np.linalg.norm(correction))
        
        # Boundary constraints
        for i in range(N):
            # Check each wall
            # Left wall: x_i >= -HALF + r
            if pts[i, 0] < -HALF + r:
                pts[i, 0] = -HALF + r
                max_change = max(max_change, r - pts[i, 0])
            # Right wall: x_i <= HALF - r
            if pts[i, 0] > HALF - r:
                pts[i, 0] = HALF - r
                max_change = max(max_change, pts[i, 0] - (HALF - r))
            # Bottom wall: y_i >= -HALF + r
            if pts[i, 1] < -HALF + r:
                pts[i, 1] = -HALF + r
                max_change = max(max_change, r - pts[i, 1])
            # Top wall: y_i <= HALF - r
            if pts[i, 1] > HALF - r:
                pts[i, 1] = HALF - r
                max_change = max(max_change, pts[i, 1] - (HALF - r))
        
        if max_change < tol:
            break
    
    return pts


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice."""
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
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF + 1e-9, HALF - 1e-9)


def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _simulated_annealing(pts, r_target, t_start=10.0, t_end=0.1, n_iter=5000, deadline=DEADLINE):
    """Simulated annealing to find a feasible configuration for radius r_target."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _compute_r(pts)
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        T = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Random perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, 0.01, size=2)
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -HALF + 1e-9, HALF - 1e-9)
        
        new_r = _compute_r(new_pts)
        delta_r = new_r - _compute_r(pts)
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / (T * 0.01)):
            pts = new_pts
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
    
    return best_pts, best_r


def _lbfgs_optimize(pts0, t, maxiter, deadline):
    """L-BFGS-B to maximize soft-min score."""
    x0 = pts0.ravel()
    
    def neg_obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_soft_min_obj(p, t)
    
    def neg_grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _grad_soft_min(p, t).ravel()
    
    try:
        res = minimize(neg_obj, x0, method='L-BFGS-B',
                       jac=neg_grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
        return pts, float(-res.fun)
    except Exception:
        return pts0, _soft_min_obj(pts0, t)


def _lbfgs_violation(pts0, r_target, t, maxiter, deadline):
    """L-BFGS-B to minimize violation of radius r_target."""
    x0 = pts0.ravel()
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation_obj(p, r_target, t)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation_grad(p, r_target, t).ravel()
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       jac=grad,
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
        return pts, float(res.fun)
    except Exception:
        return pts0, _violation_obj(pts0, r_target, t)


def _refine_with_contact_graph(pts, r, n_gs_iters=100):
    """Refine configuration using Gauss-Seidel on contact graph."""
    pts = _gauss_seidel_contact(pts, r, n_iter=n_gs_iters)
    return pts


def construct():
    start = time.time()
    deadline = start + 145.0
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def try_update(pts):
        nonlocal best_pts, best_r
        r = _compute_r(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        return r
    
    # Phase 1: Multi-start with SA + L-BFGS
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
    ]
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        # SA first
        sa_pts, sa_r = _simulated_annealing(init, 0.07, t_start=5.0, t_end=0.1, n_iter=2000, deadline=deadline)
        try_update(sa_pts)
        
        # Then L-BFGS polish
        if time.time() < deadline - 20.0:
            pts, score = _lbfgs_optimize(sa_pts, t=30.0, maxiter=100, deadline=deadline)
            try_update(pts)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search on radius with contact-graph Gauss-Seidel
    r_low = best_r
    r_high = min(r_low + 0.01, 0.1)
    
    for iteration in range(20):
        if time.time() > deadline - 20.0:
            break
        r_mid = (r_low + r_high) / 2.0
        
        # Use contact-graph Gauss-Seidel to try to achieve r_mid
        pts_gs = _refine_with_contact_graph(best_pts, r_mid, n_gs_iters=200)
        r_gs = _compute_r(pts_gs)
        
        if r_gs >= r_mid - 1e-8:
            r_low = r_mid
            if r_gs > best_r:
                best_r = r_gs
                best_pts = pts_gs.copy()
        else:
            # Try L-BFGS violation minimization
            if time.time() < deadline - 15.0:
                pts_v, viol = _lbfgs_violation(best_pts, r_mid, t=50.0, maxiter=150, deadline=deadline)
                r_v = _compute_r(pts_v)
                if r_v >= r_mid - 1e-8:
                    r_low = r_mid
                    if r_v > best_r:
                        best_r = r_v
                        best_pts = pts_v.copy()
                else:
                    r_high = r_mid
            else:
                r_high = r_mid
    
    # Phase 3: Final polish with high-precision L-BFGS
    if time.time() < deadline - 10.0:
        for t in [100.0, 200.0]:
            if time.time() > deadline - 5.0:
                break
            pts, score = _lbfgs_optimize(best_pts, t=t, maxiter=200, deadline=deadline)
            try_update(pts)
    
    # Phase 4: Contact-graph refinement at the final radius
    if time.time() < deadline - 5.0:
        pts_final = _refine_with_contact_graph(best_pts, best_r, n_gs_iters=500)
        try_update(pts_final)
    
    return best_pts
