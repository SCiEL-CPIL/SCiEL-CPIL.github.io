"""Ring-parameterized soft-min optimization for 300-circle packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hard_radius(pts):
    """Exact common radius, -inf if invalid."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(pts)):
        return -np.inf
    d0 = np.linalg.norm(pts, axis=1)
    rb = 1.0 - np.min(d0)
    if rb <= 0.0:
        return -np.inf
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    rp = 0.5 * np.sqrt(np.min(d2))
    return min(rb, rp)


def _polar_to_cart(r_arr, theta_arr):
    """Convert polar to Cartesian."""
    return np.column_stack([r_arr * np.cos(theta_arr), r_arr * np.sin(theta_arr)])


def _cart_to_polar(pts):
    """Convert Cartesian to polar (r, theta)."""
    r = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    return r, theta


def _soft_min_obj_and_grad(x, k=50.0):
    """
    Objective: minimize -soft_min(margins) where margins = [boundary_margins, pair_margins/2].
    x = [r_1,...,r_N, theta_1,...,theta_N] in polar coords.
    This directly maximizes the minimum margin via smooth surrogate.
    """
    r_arr = x[:N]
    theta_arr = x[N:]
    pts = _polar_to_cart(r_arr, theta_arr)
    
    # Boundary margins: 1 - |c_i| = 1 - r_i
    b_margins = 1.0 - r_arr
    
    # Pair margins: |c_i - c_j| / 2
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(np.maximum(d2, 1e-16))
    p_margins = 0.5 * d
    
    # Combine all margins into one vector for soft-min
    all_margins = np.concatenate([b_margins, p_margins[np.triu_indices(N, k=1)]])
    
    # Soft min: -log(sum(exp(-k*m)))/k
    m_min = np.min(all_margins)
    soft_min_val = m_min - np.log(np.sum(np.exp(-k * (all_margins - m_min)))) / k
    
    # Objective to minimize: -soft_min_val
    f = -soft_min_val
    
    # Gradient computation
    # d(soft_min)/d(margins_j) = exp(-k*m_j) / sum(exp(-k*m_l))
    exp_terms = np.exp(-k * (all_margins - m_min))
    weights = exp_terms / np.sum(exp_terms)
    
    # d(soft_min)/d(margins_j) = weights_j
    # So d(-soft_min)/d(margins_j) = -weights_j
    
    # Chain rule: margins depend on x through r and theta
    g_r = np.zeros(N)
    g_theta = np.zeros(N)
    
    # Boundary margins: b_margins[i] = 1 - r_i
    # d(b_margins[i])/dr_i = -1
    # Contribution to g_r[i] = -weights[i] * (-1) = weights[i]
    g_r += weights[:N]
    
    # Pair margins: p_margins[i,j] = 0.5 * |c_i - c_j|
    # For pair (i,j), weight is weights[N + idx] where idx is the flat index in upper triangle
    # d(p_margins[i,j])/dr_i and d(p_margins[i,j])/dtheta_i need chain rule
    
    # c_i = (r_i cos(theta_i), r_i sin(theta_i))
    # d c_i / dr_i = (cos(theta_i), sin(theta_i))
    # d c_i / dtheta_i = (-r_i sin(theta_i), r_i cos(theta_i))
    
    # d(p_margins[i,j])/d c_i = 0.5 * (c_i - c_j) / |c_i - c_j| = 0.5 * (c_i - c_j) / (2*p_margins[i,j])
    # Actually: d(0.5*|c_i-c_j|)/d c_i = 0.5 * (c_i-c_j)/|c_i-c_j|
    
    # Build pair weights matrix
    pair_weights = np.zeros((N, N))
    ii, jj = np.triu_indices(N, k=1)
    pair_weights[ii, jj] = weights[N + np.arange(len(ii))]
    pair_weights[jj, ii] = pair_weights[ii, jj]
    
    # For each point i, accumulate gradient from all pairs involving i
    # d(p_margins[i,j])/d c_i = 0.5 * (c_i - c_j) / d_ij
    # d(p_margins[i,j])/d r_i = 0.5 * (c_i - c_j) / d_ij · (cos(theta_i), sin(theta_i))
    # d(p_margins[i,j])/d theta_i = 0.5 * (c_i - c_j) / d_ij · (-r_i sin(theta_i), r_i cos(theta_i))
    
    cos_t = np.cos(theta_arr)
    sin_t = np.sin(theta_arr)
    
    # Vectorized computation
    # For each i: sum over j != i of pair_weights[i,j] * 0.5 * (c_i - c_j) / d_ij
    # = sum_j pair_weights[i,j] * 0.5 * (c_i - c_j) / d_ij
    
    # Let's compute contribution to g_r and g_theta
    # g_r[i] += -sum_j pair_weights[i,j] * d(p_margins[i,j])/dr_i
    #         = -sum_j pair_weights[i,j] * 0.5 * (c_i - c_j)·(cos_t[i], sin_t[i]) / d_ij
    
    # Compute all pair differences and distances
    diff_full = pts[:, None, :] - pts[None, :, :]
    d_full = np.sqrt(np.einsum('ijk,ijk->ij', diff_full, diff_full))
    np.fill_diagonal(d_full, np.inf)
    
    # Normalize: unit_ij = (c_i - c_j) / d_ij
    with np.errstate(divide='ignore', invalid='ignore'):
        unit = diff_full / d_full[:, :, None]
    unit = np.where(d_full[:, :, None] < 1e-12, 0.0, unit)
    
    # For each i, sum over j: pair_weights[i,j] * 0.5 * unit[i,j]
    # This gives the vector sum that contributes to d(p_margins)/d c_i
    # weighted_sum_i = sum_j pair_weights[i,j] * 0.5 * unit[i,j]
    weighted_sum = np.einsum('ij,ijk->ik', pair_weights, unit * 0.5)
    
    # d(p_margins[i,j])/dr_i = weighted_sum[i] · (cos_t[i], sin_t[i])
    # d(p_margins[i,j])/dtheta_i = weighted_sum[i] · (-r_i sin_t[i], r_i cos_t[i])
    
    g_r -= weighted_sum[:, 0] * cos_t + weighted_sum[:, 1] * sin_t
    g_theta -= -weighted_sum[:, 0] * r_arr * sin_t + weighted_sum[:, 1] * r_arr * cos_t
    
    g = np.concatenate([g_r, g_theta])
    return f, g


def _make_hex_init(jitter=0.0, seed=0):
    """Hexagonal lattice inside unit disk, with optional jitter."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    pts_list = []
    k = int(1.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - s * 0.5:
                pts_list.append([x, y])
    pts = np.array(pts_list)
    if len(pts) < N:
        s *= 0.95
        pts_list = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - s * 0.5:
                    pts_list.append([x, y])
        pts = np.array(pts_list)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(rad, kind='stable')
    pts = pts[order[:N]]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
        d0 = np.linalg.norm(pts, axis=1)
        over = d0 > 0.98
        if np.any(over):
            pts[over] *= (0.98 / d0[over])[:, None]
    return pts


def _make_ring_init(seed=0):
    """Concentric rings initialization with better distribution."""
    rng = np.random.default_rng(seed)
    best_pts, best_r = None, -np.inf
    for n_rings in range(7, 14):
        # Distribute points by area: ring i has proportional area
        radii = np.linspace(0.05, 0.95, n_rings)
        # Number of points per ring proportional to circumference
        circs = 2 * np.pi * radii
        counts = np.round(circs / circs.sum() * N).astype(int)
        # Adjust to sum to N
        diff = N - counts.sum()
        if diff > 0:
            counts[np.argsort(-counts)][:diff] += 1
        elif diff < 0:
            counts[np.argsort(counts)][:-diff] += 1
        counts = np.maximum(counts, 1)
        # Renormalize
        counts = (counts / counts.sum() * N).astype(int)
        diff = N - counts.sum()
        counts[np.argmax(counts)] += diff
        
        pts = []
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            rr = radii[i]
            ang = np.linspace(0, 2 * np.pi, c, endpoint=False) + i * 0.4
            for a in ang:
                pts.append([rr * np.cos(a), rr * np.sin(a)])
        pts = np.array(pts[:N])
        if len(pts) < N:
            continue
        r = _hard_radius(pts)
        if r > best_r:
            best_r, best_pts = r, pts.copy()
    return best_pts if best_pts is not None else _make_hex_init(0.0, seed)


def _optimize_polar(pts0, deadline, seed=0):
    """Optimize in polar coordinates using soft-min objective."""
    rng = np.random.default_rng(seed)
    r_arr, theta_arr = _cart_to_polar(pts0)
    x0 = np.concatenate([r_arr, theta_arr])
    
    best_pts = np.array(pts0, dtype=float, copy=True)
    best_r = _hard_radius(best_pts)
    if best_r <= 0:
        return None
    
    # Adaptive k: start with moderate k, increase for sharper min
    k_schedule = [20.0, 30.0, 50.0, 80.0, 120.0]
    n_anneal = len(k_schedule)
    
    for step in range(n_anneal):
        if time.time() > deadline - 3.0:
            break
        k = k_schedule[step]
        # Use L-BFGS-B with bounds on r
        bounds = [(1e-6, 0.999)] * N + [(-np.pi, np.pi)] * N
        res = minimize(_soft_min_obj_and_grad, x0, args=(k,), method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 30})
        if res.x is not None and np.all(np.isfinite(res.x)):
            r_new = res.x[:N]
            theta_new = res.x[N:]
            # Ensure r is in valid range
            r_new = np.clip(r_new, 1e-6, 0.999)
            pts_new = _polar_to_cart(r_new, theta_new)
            r = _hard_radius(pts_new)
            if r > best_r:
                best_r = r
                best_pts = pts_new.copy()
                x0 = np.concatenate([r_new, theta_new])
        # Small perturbation to escape local minima
        if step < n_anneal - 1 and step % 2 == 1:
            x0 = x0 + rng.normal(0, 0.001, size=x0.shape)
            x0[:N] = np.clip(x0[:N], 1e-6, 0.999)
    
    return best_pts


def _polish(pts, deadline):
    """Multi-point simultaneous polish using active set of critical pairs/points."""
    pts = np.array(pts, dtype=float, copy=True)
    cur = _hard_radius(pts)
    if cur <= 0:
        return None
    best_pts = pts.copy()
    best_r = cur
    step = 1e-4
    no_improve = 0
    max_no = 50
    while time.time() < deadline - 2.0 and no_improve < max_no:
        d0 = np.linalg.norm(pts, axis=1)
        rb = 1.0 - np.min(d0)
        ib = int(np.argmin(d0))
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        ii, jj = np.unravel_index(np.argmin(d2), d2.shape)
        d = np.sqrt(d2[ii, jj])
        rp = 0.5 * d
        
        # Move both the critical pair and the boundary point
        if rp < rb:
            v = pts[jj] - pts[ii]
            nrm = np.linalg.norm(v)
            if nrm < 1e-12:
                break
            v = v / nrm
            pts[ii] -= 0.5 * step * v
            pts[jj] += 0.5 * step * v
        else:
            p = pts[ib]
            nrm = np.linalg.norm(p)
            if nrm < 1e-12:
                break
            pts[ib] -= step * (p / nrm)
        
        # Clamp
        d0 = np.linalg.norm(pts, axis=1)
        over = d0 > 0.9999
        if np.any(over):
            pts[over] *= (0.9999 / d0[over])[:, None]
        new_r = _hard_radius(pts)
        if new_r > cur:
            cur = new_r
            step = max(1e-7, step * 0.93)
            no_improve = 0
            if cur > best_r:
                best_r = cur
                best_pts = pts.copy()
        else:
            # Revert
            if rp < rb:
                v = pts[jj] - pts[ii]
                nrm = np.linalg.norm(v)
                if nrm < 1e-12:
                    break
                v = v / nrm
                pts[ii] += 0.5 * step * v
                pts[jj] -= 0.5 * step * v
            else:
                p = pts[ib]
                nrm = np.linalg.norm(p)
                if nrm < 1e-12:
                    break
                pts[ib] += step * (p / nrm)
            step *= 0.6
            no_improve += 1
            if step < 1e-9:
                break
    return best_pts


def construct():
    start = time.time()
    deadline = start + 145.0
    
    # Generate initializations
    inits = []
    inits.append(_make_hex_init(0.0, seed=0))
    for s in range(1, 4):
        inits.append(_make_hex_init(0.002, seed=s))
    inits.append(_make_ring_init(seed=10))
    inits.append(_make_ring_init(seed=20))
    
    # Score and sort
    scored = []
    for i, pts in enumerate(inits):
        if pts is None or len(pts) != N:
            continue
        r = _hard_radius(pts)
        if r > 0:
            scored.append((r, i, pts))
    scored.sort(key=lambda x: -x[0])
    
    if not scored:
        return _make_hex_init(0.0, seed=0)
    
    best_pts = scored[0][2].copy()
    best_r = scored[0][0]
    
    # Allocate time: 60% for optimization, 30% for polish, 10% buffer
    opt_time = 0.6 * (deadline - start - 5.0)
    n_opt = min(3, len(scored))
    per_opt = opt_time / n_opt
    
    for idx in range(n_opt):
        if time.time() > deadline - 40.0:
            break
        r0, i0, pts0 = scored[idx]
        t_end = time.time() + per_opt
        res = _optimize_polar(pts0, t_end, seed=100 + idx)
        if res is not None:
            r = _hard_radius(res)
            if r > best_r:
                best_r = r
                best_pts = res.copy()
    
    # Polish
    if time.time() < deadline - 5.0:
        polished = _polish(best_pts, deadline)
        if polished is not None:
            r = _hard_radius(polished)
            if r > best_r:
                best_r = r
                best_pts = polished
    
    if _hard_radius(best_pts) <= 0.0:
        return scored[0][2]
    return best_pts
# EVOLVE-BLOCK-END
