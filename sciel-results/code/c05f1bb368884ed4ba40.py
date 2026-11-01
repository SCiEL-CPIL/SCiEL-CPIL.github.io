"""Packing 50 equal circles in unit square using soft-min optimization + SA restarts."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 145.0  # seconds before 150s kill


def _compute_radius(c):
    """Compute the max common radius for centers c."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _soft_min_obj_and_grad(c, tau):
    """
    Soft-min objective: smooth approximation of -min(all clearances).
    
    We want to maximize r = min(min_pair_dist/2, min_wall_clearance).
    Equivalently, minimize -r.
    
    Let z_k be all clearance values (pairwise half-distances and wall clearances).
    r = min_k z_k
    -r ≈ tau * log(sum exp(-z_k/tau))  (negative, so we minimize this)
    
    Gradient of tau * log(sum exp(-z_k/tau)) w.r.t. x:
    = -sum_k [exp(-z_k/tau) / sum_j exp(-z_j/tau)] * dz_k/dx
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]  # (n,n,2)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n,n)
    np.fill_diagonal(dist, np.inf)
    
    # Wall clearances
    wall_c = np.column_stack([
        c[:, 0] + HALF,  # right wall
        HALF - c[:, 0],  # left wall
        c[:, 1] + HALF,  # top wall
        HALF - c[:, 1]   # bottom wall
    ])  # (n, 4)
    
    # Collect all clearances
    # Pair clearances: dist/2 for i<j
    pair_idx = np.triu_indices(n, k=1)
    pair_dists = dist[pair_idx] / 2.0  # (n_pairs,)
    
    # All z values
    z = np.concatenate([pair_dists, wall_c.ravel()])  # (n_pairs + 4n,)
    n_z = len(z)
    
    # For numerical stability
    z_min = np.min(z)
    exp_z = np.exp(-(z - z_min) / tau)
    sum_exp = np.sum(exp_z)
    log_sum = np.log(sum_exp)
    obj = -tau * (log_sum - z_min / tau)  # = tau * log(sum exp(-(z-z_min)/tau)) - z_min
    # Actually: softmin ≈ z_min + tau * log(sum exp(-(z - z_min)/tau))
    # We want to maximize this, so minimize -softmin
    softmin_val = z_min + tau * log_sum
    obj = -softmin_val
    
    # Gradient
    # p_k = exp(-(z_k - z_min)/tau) / sum_exp
    p = exp_z / sum_exp  # (n_z,)
    
    grad = np.zeros_like(c)
    
    # Pair contributions: z_k = dist_{ij}/2 = 0.5 * ||c_i - c_j||
    # dz_k/dc_i = 0.5 * (c_i - c_j) / dist_{ij}
    # dz_k/dc_j = 0.5 * (c_j - c_i) / dist_{ij}
    for idx_k in range(len(pair_idx[0])):
        i, j = pair_idx[0][idx_k], pair_idx[1][idx_k]
        d_ij = dist[i, j]
        if d_ij < 1e-12:
            continue
        d_z_dc_i = 0.5 * (c[i] - c[j]) / d_ij
        d_z_dc_j = 0.5 * (c[j] - c[i]) / d_ij
        grad[i] -= p[idx_k] * d_z_dc_i
        grad[j] -= p[idx_k] * d_z_dc_j
    
    # Wall contributions
    # z for wall 0 (right): c_x + HALF, dz/dc_x = 1
    # z for wall 1 (left): HALF - c_x, dz/dc_x = -1
    # z for wall 2 (top): c_y + HALF, dz/dc_y = 1
    # z for wall 3 (bottom): HALF - c_y, dz/dc_y = -1
    wall_p = p[len(pair_dists):].reshape(n, 4)
    # d_z/dc for each wall:
    # wall 0: [1, 0], wall 1: [-1, 0], wall 2: [0, 1], wall 3: [0, -1]
    wall_grad = np.column_stack([
        wall_p[:, 0] - wall_p[:, 1],  # x component
        wall_p[:, 2] - wall_p[:, 3]   # y component
    ])
    grad -= wall_grad
    
    return obj, grad


def _soft_min_polish(c, tau_start=0.05, tau_end=0.001, n_phases=10, deadline=None):
    """Polish a configuration using soft-min with decreasing tau."""
    c = np.asarray(c, dtype=np.float64).copy()
    
    for phase in range(n_phases):
        if deadline is not None and time.time() > deadline:
            break
        tau = tau_start * (tau_end / tau_start) ** (phase / max(n_phases - 1, 1))
        
        def obj_grad(x):
            o, g = _soft_min_obj_and_grad(x, tau)
            return o, g
        
        res = minimize(obj_grad, c, jac=True, method='L-BFGS-B',
                       bounds=[(-HALF, HALF)] * (2 * N),
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.success or res.fun < obj_grad(c)[0]:
            c = res.x
    
    # Final clip to valid region (with small epsilon for safety)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    return c


def _simulated_annealing(c, r_target, n_steps=5000, T_start=0.01, T_end=0.0001, deadline=None):
    """
    SA that tries to find a configuration where all circles of radius r_target fit.
    Objective: minimize max violation (negative clearance).
    """
    c = np.asarray(c, dtype=np.float64).copy()
    n = len(c)
    
    def violation(x):
        """Max violation: positive means infeasible."""
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.max(2 * r_target - dist)  # positive if any pair too close
        wall_viol = np.max(2 * r_target - 2 * np.minimum(
            np.minimum(x[:, 0] + HALF, HALF - x[:, 0]),
            np.minimum(x[:, 1] + HALF, HALF - x[:, 1])
        )) / 2.0  # Hmm, let me redo wall
        # Wall: circle i is at x_i, radius r. It fits if |x_i| <= HALF - r.
        # Violation: r - (HALF - |x_i|) if positive.
        wall_clear = HALF - np.abs(x)  # (n, 2)
        wall_viol = np.max(r_target - wall_clear)
        return max(pair_viol, wall_viol)
    
    def energy(x):
        return violation(x)
    
    e_curr = energy(c)
    best_c = c.copy()
    best_e = e_curr
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (step / max(n_steps - 1, 1))
        
        # Propose: move a random circle
        i = np.random.randint(n)
        delta = np.random.randn(2) * 0.01 * np.sqrt(T / T_start + 0.01)
        new_c = c.copy()
        new_c[i] += delta
        new_c = np.clip(new_c, -HALF + r_target - 0.02, HALF - r_target + 0.02)
        
        e_new = energy(new_c)
        de = e_new - e_curr
        
        if de < 0 or np.random.rand() < np.exp(-de / max(T, 1e-12)):
            c = new_c
            e_curr = e_new
            if e_curr < best_e:
                best_c = c.copy()
                best_e = e_curr
            if best_e <= 1e-9:
                break
    
    return best_c, best_e


def _hex_init(r):
    """Hexagonal lattice initialization."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        offset = 0.5 * dx if (row % 2) else 0.0
        x = -HALF + r + offset
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)), dtype=np.float64)
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        # Fill with grid
        m = int(np.ceil(np.sqrt(N - len(pts))))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_init(r):
    """Square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks], dtype=np.float64)
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _random_init(r):
    """Random initialization with rejection sampling for non-overlap."""
    pts = []
    attempts = 0
    while len(pts) < N and attempts < 100000:
        attempts += 1
        p = np.random.uniform(-HALF + r, HALF - r, size=2)
        if len(pts) == 0:
            pts.append(p)
        else:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 2 * r - 1e-9:
                    ok = False
                    break
            if ok:
                pts.append(p)
    if len(pts) < N:
        # Fall back to grid
        return _grid_init(r)
    return np.array(pts, dtype=np.float64)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_c = _hex_init(0.071)
    best_r = _compute_radius(best_c)
    
    # Phase 1: Try to find a good initial configuration via SA at various radii
    r_trials = [0.071, 0.0705, 0.070, 0.0695, 0.069, 0.068, 0.067, 0.066]
    
    for r_try in r_trials:
        if time.time() > deadline - 30:
            break
        
        # Try a few different initializations
        inits = [_hex_init(r_try), _grid_init(r_try)]
        
        for init_c in inits:
            if time.time() > deadline - 30:
                break
            
            # SA to try to achieve radius r_try
            sa_c, sa_e = _simulated_annealing(
                init_c, r_try, n_steps=3000,
                T_start=0.005, T_end=0.00005,
                deadline=deadline - 25
            )
            
            # Polish the result
            polished = _soft_min_polish(sa_c, tau_start=0.03, tau_end=0.0005,
                                         n_phases=8, deadline=deadline - 10)
            r_achieved = _compute_radius(polished)
            
            if r_achieved > best_r:
                best_r = r_achieved
                best_c = polished.copy()
    
    # Phase 2: Direct soft-min optimization from best configuration
    if time.time() < deadline - 15:
        # Progressive tau reduction for maximum precision
        polished = _soft_min_polish(
            best_c, tau_start=0.02, tau_end=0.0001,
            n_phases=12, deadline=deadline - 5
        )
        r_achieved = _compute_radius(polished)
        if r_achieved > best_r:
            best_r = r_achieved
            best_c = polished.copy()
    
    # Phase 3: Local refinement with very small tau
    if time.time() < deadline - 5:
        final_polish = _soft_min_polish(
            best_c, tau_start=0.005, tau_end=0.00002,
            n_phases=8, deadline=deadline - 2
        )
        r_final = _compute_radius(final_polish)
        if r_final > best_r:
            best_r = r_final
            best_c = final_polish.copy()
    
    return best_c
