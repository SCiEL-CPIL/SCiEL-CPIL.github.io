"""Circle packing in unit square: 26 circles, maximize sum of radii.
Uses L-BFGS-B with smooth penalty for overlaps, multiple diverse starts,
and a final polish pass with tight tolerances.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 130.0  # safety margin before 150s kill


def _smooth_penalty(x, y, r, softness=0.01):
    """Smooth penalty for overlaps and boundary violations using soft-min."""
    # Pairwise overlap: max(0, r_i + r_j - d_ij)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    r_sum = r[:, None] + r[None, :]
    overlap = r_sum - d
    # Soft max of positive overlaps
    pos_overlap = np.maximum(overlap, 0)
    log_sum_exp = np.log(np.sum(np.exp(pos_overlap / softness), axis=1) * softness)
    pairwise_penalty = np.sum(log_sum_exp)
    
    # Boundary violations
    bxl = np.maximum(0, r - x)
    bxr = np.maximum(0, r - (1 - x))
    byl = np.maximum(0, r - y)
    byr = np.maximum(0, r - (1 - y))
    boundary_penalty = np.sum(bxl) + np.sum(bxr) + np.sum(byl) + np.sum(byr)
    
    return pairwise_penalty + boundary_penalty


def _objective(z, softness=0.01):
    """Objective: minimize -sum(r) + penalty * penalty_terms."""
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:]
    
    # Objective: maximize sum of radii, so minimize -sum(r)
    obj = -np.sum(r)
    
    # Add smooth penalty for constraints
    penalty = _smooth_penalty(x, y, r, softness)
    lam = 10.0
    return obj + lam * penalty


def _gradient(z, softness=0.01):
    """Numerical gradient for the objective."""
    eps = 1e-7
    grad = np.zeros_like(z)
    for i in range(len(z)):
        z_plus = z.copy()
        z_plus[i] += eps
        z_minus = z.copy()
        z_minus[i] -= eps
        grad[i] = (_objective(z_plus, softness) - _objective(z_minus, softness)) / (2 * eps)
    return grad


def _validate(x, y, r, tol=1e-9):
    """Check validity: boundary and no overlaps."""
    if np.any(r < 0):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or \
       np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    r_sum = r[:, None] + r[None, :]
    if np.any(d < r_sum - tol):
        return False
    return True


def _repair(x, y, r, deadline):
    """Shrink radii to eliminate overlaps and boundary violations."""
    for _ in range(200):
        if time.time() > deadline:
            break
        # Clamp to boundary
        r_max_b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, r_max_b)
        r = np.maximum(r, 1e-6)
        
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        overlaps = d < r_sum - 1e-9
        np.fill_diagonal(overlaps, False)
        
        if not np.any(overlaps):
            break
        
        # Shrink circles involved in overlaps
        overlap_count = np.sum(overlaps, axis=1)
        for i in range(N):
            if overlap_count[i] > 0:
                r[i] *= (1 - 0.05 * overlap_count[i])
        r = np.maximum(r, 1e-6)
    
    return x, y, r


def _polish(x, y, r, deadline):
    """Final high-precision polish: adjust positions and radii with tight tolerances."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    best_sum = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Iteratively maximize each radius given others
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Compute max radius for circle i given others
            r_max = np.minimum(np.minimum(x[i], 1 - x[i]), np.minimum(y[i], 1 - y[i]))
            for j in range(N):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx**2 + dy**2)
                r_max = min(r_max, d - r[j])
            r_max = max(r_max, 1e-6)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
        
        s = np.sum(r)
        if s > best_sum + 1e-10:
            best_sum = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        if not improved:
            break
    
    return best_x, best_y, best_r


def _hex_init(n, scale=0.18):
    """Place n circles on a hexagonal lattice."""
    pts = []
    cols = int(np.ceil(1.0 / scale)) + 2
    rows = int(np.ceil(1.0 / (scale * np.sqrt(3) / 2))) + 2
    for j in range(rows):
        y = j * scale * np.sqrt(3) / 2
        offset = 0 if j % 2 == 0 else scale / 2
        for i in range(cols):
            x = i * scale + offset
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < n:
        extra = np.random.rand(n - len(pts), 2) * 0.8 + 0.1
        pts = np.vstack([pts, extra])
    return pts[:n]


def _lbfgs_solve(x, y, r, deadline, softness=0.01):
    """Run L-BFGS-B optimization starting from (x, y, r)."""
    z = np.concatenate([x, y, r])
    
    # Bounds: x, y in [0.01, 0.99], r in [0.01, 0.4]
    bounds = [(0.01, 0.99)] * N + [(0.01, 0.99)] * N + [(0.01, 0.4)] * N
    
    try:
        result = minimize(
            _objective, z, args=(softness,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                     'maxls': 50, 'disp': False}
        )
        z_opt = result.x
    except Exception:
        z_opt = z
    
    x_opt = z_opt[:N]
    y_opt = z_opt[N:2*N]
    r_opt = z_opt[2*N:]
    
    # Repair any remaining violations
    x_opt, y_opt, r_opt = _repair(x_opt, y_opt, r_opt, deadline)
    
    return x_opt, y_opt, r_opt


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    
    best_score = -np.inf
    best_config = None
    
    def _try_config(x, y, r):
        nonlocal best_score, best_config
        x, y, r = _repair(x, y, r, deadline)
        if _validate(x, y, r):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_config = (x.copy(), y.copy(), r.copy())
    
    # Multiple diverse initializations
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(2000 + restart * 191)
        
        if restart % 4 == 0:
            # Hex lattice with varying scale
            scale = 0.15 + 0.02 * (restart // 4)
            pts = _hex_init(N, scale)
            x = pts[:, 0].copy()
            y = pts[:, 1].copy()
            r = np.full(N, 0.04)
        elif restart % 4 == 1:
            # Jittered grid
            side = int(np.ceil(np.sqrt(N)))
            xs, ys, rs = [], [], []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        px = (i + 0.5) / side + np.random.uniform(-0.03, 0.03)
                        py = (j + 0.5) / side + np.random.uniform(-0.03, 0.03)
                        px = np.clip(px, 0.05, 0.95)
                        py = np.clip(py, 0.05, 0.95)
                        xs.append(px)
                        ys.append(py)
                        rs.append(0.06)
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        elif restart % 4 == 2:
            # Random
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.random.uniform(0.03, 0.08, N)
        else:
            # Clustered: few large, many small
            xs, ys, rs = [], [], []
            for _ in range(6):
                xs.append(np.random.uniform(0.2, 0.8))
                ys.append(np.random.uniform(0.2, 0.8))
                rs.append(np.random.uniform(0.06, 0.10))
            for _ in range(N - 6):
                xs.append(np.random.uniform(0.1, 0.9))
                ys.append(np.random.uniform(0.1, 0.9))
                rs.append(np.random.uniform(0.02, 0.05))
            x = np.array(xs)
            y = np.array(ys)
            r = np.array(rs)
        
        # L-BFGS-B optimization
        x_opt, y_opt, r_opt = _lbfgs_solve(x, y, r, deadline)
        _try_config(x_opt, y_opt, r_opt)
        
        if time.time() > deadline:
            break
        
        # Polish pass
        x_pol, y_pol, r_pol = _polish(x_opt, y_opt, r_opt, deadline)
        _try_config(x_pol, y_pol, r_pol)
    
    # Fallback: simple grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_config = (np.array([c[0] for c in cells[:N]]),
                       np.array([c[1] for c in cells[:N]]),
                       np.array([c[2] for c in cells[:N]]))
    
    x, y, r = best_config
    return np.column_stack((x, y, r))
