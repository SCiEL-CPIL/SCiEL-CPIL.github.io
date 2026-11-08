"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Structural change: contact-graph optimization with differentiable min-pooling
over pairwise gaps. Maximizes the minimum gap (d_ij - r_i - r_j) to drive
toward a fully-stressed configuration, then grows radii under the constraint
that all gaps remain non-negative. Uses S-softmin for smooth differentiability.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 115.0  # Safety margin
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5

def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[mask] < s[mask] - tol):
        return False
    return True

def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf

def _clamp(c):
    """Clamp positions/radii to a feasible box (keeps circles inside square)."""
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])

def _s_softmin(v, T):
    """Differentiable soft-min: -T * log(sum(exp(-v_i/T))). Returns (value, weights)."""
    a = -v / T
    a_max = np.max(a)
    a_shifted = a - a_max
    exp_a = np.exp(a_shifted)
    log_sum = a_max + np.log(np.sum(exp_a))
    value = -T * log_sum
    # Weights: softmax of -v/T
    weights = exp_a / np.sum(exp_a)
    return value, weights

def _gap_objective(params, T):
    """
    Contact-graph objective: maximize sum of radii subject to min gap >= 0.
    Uses S-softmin over all pairwise gaps and wall gaps.
    
    Returns (obj, grad) where obj = -sum(r) + lambda * softmin_penalty.
    """
    c = params.reshape(N, 3)
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    
    # Compute all gaps
    # Wall gaps: x - r, 1 - x - r, y - r, 1 - y - r
    wall_gaps = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise gaps: d_ij - r_i - r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    gap_matrix = d - s
    np.fill_diagonal(gap_matrix, np.inf)
    
    # Upper triangle pairwise gaps
    tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_gaps = gap_matrix[tri_mask]
    
    # All gaps
    all_gaps = np.concatenate([wall_gaps, pair_gaps])
    
    # S-softmin of all gaps
    min_gap, weights = _s_softmin(all_gaps, T)
    
    # Objective: -sum(r) + lambda * max(0, -min_gap)
    # But we want to maximize min_gap, so: -sum(r) - lambda * min_gap if min_gap < 0
    # Actually: maximize sum(r) subject to min_gap >= 0
    # Lagrangian: sum(r) - lambda * max(0, -min_gap)
    # We minimize: -sum(r) + lambda * max(0, -min_gap)
    
    lambda_weight = 100.0
    penalty = lambda_weight * np.maximum(0.0, -min_gap)
    obj = -np.sum(r) + penalty
    
    # Gradient
    grad = np.zeros_like(params)
    
    # d(-sum(r))/dr_i = -1
    grad[:, 2] = -1.0
    
    # Gradient of penalty
    if min_gap < 0:
        # d penalty / d params = lambda * d(-min_gap)/d params
        # d(-min_gap)/d params = -sum_j w_j * d(gap_j)/d params
        # where w_j are the softmin weights
        
        # Wall gap gradients
        # gap = x - r: d/dx = 1, d/dr = -1
        # gap = 1 - x - r: d/dx = -1, d/dr = -1
        # gap = y - r: d/dy = 1, d/dr = -1
        # gap = 1 - y - r: d/dy = -1, d/dr = -1
        
        w_wall = weights[:4*N]
        w_left = w_wall[0:N]
        w_right = w_wall[N:2*N]
        w_bottom = w_wall[2*N:3*N]
        w_top = w_wall[3*N:4*N]
        
        grad[:, 0] += lambda_weight * (-np.sum(w_left) + np.sum(w_right))
        grad[:, 1] += lambda_weight * (-np.sum(w_bottom) + np.sum(w_top))
        grad[:, 2] += lambda_weight * (np.sum(w_left) + np.sum(w_right) + np.sum(w_bottom) + np.sum(w_top))
        
        # Pairwise gap gradients
        w_pair = weights[4*N:]
        
        # For pair (i,j): gap = d_ij - r_i - r_j
        # d(gap)/dx_i = (x_i - x_j)/d_ij
        # d(gap)/dx_j = (x_j - x_i)/d_ij = -(x_i - x_j)/d_ij
        # d(gap)/dr_i = -1
        # d(gap)/dr_j = -1
        
        # Reconstruct pair indices
        pair_indices = np.argwhere(tri_mask)
        i_idx = pair_indices[:, 0]
        j_idx = pair_indices[:, 1]
        
        ux = (x[i_idx] - x[j_idx]) / d[i_idx, j_idx]
        uy = (y[i_idx] - y[j_idx]) / d[i_idx, j_idx]
        
        # Accumulate gradients
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            wk = w_pair[k]
            grad_x[i] += wk * ux[k]
            grad_x[j] -= wk * ux[k]
            grad_y[i] += wk * uy[k]
            grad_y[j] -= wk * uy[k]
            grad_r[i] -= wk
            grad_r[j] -= wk
        
        grad[:, 0] += lambda_weight * grad_x
        grad[:, 1] += lambda_weight * grad_y
        grad[:, 2] += lambda_weight * grad_r
    
    return obj, grad

def _gap_objective_v2(params, T, lambda_weight):
    """
    Vectorized contact-graph objective with S-softmin.
    More efficient: precompute pair indices once.
    """
    c = params.reshape(N, 3)
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    
    # Precomputed pair indices (use global)
    global _PAIR_I, _PAIR_J
    if '_PAIR_I' not in globals():
        tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        _PAIR_I, _PAIR_J = np.argwhere(tri_mask).T
        _PAIR_I = _PAIR_I.astype(int)
        _PAIR_J = _PAIR_J.astype(int)
    
    i_idx = _PAIR_I
    j_idx = _PAIR_J
    
    # Compute pairwise distances and gaps
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    d_pair = np.sqrt(dx**2 + dy**2)
    s_pair = r[i_idx] + r[j_idx]
    gap_pair = d_pair - s_pair
    
    # Wall gaps
    gap_left = x - r
    gap_right = 1.0 - x - r
    gap_bottom = y - r
    gap_top = 1.0 - y - r
    
    all_gaps = np.concatenate([gap_left, gap_right, gap_bottom, gap_top, gap_pair])
    
    # S-softmin
    min_gap, weights = _s_softmin(all_gaps, T)
    
    # Objective
    penalty = lambda_weight * np.maximum(0.0, -min_gap)
    obj = -np.sum(r) + penalty
    
    # Gradient
    grad = np.zeros_like(params)
    grad[:, 2] = -1.0
    
    if min_gap < 0:
        w_left = weights[:N]
        w_right = weights[N:2*N]
        w_bottom = weights[2*N:3*N]
        w_top = weights[3*N:4*N]
        w_pair = weights[4*N:]
        
        # Wall contributions
        grad[:, 0] += lambda_weight * (-w_left + w_right)
        grad[:, 1] += lambda_weight * (-w_bottom + w_top)
        grad[:, 2] += lambda_weight * (w_left + w_right + w_bottom + w_top)
        
        # Pairwise contributions
        ux = dx / d_pair
        uy = dy / d_pair
        
        # Vectorized accumulation using np.add.at
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        grad_r = np.zeros(N)
        
        np.add.at(grad_x, i_idx, w_pair * ux)
        np.add.at(grad_x, j_idx, -w_pair * ux)
        np.add.at(grad_y, i_idx, w_pair * uy)
        np.add.at(grad_y, j_idx, -w_pair * uy)
        np.add.at(grad_r, i_idx, -w_pair)
        np.add.at(grad_r, j_idx, -w_pair)
        
        grad[:, 0] += lambda_weight * grad_x
        grad[:, 1] += lambda_weight * grad_y
        grad[:, 2] += lambda_weight * grad_r
    
    return obj, grad

def _optimize_gap(c0, T, lambda_weight, max_iter=300):
    """Run L-BFGS-B with gap objective."""
    x0 = c0.flatten()
    
    def _f(p):
        val, _ = _gap_objective_v2(p, T, lambda_weight)
        return val
    
    def _g(p):
        _, g = _gap_objective_v2(p, T, lambda_weight)
        return g
    
    try:
        result = minimize(
            _f, x0, jac=_g,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        c_new = result.x.reshape(N, 3)
        c_new = _clamp(c_new)
        return c_new
    except Exception:
        return c0

def _grow_radii(c, target_r, max_iter=100):
    """
    Grow all radii toward target_r while maintaining non-negative gaps.
    Uses the gap objective with increasing lambda to enforce constraints.
    """
    c_current = c.copy()
    for i in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
        # Try to increase radii slightly
        c_try = c_current.copy()
        c_try[:, 2] = np.minimum(c_try[:, 2] + target_r * 0.01, target_r)
        c_try = _clamp(c_try)
        
        # Optimize positions to accommodate larger radii
        c_try = _optimize_gap(c_try, T=0.005, lambda_weight=500.0, max_iter=50)
        
        # Check if valid
        if _is_valid(c_try, tol=1e-5):
            c_current = c_try
        else:
            # Try smaller growth
            c_try2 = c_current.copy()
            c_try2[:, 2] = np.minimum(c_try2[:, 2] + target_r * 0.005, target_r)
            c_try2 = _clamp(c_try2)
            c_try2 = _optimize_gap(c_try2, T=0.005, lambda_weight=500.0, max_iter=50)
            if _is_valid(c_try2, tol=1e-5):
                c_current = c_try2
            else:
                break
    
    return c_current

def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])

def _seed_hex():
    """Hexagonal packing initialization."""
    positions = []
    r = 0.15
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(6):
        for col in range(6):
            x = r + col * dx + (row % 2) * dx / 2
            y = r + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append((x, y, r))
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r_small = RNG.uniform(0.05, 0.1)
        positions.append((x, y, r_small))
    return np.array(positions[:N])

def _seed_random():
    x = RNG.uniform(0.1, 0.9, N)
    y = RNG.uniform(0.1, 0.9, N)
    r = RNG.uniform(0.03, 0.08, N)
    return np.column_stack([x, y, r])

def _seed_corner_heavy():
    """Initialize with larger circles in corners, smaller in center."""
    positions = []
    radii = []
    for cx, cy in [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]:
        positions.append((cx, cy))
        radii.append(0.11)
    for cx, cy in [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]:
        positions.append((cx, cy))
        radii.append(0.09)
    for cx, cy in [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]:
        positions.append((cx, cy))
        radii.append(0.08)
    positions.append((0.5, 0.5))
    radii.append(0.07)
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r = RNG.uniform(0.03, 0.06)
        positions.append((x, y))
        radii.append(r)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])

def _seed_symmetric():
    """Symmetric 5x5 grid with center emphasis."""
    positions = []
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            dist_center = np.hypot(x - 0.5, y - 0.5)
            r = 0.08 + 0.03 * dist_center
            positions.append((x, y, r))
    positions.append((0.5, 0.5, 0.04))
    return np.array(positions[:N])

def construct():
    t0 = time.time()
    
    best_c = None
    best_val = -np.inf
    
    # Phase 1: Multi-start with gap-based optimization
    initializers = [_seed_grid, _seed_hex, _seed_corner_heavy, _seed_random, _seed_symmetric]
    
    for init_func in initializers:
        if time.time() > DEADLINE - 25.0:
            break
        
        c0 = init_func()
        c0 = _clamp(c0)
        
        # Stage 1: Relax positions with moderate T
        c_current = _optimize_gap(c0, T=0.02, lambda_weight=100.0, max_iter=200)
        
        # Stage 2: Tighten with lower T
        c_current = _optimize_gap(c_current, T=0.005, lambda_weight=300.0, max_iter=200)
        
        # Stage 3: Very tight
        c_current = _optimize_gap(c_current, T=0.001, lambda_weight=1000.0, max_iter=300)
        
        # Stage 4: Grow radii
        c_current = _grow_radii(c_current, target_r=0.12, max_iter=30)
        
        val = _score(c_current)
        if val > best_val:
            best_val = val
            best_c = c_current.copy()
    
    # Phase 2: More random starts
    n_extra = 10
    for i in range(n_extra):
        if time.time() > DEADLINE - 20.0:
            break
        
        c0 = _seed_random()
        c0 = _clamp(c0)
        
        c_current = _optimize_gap(c0, T=0.015, lambda_weight=150.0, max_iter=150)
        c_current = _optimize_gap(c_current, T=0.003, lambda_weight=500.0, max_iter=200)
        c_current = _optimize_gap(c_current, T=0.001, lambda_weight=1000.0, max_iter=200)
        c_current = _grow_radii(c_current, target_r=0.10, max_iter=20)
        
        val = _score(c_current)
        if val > best_val:
            best_val = val
            best_c = c_current.copy()
    
    # Phase 3: Polish best result
    if best_c is not None and time.time() < DEADLINE - 15.0:
        c_polished = best_c.copy()
        c_polished = _optimize_gap(c_polished, T=0.0005, lambda_weight=2000.0, max_iter=300)
        c_polished = _grow_radii(c_polished, target_r=0.15, max_iter=20)
        
        val = _score(c_polished)
        if val > best_val:
            best_val = val
            best_c = c_polished.copy()
    
    # Final safety
    if best_c is None or best_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_c = np.array(cells[:N])
    
    return best_c
