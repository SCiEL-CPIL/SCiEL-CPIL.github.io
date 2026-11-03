"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius_fast(pts):
    """Compute the feasible radius for a set of points in [-0.5,0.5]^2."""
    if len(pts) < 2:
        return np.inf
    # Pairwise distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    # Distance to walls
    wall = np.minimum(0.5 - np.abs(pts), axis=1).min()
    return float(min(min_pair, wall))


def _soft_min_radius(pts, beta=80.0):
    """Differentiable soft-min approximation of the radius objective."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e6)
    
    triu_idx = np.triu_indices(N, k=1)
    pair_d = d[triu_idx] / 2.0
    
    # Soft min over pairs
    m = pair_d.min()
    exp_vals = np.exp(-beta * (pair_d - m))
    soft_pair = m - np.log(np.sum(exp_vals)) / beta
    
    # Wall distances
    wall_d = np.minimum(0.5 - np.abs(pts), axis=1)
    m_w = wall_d.min()
    exp_vals_w = np.exp(-beta * (wall_d - m_w))
    soft_wall = m_w - np.log(np.sum(exp_vals_w)) / beta
    
    # Overall soft min
    m_all = min(soft_pair, soft_wall)
    soft_all = m_all - np.log(
        np.exp(-beta * (soft_pair - m_all)) + np.exp(-beta * (soft_wall - m_all))
    ) / beta
    
    return soft_all


def _soft_min_radius_neg(pts, beta=80.0):
    """Negative of soft-min radius (for minimization)."""
    return -_soft_min_radius(pts, beta)


def _soft_min_radius_neg_grad(pts, beta=80.0):
    """Gradient of negative soft-min radius."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e6)
    
    triu_idx = np.triu_indices(N, k=1)
    pair_d = d[triu_idx] / 2.0
    
    # Soft min weights for pairs
    m = pair_d.min()
    exp_vals = np.exp(-beta * (pair_d - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_pair w.r.t. pts
    grad = np.zeros_like(pts)
    for idx in range(len(triu_idx[0])):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        w = weights[idx]
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            grad[i] += w * diff / dist
            grad[j] -= w * diff / dist
    
    # Wall distances
    wall_d = np.minimum(0.5 - np.abs(pts), axis=1)
    m_w = wall_d.min()
    exp_vals_w = np.exp(-beta * (wall_d - m_w))
    weights_w = exp_vals_w / np.sum(exp_vals_w)
    
    for i in range(N):
        w = weights_w[i]
        dx = 0.5 - np.abs(pts[i, 0])
        dy = 0.5 - np.abs(pts[i, 1])
        if dx <= dy:
            grad[i, 0] += w * (-np.sign(pts[i, 0]))
        else:
            grad[i, 1] += w * (-np.sign(pts[i, 1]))
    
    # Combine using soft min of soft_pair and soft_wall
    # For simplicity, approximate the combined gradient
    # The actual gradient would require tracking which term dominates
    
    # Compute soft_pair and soft_wall for weighting
    soft_pair = m - np.log(np.sum(exp_vals)) / beta
    soft_wall = m_w - np.log(np.sum(exp_vals_w)) / beta
    
    m_all = min(soft_pair, soft_wall)
    w_pair = np.exp(-beta * (soft_pair - m_all))
    w_wall = np.exp(-beta * (soft_wall - m_all))
    total_w = w_pair + w_wall
    
    combined_grad = (w_pair * grad + w_wall * grad) / total_w  # Approximation
    
    return -combined_grad


def _hard_disk_expand(pts, r0, dr=0.001, max_iters=2000):
    """Incrementally increase radius while maintaining feasibility via projection."""
    r = r0
    pts = pts.copy()
    
    for it in range(max_iters):
        # Check if current radius is feasible
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        wall = np.minimum(0.5 - np.abs(pts), axis=1).min()
        current_r = min(min_pair, wall)
        
        if current_r < r - 1e-10:
            # Need to project: move points apart
            # Find the most violated pair
            d2 = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            np.fill_diagonal(d2, np.inf)
            min_idx = np.unravel_index(d2.argmin(), d2.shape)
            i, j = min_idx
            
            if d2[i, j] < 2 * r:
                # Move i and j apart
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([dr, 0])
                    dist = dr
                else:
                    diff = diff / dist
                
                # Move each point half the required distance
                required = 2 * r - d2[i, j]
                pts[i] += diff * required / 2
                pts[j] -= diff * required / 2
            else:
                # Check walls
                for k in range(N):
                    for dim in range(2):
                        if abs(pts[k, dim]) + r > 0.5:
                            # Push back
                            if pts[k, dim] > 0:
                                pts[k, dim] = 0.5 - r
                            else:
                                pts[k, dim] = -(0.5 - r)
        
        # Try to increase radius slightly
        r += dr * 0.01
        
        # Clamp to valid region
        pts = np.clip(pts, -(0.5 - r), 0.5 - r)
    
    return pts, _radius_fast(pts)


def _relax_hard_disk(pts, r, n_iter=500):
    """Relax points to achieve radius r using iterative projection."""
    pts = pts.copy()
    pts = np.clip(pts, -(0.5 - r), 0.5 - r)
    
    for it in range(n_iter):
        # Pairwise constraints
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        
        # Find violations
        for i in range(N):
            for j in range(i+1, N):
                if d[i, j] < 2 * r - 1e-10:
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        diff = np.array([1e-3, 0])
                        dist = 1e-3
                    else:
                        diff = diff / dist
                    
                    required = (2 * r - d[i, j]) / 2
                    pts[i] += diff * required
                    pts[j] -= diff * required
        
        # Wall constraints
        for i in range(N):
            for dim in range(2):
                if pts[i, dim] > 0.5 - r:
                    pts[i, dim] = 0.5 - r
                elif pts[i, dim] < -(0.5 - r):
                    pts[i, dim] = -(0.5 - r)
    
    return pts


def _make_hex_init():
    """Create a hexagonal grid initialization."""
    pts = []
    r_est = 0.07
    row_spacing = r_est * np.sqrt(3)
    col_spacing = r_est * 2
    
    y = -0.5 + r_est
    row_idx = 0
    while y < 0.5:
        x_offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        x = -0.5 + r_est + x_offset
        while x < 0.5:
            pts.append([x, y])
            x += col_spacing
        y += row_spacing
        row_idx += 1
    
    pts = np.array(pts[:N])
    return pts


def _make_grid_init():
    """Create a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _slsqp_epigraph_polish(pts, r_init, beta=100.0, maxiter=300):
    """SLSQP epigraph formulation: maximize r subject to constraints."""
    x0 = np.concatenate([pts.flatten(), [r_init]])
    
    def objective(x):
        return -x[-1]  # Maximize r = minimize -r
    
    def constraints(x):
        p = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        cons = []
        # Pairwise: |p_i - p_j| >= 2r
        for i in range(N):
            for j in range(i+1, N):
                diff = p[i] - p[j]
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 
                             np.linalg.norm(x[:2*N].reshape(N,2)[i] - x[:2*N].reshape(N,2)[j]) - 2*x[-1]})
        
        # Wall constraints: |p_ik| + r <= 0.5
        for i in range(N):
            for k in range(2):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, k=k:
                             0.5 - np.abs(x[:2*N].reshape(N,2)[i, k]) - x[-1]})
        
        return cons
    
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)]
    
    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        if result.success:
            new_pts = result.x[:2*N].reshape(N, 2)
            new_pts = np.clip(new_pts, -(0.5 - result.x[-1]), 0.5 - result.x[-1])
            return new_pts, result.x[-1]
    except Exception:
        pass
    
    return pts, r_init


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Multiple initializations
    inits = []
    inits.append(_make_grid_init())
    inits.append(_make_hex_init())
    
    # Jittered grid
    grid = _make_grid_init()
    jitter = grid + np.random.randn(N, 2) * 0.01
    inits.append(jitter)
    
    # Random initialization
    random_pts = np.random.uniform(-0.4, 0.4, (N, 2))
    inits.append(random_pts)
    
    best_pts = inits[0].copy()
    best_r = _radius_fast(best_pts)
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # First, do hard disk relaxation to get a feasible configuration
        r_est = _radius_fast(init_pts)
        
        # Try to expand radius
        relaxed_pts, r_relaxed = _relax_hard_disk(init_pts, r_est, n_iter=200)
        
        if r_relaxed > best_r:
            best_r = r_relaxed
            best_pts = relaxed_pts.copy()
        
        # Now do SLSQP epigraph polish
        if time.time() > deadline - 20:
            break
        
        polished_pts, r_polished = _slsqp_epigraph_polish(relaxed_pts, r_relaxed, maxiter=200)
        
        if r_polished > best_r:
            best_r = r_polished
            best_pts = polished_pts.copy()
        
        # Soft min refinement
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _soft_min_radius_neg,
                best_pts.flatten(),
                args=(80.0,),
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            if result.success or result.fun < -best_r:
                new_pts = result.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -(0.5 - best_r), 0.5 - best_r)
                new_r = _radius_fast(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Final micro-optimization
    if time.time() < deadline - 5:
        try:
            # Higher beta soft min
            result = minimize(
                _soft_min_radius_neg,
                best_pts.flatten(),
                args=(150.0,),
                method='SLSQP',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 50, 'ftol': 1e-15}
            )
            if result.success:
                new_pts = result.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -(0.5 - best_r), 0.5 - best_r)
                new_r = _radius_fast(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_pts = np.clip(best_pts, -(0.5 - best_r + 1e-10), 0.5 - best_r + 1e-10)
    
    return best_pts
# EVOLVE-BLOCK-END
