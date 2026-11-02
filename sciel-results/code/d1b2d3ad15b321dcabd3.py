"""Packing 50 equal circles in unit square via SLSQP + soft-min polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
_IU = np.triu_indices(N, 1)
_NUM_PAIRS = N * (N - 1) // 2


def _score(c):
    """Largest common radius for centres c (N,2)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    min_bound = np.min(HALF - np.abs(c))
    return min(min_pair, min_bound)


def _soft_min_obj_and_grad(x_flat, beta=100.0):
    """
    Objective: -softmin(all distances) where softmin uses log-sum-exp.
    Returns (objective, gradient).
    """
    pts = x_flat.reshape(N, 2)
    
    # Pairwise distances (unique pairs)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    iu = _IU
    pair_dists = 0.5 * dist[iu]
    
    # Wall distances (4 per point)
    wall_dists = np.concatenate([
        (HALF - pts[:, 0]).flatten(),
        (HALF + pts[:, 0]).flatten(),
        (HALF - pts[:, 1]).flatten(),
        (HALF + pts[:, 1]).flatten()
    ])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min via log-sum-exp
    max_d = np.max(all_dists)
    exp_shifted = np.exp(-beta * (all_dists - max_d))
    sum_exp = np.sum(exp_shifted)
    soft_min = max_d - np.log(sum_exp) / beta
    obj = -soft_min
    
    # Weights
    weights = exp_shifted / sum_exp
    
    # Gradient
    grad_pts = np.zeros((N, 2))
    w_pair = weights[:len(pair_dists)]
    w_wall = weights[len(pair_dists):]
    
    # Pairwise gradient
    for idx in range(len(pair_dists)):
        wi, wj = iu[0][idx], iu[1][idx]
        w = w_pair[idx]
        if w < 1e-15:
            continue
        dvec = pts[wi] - pts[wj]
        dnorm = np.linalg.norm(dvec)
        if dnorm < 1e-12:
            continue
        grad_pts[wi] -= w * dvec / (2.0 * dnorm)
        grad_pts[wj] += w * dvec / (2.0 * dnorm)
    
    # Wall gradient
    grad_pts[:, 0] -= w_wall[0:N]
    grad_pts[:, 0] += w_wall[N:2*N]
    grad_pts[:, 1] -= w_wall[2*N:3*N]
    grad_pts[:, 1] += w_wall[3*N:4*N]
    
    grad_flat = -grad_pts.flatten()
    return obj, grad_flat


def _hex_seed(r):
    """Generate a hexagonal lattice seed with spacing 2r."""
    dx = 2.0 * r
    dy = np.sqrt(3.0) * r
    xs, ys = [], []
    row = 0
    y = -HALF + r
    while y <= HALF - r + 1e-9:
        x = -HALF + r + (0.5 * dx if (row % 2) else 0.0)
        while x <= HALF - r + 1e-9:
            xs.append(x)
            ys.append(y)
            x += dx
        y += dy
        row += 1
    pts = np.array(list(zip(xs, ys)))
    if len(pts) >= N:
        d2 = np.sum(pts * pts, axis=1)
        idx = np.argsort(d2)[:N]
        pts = pts[idx]
    else:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, yy) for yy in ticks for x in ticks])
        pts = np.vstack([pts, grid[:N - len(pts)]])
    return np.clip(pts, -HALF + r, HALF - r)


def _grid_seed(r):
    """Generate a grid seed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        d2 = np.sum(grid * grid, axis=1)
        idx = np.argsort(d2)[:N]
        grid = grid[idx]
    return np.clip(grid, -HALF + r, HALF - r)


def _slsqp_max_radius(x0, r_target, deadline=None, maxiter=500):
    """
    Use SLSQP to find configuration where all pairwise distances >= 2*r_target
    and all wall distances >= r_target.
    """
    x0 = x0.astype(np.float64).flatten()
    
    def obj(x):
        pts = x.reshape(N, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        min_wall = np.min(HALF - np.abs(pts))
        return -(min_pair / (2.0 * r_target) + min_wall / r_target) / 2.0
    
    def grad(x):
        pts = x.reshape(N, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        min_pair = np.min(dist)
        min_wall = np.min(HALF - np.abs(pts))
        
        # Find which pair is closest
        min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
        # Find which wall is closest
        wall_dists = np.concatenate([
            (HALF - pts[:, 0]).flatten(),
            (HALF + pts[:, 0]).flatten(),
            (HALF - pts[:, 1]).flatten(),
            (HALF + pts[:, 1]).flatten()
        ])
        min_wall_idx = np.argmin(wall_dists)
        
        grad_pts = np.zeros((N, 2))
        
        # Pairwise gradient
        i, j = min_pair_idx
        if dist[i, j] > 1e-12:
            dvec = pts[i] - pts[j]
            dnorm = dist[i, j]
            grad_pts[i] += dvec / dnorm
            grad_pts[j] -= dvec / dnorm
        
        # Wall gradient
        if min_wall_idx < N:
            grad_pts[min_wall_idx, 0] += 1.0
        elif min_wall_idx < 2*N:
            grad_pts[min_wall_idx - N, 0] -= 1.0
        elif min_wall_idx < 3*N:
            grad_pts[min_wall_idx - 2*N, 1] += 1.0
        else:
            grad_pts[min_wall_idx - 3*N, 1] -= 1.0
        
        return -grad_pts.flatten() / (2.0 * r_target)
    
    # Constraints: pairwise distances >= 2*r_target, wall distances >= r_target
    constraints = []
    
    # Pairwise constraints (only add the closest few for efficiency)
    diff = x0.reshape(N, 2)[:, None, :] - x0.reshape(N, 2)[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Add constraints for pairs that are close
    for idx in range(_NUM_PAIRS):
        i, j = _IU[0][idx], _IU[1][idx]
        if dist[i, j] < 2.5 * r_target:  # Only constrain nearby pairs
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.linalg.norm(x.reshape(N,2)[i] - x.reshape(N,2)[j]) - 2.0*r_target,
            })
    
    # Wall constraints
    for k in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, k=k: x.reshape(N,2)[k,0] + r_target + HALF,
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, k=k: HALF - r_target - x.reshape(N,2)[k,0],
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, k=k: x.reshape(N,2)[k,1] + r_target + HALF,
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, k=k: HALF - r_target - x.reshape(N,2)[k,1],
        })
    
    bounds = [(-HALF + r_target, HALF - r_target)] * (2*N)
    
    try:
        res = minimize(
            obj, x0, jac=grad, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
        )
        return res.x.reshape(N, 2), res.success
    except Exception:
        return x0.reshape(N, 2), False


def _polish(pts, deadline=None):
    """
    High-precision polish using L-BFGS-B on smooth surrogate with multiple betas.
    """
    pts = pts.astype(np.float64).copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for beta in [50.0, 80.0, 120.0, 200.0]:
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            
            new_pts = res.x.reshape(N, 2)
            new_pts = np.clip(new_pts, -HALF, HALF)
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            continue
    
    return best_pts, best_score


def _violation_polish(pts, r_target, deadline=None, maxiter=2000):
    """
    Polish to satisfy constraints: all pairs >= 2*r_target, all walls >= r_target.
    Uses projected gradient with adaptive step.
    """
    pts = pts.astype(np.float64).copy()
    best_min = 0.0
    min_pair_target = 2 * r_target
    min_wall_target = r_target
    
    for it in range(maxiter):
        if deadline is not None and time.time() > deadline:
            break
        
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces
        overlap = (dist < 2 * r_target) & (dist > 1e-12)
        force_mag = np.where(overlap, (2 * r_target - dist), 0.0)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)
        
        # Wall forces
        wall_force = np.zeros_like(pts)
        for k in range(2):
            too_right = pts[:, k] > HALF - r_target
            too_left = pts[:, k] < -HALF + r_target
            wall_force[too_right, k] = -(pts[too_right, k] - (HALF - r_target))
            wall_force[too_left, k] = (pts[too_left, k] - (-HALF + r_target))
        
        grad += wall_force
        
        # Adaptive step
        step = 0.05
        for _back in range(10):
            new_pts = pts + step * grad
            new_pts = np.clip(new_pts, -HALF + r_target, HALF - r_target)
            
            new_diff = new_pts[:, None, :] - new_pts[None, :, :]
            new_dist = np.sqrt(np.sum(new_diff * new_diff, axis=2))
            np.fill_diagonal(new_dist, np.inf)
            min_pair = np.min(new_dist)
            min_wall = np.min(HALF - np.abs(new_pts))
            
            if min_pair >= min_pair_target and min_wall >= min_wall_target:
                return True, new_pts
            
            if min_pair > best_min:
                best_min = min_pair
                pts = new_pts
                break
            step *= 0.5
        
        if it > 200 and best_min < min_pair_target - 1e-8:
            if it > 800:
                break
    
    dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    success = (min_pair >= min_pair_target) and (min_wall >= min_wall_target)
    return success, pts


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    np.random.seed(42)
    
    # Start with reasonable radius estimates
    r_low = 0.070
    r_high = 0.075
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius
    for it in range(12):
        if time.time() > deadline - 45:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        success = False
        trial_pts = None
        
        # Generate diverse seeds
        seeds = [_hex_seed(r_mid), _grid_seed(r_mid)]
        for _ in range(3):
            seed = _hex_seed(r_mid) + np.random.randn(N, 2) * 0.004
            seed = np.clip(seed, -HALF + r_mid, HALF - r_mid)
            seeds.append(seed)
        
        for seed in seeds:
            if time.time() > deadline - 40:
                break
            
            # First try violation polish (faster)
            ok, pts = _violation_polish(seed, r_mid, deadline=deadline - 35, maxiter=800)
            if ok:
                success = True
                trial_pts = pts
                break
            
            # If violation polish fails, try SLSQP
            if time.time() > deadline - 35:
                continue
            slsqp_pts, slsqp_ok = _slsqp_max_radius(seed, r_mid, deadline=deadline - 30, maxiter=200)
            if slsqp_ok:
                # Verify
                d = slsqp_pts[:, None, :] - slsqp_pts[None, :, :]
                dist = np.sqrt(np.sum(d * d, axis=2))
                np.fill_diagonal(dist, np.inf)
                min_pair = np.min(dist)
                min_wall = np.min(HALF - np.abs(slsqp_pts))
                if min_pair >= 2*r_mid - 1e-9 and min_wall >= r_mid - 1e-9:
                    success = True
                    trial_pts = slsqp_pts
                    break
        
        if success:
            r_low = r_mid
            best_pts = trial_pts.copy()
            best_r = _score(best_pts)
        else:
            r_high = r_mid
    
    # Final refinement
    if time.time() < deadline - 15:
        # Soft-min polish with multiple betas
        for trial in range(5):
            if time.time() > deadline - 3:
                break
            
            if trial == 0:
                perturbed = best_pts.copy()
            elif trial == 1:
                perturbed = best_pts + np.random.randn(N, 2) * 0.003
            elif trial == 2:
                perturbed = best_pts + np.random.randn(N, 2) * 0.001
            else:
                perturbed = _hex_seed(r_low) + np.random.randn(N, 2) * 0.002
            
            perturbed = np.clip(perturbed, -HALF, HALF)
            
            polished_pts, polished_r = _polish(perturbed, deadline=deadline - 2)
            if polished_r > best_r:
                best_pts = polished_pts
                best_r = polished_r
    
    return best_pts
