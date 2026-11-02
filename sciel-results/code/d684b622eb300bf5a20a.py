"""Binary search on radius with projected gradient descent for 50 circles in unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
# Precompute upper triangle indices for pairwise operations
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


def _feasible(pts, r, max_iter=2000, deadline=None):
    """
    Try to find a configuration of N circles of radius r inside the square.
    Uses projected gradient descent to push overlapping circles apart and
    keep them within bounds. Returns (success, pts).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    best_min = 0.0
    min_pair_target = 2 * r - 1e-10
    min_wall_target = r - 1e-10
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute all pairwise distances and forces
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces for overlapping pairs
        overlap = (dist < 2 * r) & (dist > 1e-12)
        force_mag = np.where(overlap, (2 * r - dist), 0.0)
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        force = diff * (force_mag * inv_dist)[:, :, None]
        grad = np.sum(force, axis=1)  # (n, 2)
        
        # Wall constraints: push away from walls
        wall_force = np.zeros_like(pts)
        for k in range(2):
            too_right = pts[:, k] > HALF - r
            too_left = pts[:, k] < -HALF + r
            wall_force[too_right, k] = -(pts[too_right, k] - (HALF - r))
            wall_force[too_left, k] = (pts[too_left, k] - (-HALF + r))
        
        grad += wall_force
        
        # Adaptive step size with backtracking
        step = 0.05
        for _back in range(10):
            new_pts = pts + step * grad
            new_pts = np.clip(new_pts, -HALF + r, HALF - r)
            
            # Check if all constraints are satisfied
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
        
        # Early termination if not making progress
        if it > 200 and best_min < min_pair_target - 1e-8:
            if it > 800:
                break
    
    # Final check
    dist = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    min_wall = np.min(HALF - np.abs(pts))
    success = (min_pair >= min_pair_target) and (min_wall >= min_wall_target)
    return success, pts


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


def _smooth_obj_and_grad(x_flat, beta=80.0):
    """
    Objective: -softmin(all distances) where softmin uses log-sum-exp.
    Returns (objective, gradient) for L-BFGS-B.
    """
    pts = x_flat.reshape(N, 2)
    
    # Pairwise distances (N*(N-1)/2 unique pairs)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Upper triangle indices for unique pairs
    iu = np.triu_indices(N, 1)
    pair_dists = 0.5 * dist[iu]  # (N*(N-1)/2,)
    
    # Wall distances (4 per point)
    wall_dists = np.concatenate([
        (HALF - pts[:, 0]).flatten(),
        (HALF + pts[:, 0]).flatten(),
        (HALF - pts[:, 1]).flatten(),
        (HALF + pts[:, 1]).flatten()
    ])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min via log-sum-exp (numerically stable)
    max_d = np.max(all_dists)
    exp_shifted = np.exp(-beta * (all_dists - max_d))
    sum_exp = np.sum(exp_shifted)
    soft_min = max_d - np.log(sum_exp) / beta
    obj = -soft_min
    
    # Weights for soft-min
    weights = exp_shifted / sum_exp
    
    # Gradient w.r.t. pts
    grad_pts = np.zeros((N, 2))
    
    # Pairwise gradient contribution
    w_pair = weights[:len(pair_dists)]  # weights for pairs
    w_wall = weights[len(pair_dists):]  # weights for walls
    
    # For each pair (i,j), d = |p_i - p_j|/2, grad_d/dp_i = (p_i - p_j)/|p_i - p_j|
    # softmin = -log(sum exp(-beta*d))/beta
    # d(softmin)/d(p_i) = -sum_j w_j * beta * (-1) * d(d_j)/d(p_i) / beta = sum_j w_j * d(d_j)/d(p_i)
    # Wait: softmin = -log(sum exp(-beta*d_k))/beta
    # d(softmin)/dx = -1/beta * (-sum_k exp(-beta*d_k)*beta*d(d_k)/dx) / sum_k exp(-beta*d_k)
    #                = sum_k w_k * d(d_k)/dx
    # So d(softmin)/dx = sum_k w_k * d(d_k)/dx
    # And we minimize -softmin, so grad_obj = -sum_k w_k * d(d_k)/dx
    
    # Pairwise: d_k = |p_i - p_j|/2
    # d(d_k)/d(p_i) = (p_i - p_j) / |p_i - p_j|  (the 1/2 cancels with the 2 in the norm derivative)
    # Actually: d(|p_i-p_j|/2)/d(p_i) = (p_i-p_j)/(2*|p_i-p_j|) * 2 = (p_i-p_j)/|p_i-p_j|
    # Hmm, let me be careful. f = |v|/2 where v = p_i - p_j. df/dp_i = v/|v| * (1/2) * 2 = v/|v|
    # No: d|v|/d v = v/|v|. d(|v|/2)/d v = v/(2|v|). 
    # So d(pair_dist)/d(p_i) = (p_i - p_j) / (2 * |p_i - p_j|)
    
    # For efficiency, accumulate gradient from pairs
    for idx in range(len(pair_dists)):
        wi, wj = iu[0][idx], iu[1][idx]
        w = w_pair[idx]
        if w < 1e-15:
            continue
        dvec = pts[wi] - pts[wj]
        dnorm = np.linalg.norm(dvec)
        if dnorm < 1e-12:
            continue
        # d(pair_dist)/d(pts[wi]) = dvec / (2 * dnorm)
        # d(pair_dist)/d(pts[wj]) = -dvec / (2 * dnorm)
        grad_pts[wi] -= w * dvec / (2.0 * dnorm)
        grad_pts[wj] += w * dvec / (2.0 * dnorm)
    
    # Wall gradient contribution
    # wall_dists[0:N] = HALF - pts[:,0], d/d(pts[:,0]) = -1
    # wall_dists[N:2N] = HALF + pts[:,0], d/d(pts[:,0]) = +1
    # wall_dists[2N:3N] = HALF - pts[:,1], d/d(pts[:,1]) = -1
    # wall_dists[3N:4N] = HALF + pts[:,1], d/d(pts[:,1]) = +1
    grad_pts[:, 0] -= w_wall[0:N]
    grad_pts[:, 0] += w_wall[N:2*N]
    grad_pts[:, 1] -= w_wall[2*N:3*N]
    grad_pts[:, 1] += w_wall[3*N:4*N]
    
    # Objective is -softmin, so gradient of objective = -gradient of softmin
    grad_flat = -grad_pts.flatten()
    
    return obj, grad_flat


def _polish(pts, deadline=None):
    """
    High-precision polish using L-BFGS-B on smooth surrogate.
    """
    pts = pts.astype(np.float64).copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Try multiple betas for robustness - use higher betas for sharper min
    for beta in [60.0, 100.0, 150.0]:
        if deadline is not None and time.time() > deadline:
            break
        
        try:
            res = minimize(
                _smooth_obj_and_grad,
                best_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-11, 'ftol': 1e-13}
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


def _sa_polish(pts, r_target, deadline=None, max_iter=5000):
    """
    Simulated annealing polish targeting radius r_target.
    """
    pts = pts.astype(np.float64).copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    pts_score = best_score
    
    # Temperature schedule
    T0 = 0.005
    T_end = 1e-7
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T0 * (cooling ** it)
        
        # Perturb
        delta = np.random.randn(N, 2) * T
        new_pts = pts + delta
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_score = _score(new_pts)
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score > best_score:
            best_pts = new_pts.copy()
            best_score = new_score
        
        if new_score >= pts_score or np.random.random() < np.exp((new_score - pts_score) / max(T, 1e-15)):
            pts = new_pts
            pts_score = new_score
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 142.0  # safety margin before 150s kill
    np.random.seed(42)
    
    # Start with a reasonable radius estimate
    r_low = 0.070
    r_high = 0.075
    
    best_pts = _hex_seed(0.071).copy()
    best_r = _score(best_pts)
    
    # Binary search on radius - limit to 15 iterations to save time for polish
    for it in range(15):
        if time.time() > deadline - 40:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        success = False
        trial_pts = None
        
        # Generate diverse seeds
        seeds = [_hex_seed(r_mid), _grid_seed(r_mid)]
        
        # Add perturbed hex seeds
        for _ in range(2):
            seed = _hex_seed(r_mid) + np.random.randn(N, 2) * 0.005
            seed = np.clip(seed, -HALF + r_mid, HALF - r_mid)
            seeds.append(seed)
        
        for seed in seeds:
            if time.time() > deadline - 35:
                break
            ok, pts = _feasible(seed, r_mid, max_iter=1200, deadline=deadline - 35)
            if ok:
                success = True
                trial_pts = pts
                break
        
        if success:
            r_low = r_mid
            best_pts = trial_pts.copy()
            best_r = _score(best_pts)
        else:
            r_high = r_mid
    
    # Final refinement: SA polish + L-BFGS polish
    if time.time() < deadline - 10:
        # SA polish first to escape local minima
        sa_pts, sa_r = _sa_polish(best_pts, r_low, deadline=deadline - 15, max_iter=3000)
        if sa_r > best_r:
            best_pts = sa_pts
            best_r = sa_r
        
        # L-BFGS polish with multiple betas and perturbations
        for trial in range(4):
            if time.time() > deadline - 3:
                break
            
            if trial == 0:
                perturbed = best_pts.copy()
            elif trial == 1:
                perturbed = best_pts + np.random.randn(N, 2) * 0.002
            else:
                perturbed = best_pts + np.random.randn(N, 2) * 0.001
            
            perturbed = np.clip(perturbed, -HALF, HALF)
            
            polished_pts, polished_r = _polish(perturbed, deadline=deadline - 2)
            if polished_r > best_r:
                best_pts = polished_pts
                best_r = polished_r
    
    return best_pts
