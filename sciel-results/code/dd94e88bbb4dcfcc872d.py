"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _true_score(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    min_wall = np.min(wall_dists)
    return min(min_pair, 2.0 * min_wall) / 2.0

def _soft_min(vals, alpha=50.0):
    """Soft minimum approximation using log-sum-exp for stability."""
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return 0.0
    v_min = vals.min()
    shifted = vals - v_min
    shifted = np.clip(shifted, 0, 100.0 / alpha)
    log_sum = np.log(np.sum(np.exp(-alpha * shifted))) / alpha
    return v_min - log_sum

def _soft_radius(pts, alpha=50.0):
    """Compute a smooth surrogate for the minimum radius."""
    pts = np.asarray(pts, dtype=np.float64)
    # Pairwise distances (only upper triangle to save memory)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = np.sqrt(dists_sq[iu]) / 2.0
    # Wall distances
    wall_dists = np.minimum(
        np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
        np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
    )
    # All constraints
    all_constraints = np.concatenate([pair_dists, wall_dists])
    return _soft_min(all_constraints, alpha)

def _negative_soft_radius(x, alpha=50.0):
    """Negative soft radius for minimization."""
    return -_soft_radius(x.reshape(N, 2), alpha)

def _init_grid():
    """Create a square grid initialization, trying multiple configurations."""
    best_pts = None
    best_r = -1
    
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7)]:
        total = rows * cols
        if total < N:
            continue
        
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        
        if total == N:
            pts = grid
        else:
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]
        
        r = _true_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N].copy()
    
    return best_pts

def _init_hexagonal():
    """Create a hexagonal lattice initialization."""
    s = 1.0 / 7.0
    pts = []
    for i in range(9):
        for j in range(9):
            x = j * s + (i % 2) * s / 2 - 0.5 + s / 4
            y = i * s * np.sqrt(3) / 2 - 0.5 + s * np.sqrt(3) / 4
            if abs(x) <= 0.5 and abs(y) <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts)
    if len(pts) >= N:
        # Select N points that maximize minimum distance (greedy farthest-point)
        selected = []
        remaining = pts.copy()
        for _ in range(N):
            if len(remaining) == 0:
                break
            if len(selected) == 0:
                idx = 0
            else:
                sel = np.array(selected)
                dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
                idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
        
        if len(selected) == N:
            return np.array(selected)
    
    return _init_grid()

def _init_random(seed=0):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.45, 0.45, (N, 2))
    return pts

def _force_relax(pts, n_steps=300, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to spread points apart."""
    pts = pts.copy()
    lr = lr_init
    
    for step in range(n_steps):
        forces = np.zeros_like(pts)
        
        # Wall repulsion
        for k in range(2):
            mask_r = pts[:, k] > 0.45
            forces[mask_r, k] -= (pts[mask_r, k] - 0.45) * 20
            mask_l = pts[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - pts[mask_l, k]) * 20
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        threshold = 0.12
        mask = dists < threshold
        np.fill_diagonal(mask, False)
        
        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 50)
        
        forces += (force_mag[:, :, np.newaxis] * diff).sum(axis=1)
        
        pts += forces * lr
        pts = np.clip(pts, -0.5, 0.5)
        
        lr *= lr_decay
    
    return pts

def _optimize_lbfgsb(pts, time_budget=15.0):
    """Optimize using L-BFGS-B with progressive beta schedule."""
    start_time = time.time()
    
    x0 = pts.ravel()
    best_x = x0.copy()
    best_score = _true_score(pts)
    
    # Progressive beta schedule: start smooth, end sharp
    betas = [20, 40, 80, 150, 300, 500]
    
    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2:
            break
        
        def objective(x):
            return _negative_soft_radius(x, beta)
        
        try:
            maxiter = 500 if beta < 100 else 800
            ftol = 1e-12
            gtol = 1e-10
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )
            
            new_pts = result.x.reshape(N, 2)
            new_score = _true_score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < time_budget - 3:
        try:
            def objective_sharp(x):
                return _negative_soft_radius(x, 1000.0)
            
            result = minimize(
                objective_sharp,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            new_pts = result.x.reshape(N, 2)
            new_score = _true_score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_score

def _optimize_sa(pts, time_budget=10.0):
    """Simulated annealing optimization with adaptive step size."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _true_score(pts)
    current_pts = pts.copy()
    current_score = best_score
    
    T = 0.005
    T_min = 0.00005
    cooling = 0.995
    step_scale = 0.3
    
    while time.time() - start_time < time_budget and T > T_min:
        idx = np.random.randint(N)
        delta = np.random.randn(2) * T * step_scale
        new_pts = current_pts.copy()
        new_pts[idx] = np.clip(current_pts[idx] + delta, -0.5, 0.5)
        
        new_score = _true_score(new_pts)
        delta_score = new_score - current_score
        
        if delta_score > 0 or (T > 1e-10 and np.random.rand() < np.exp(delta_score / T)):
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        T *= cooling
    
    return best_pts, best_score

def _local_refine(pts, time_budget=20.0):
    """Local refinement using coordinate descent with multiple step sizes."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _true_score(pts)
    
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
    
    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break
        
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for idx in range(N):
                if time.time() - start_time > time_budget:
                    break
                for axis in range(2):
                    for sign in [1, -1]:
                        new_pts = best_pts.copy()
                        new_pts[idx, axis] += sign * step
                        if -0.5 <= new_pts[idx, axis] <= 0.5:
                            new_score = _true_score(new_pts)
                            if new_score > best_score + 1e-14:
                                best_pts = new_pts
                                best_score = new_score
                                improved = True
    
    return best_pts, best_score

def _radius_expansion(pts, max_iter=1000, deadline=None):
    """Iteratively expand the radius by pushing points apart."""
    pts = pts.copy()
    current_r = _true_score(pts)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the most constrained pair or wall
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.minimum(
            np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
            np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])
        )
        
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_wall_idx = np.argmin(wall_dists)
        
        min_pair_dist = dists[min_pair_idx]
        min_wall_dist = 2.0 * wall_dists[min_wall_idx]
        
        if min_pair_dist < min_wall_dist:
            i, j = min_pair_idx
            direction = (pts[i] - pts[j]) / (min_pair_dist + 1e-15)
            shift = 0.001
            pts[i] += direction * shift
            pts[j] -= direction * shift
        else:
            i = min_wall_idx
            for axis in range(2):
                if pts[i, axis] > 0:
                    pts[i, axis] -= 0.001
                else:
                    pts[i, axis] += 0.001
        
        pts = np.clip(pts, -0.5, 0.5)
        
        new_r = _true_score(pts)
        if new_r > current_r:
            current_r = new_r
        else:
            break
    
    return pts, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Multiple diverse initializations
    inits = [
        _init_grid(),
        _init_hexagonal(),
        _init_random(0),
        _init_random(1),
        _init_random(2),
        _init_random(3),
    ]
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Force relaxation
        relaxed = _force_relax(init, n_steps=200)
        
        # L-BFGS-B optimization on soft surrogate with progressive beta
        lbfgsb_pts, lbfgsb_score = _optimize_lbfgsb(relaxed, time_budget=12.0)
        if lbfgsb_score > best_score:
            best_pts = lbfgsb_pts.copy()
            best_score = lbfgsb_score
        
        # Simulated annealing
        if time.time() < deadline - 15.0:
            sa_pts, sa_score = _optimize_sa(lbfgsb_pts, time_budget=8.0)
            if sa_score > best_score:
                best_pts = sa_pts.copy()
                best_score = sa_score
        
        # Local refinement
        if time.time() < deadline - 10.0:
            ref_pts, ref_score = _local_refine(best_pts, time_budget=15.0)
            if ref_score > best_score:
                best_pts = ref_pts.copy()
                best_score = ref_score
        
        # Radius expansion
        if time.time() < deadline - 5.0:
            exp_pts, exp_score = _radius_expansion(best_pts, max_iter=500, deadline=deadline)
            if exp_score > best_score:
                best_pts = exp_pts.copy()
                best_score = exp_score
    
    # Final refinement pass
    if best_pts is not None and time.time() < deadline - 2.0:
        final_pts, final_score = _local_refine(best_pts, time_budget=10.0)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    if best_pts is None:
        return _init_grid()
    
    return best_pts
# EVOLVE-BLOCK-END
