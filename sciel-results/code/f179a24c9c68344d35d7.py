import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 135.0  # Safety margin for 150s limit

def compute_radius(pts):
    """
    Compute the maximum radius for the given centers.
    Returns -1e6 if invalid (outside cube).
    """
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)

def neg_radius(pts):
    r = compute_radius(pts)
    return -r

def soft_min_radius(pts, beta=50.0):
    """
    Smooth surrogate for the negative radius using log-sum-exp soft-min.
    This is differentiable and amenable to gradient-based optimization.
    """
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_dists = dists / 2.0
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    pair_flat = pair_dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

def repulsion_anneal(pts, n_steps=80, lr_start=0.008, lr_end=0.0005):
    """
    Vectorized repulsion annealing to spread spheres apart.
    """
    pts = pts.copy().astype(np.float64)
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        lr = lr_start * (lr_end / lr_start) ** (step / max(1, n_steps - 1))
        
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e6)
        
        inv_dist3 = 1.0 / (dists ** 3 + 1e-8)
        np.fill_diagonal(inv_dist3, 0)
        
        forces += np.sum(inv_dist3[:, :, np.newaxis] * diff, axis=1)
        
        for k in range(3):
            pos = pts[:, k]
            wall_dist_pos = 0.5 - pos
            wall_dist_neg = 0.5 + pos
            
            force_pos = np.where(wall_dist_pos < 0.2, -1.0 / (wall_dist_pos + 0.02), 0.0)
            force_neg = np.where(wall_dist_neg < 0.2, 1.0 / (wall_dist_neg + 0.02), 0.0)
            
            forces[:, k] += force_pos + force_neg
        
        pts += lr * forces
        pts = np.clip(pts, -0.495, 0.495)
    
    return pts

def metropolis_mc(pts, n_steps=3000, temp_start=0.03, temp_end=0.002):
    """
    Metropolis Monte Carlo to escape local minima.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_r = compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        
        t = step / max(1, n_steps - 1)
        temp = temp_start * (temp_end / temp_start) ** t
        
        n_move = np.random.randint(1, 4)
        idx = np.random.choice(N, size=n_move, replace=False)
        delta = np.random.normal(0, 0.015, size=(n_move, 3))
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.499, 0.499)
        
        new_r = compute_radius(new_pts)
        
        if new_r >= best_r:
            pts = new_pts
            best_pts = new_pts.copy()
            best_r = new_r
        else:
            delta_r = best_r - new_r
            if np.random.random() < np.exp(-delta_r / max(temp, 1e-10)):
                pts = new_pts
    
    return best_pts

def get_552_init():
    """
    Generate a 5x5x2 grid initialization (exactly 50 points).
    """
    nx, ny, nz = 5, 5, 2
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts += np.random.normal(0, 0.001, size=(N, 3))
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def get_fcc_init():
    """
    Generate an FCC lattice initialization.
    """
    fcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]:
                    p = cell_size * np.array([i, j, k]) + cell_size * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        fcc_pts.append(p)
    
    fcc_pts = np.array(fcc_pts)
    
    if len(fcc_pts) < N:
        return get_552_init()
    
    # Greedy selection for spread
    selected = []
    remaining = fcc_pts.copy()
    for _ in range(N):
        if not remaining.size:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    
    return np.array(selected)

def get_bcc_init():
    """
    Generate a BCC lattice initialization.
    """
    bcc_pts = []
    cell_size = 1.0 / 3.0
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in [[0,0,0], [0.5,0.5,0.5]]:
                    p = cell_size * np.array([i, j, k]) + cell_size * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        bcc_pts.append(p)
    
    bcc_pts = np.array(bcc_pts)
    
    if len(bcc_pts) < N:
        return get_552_init()
    
    selected = []
    remaining = bcc_pts.copy()
    for _ in range(N):
        if not remaining.size:
            break
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    
    return np.array(selected)

def local_refine_smooth(pts):
    """
    Refine using smooth surrogate (L-BFGS-B) followed by exact polish (Nelder-Mead).
    """
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Phase 1: Smooth surrogate optimization with annealing beta
    betas = [15.0, 40.0, 100.0, 250.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        try:
            def obj(x):
                p = x.reshape(N, 3)
                return soft_min_radius(p, beta)
            
            res = minimize(
                obj, x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x = res.x
        except Exception:
            pass
    
    # Phase 2: Exact polish with Nelder-Mead
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499, 0.499)
    
    try:
        res = minimize(
            neg_radius, x_2d,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        x_2d = res.x
    except Exception:
        pass
    
    x_2d = np.clip(x_2d, -0.499, 0.499)
    return x_2d

def local_refine_exact(pts):
    """
    Refine using exact Nelder-Mead only.
    """
    x = pts.flatten().copy()
    
    try:
        res = minimize(
            neg_radius, x,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
        )
        x = res.x
    except Exception:
        pass
    
    x = np.clip(x, -0.499, 0.499)
    return x.reshape(N, 3)

def construct():
    best_pts = None
    best_r = -np.inf
    
    # Generate diverse initializations
    inits = [
        get_552_init(),
        get_fcc_init(),
        get_bcc_init(),
        np.random.uniform(-0.45, 0.45, size=(N, 3))
    ]
    
    # Add more random restarts if time permits
    while time.time() - START_TIME < TIME_LIMIT - 40.0:
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Try each initialization
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        # Apply repulsion annealing first
        annealed = repulsion_anneal(init_pts)
        
        # Alternate between smooth and exact refinement strategies
        if i % 2 == 0:
            refined = local_refine_smooth(annealed)
        else:
            refined = local_refine_exact(annealed)
        
        refined = np.clip(refined, -0.499, 0.499)
        r = compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
        
        # Apply Metropolis MC to escape local minima
        if time.time() - START_TIME < TIME_LIMIT - 25.0:
            mc_pts = metropolis_mc(refined)
            mc_r = compute_radius(mc_pts)
            if mc_r > best_r:
                best_r = mc_r
                best_pts = mc_pts.copy()
        
        # If we have a good result, do a final polish
        if best_r > 0.135 and time.time() - START_TIME < TIME_LIMIT - 5.0:
            try:
                res = minimize(
                    neg_radius, best_pts,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                r2 = compute_radius(res.x)
                if r2 > best_r:
                    best_r = r2
                    best_pts = res.x.copy()
            except Exception:
                pass
    
    # If no valid points found, return grid
    if best_pts is None:
        nx, ny, nz = 5, 5, 2
        xs = (np.arange(nx) + 0.5) / nx - 0.5
        ys = (np.arange(ny) + 0.5) / ny - 0.5
        zs = (np.arange(nz) + 0.5) / nz - 0.5
        grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return grid[:N]
    
    return best_pts
