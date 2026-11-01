"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice + diverse ring-based restarts.
2. Minimize smooth repulsive energy (1/d^2) with L-BFGS-B for global spread.
3. Multi-stage local polish: coarse-to-fine step sizes to maximize min distance.
4. Final high-precision polish with tight tolerances.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)

def _energy_and_grad(params):
    """Smooth repulsive energy: sum 1/d^2 for pairs and walls, with analytic gradient."""
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    E_pair = np.triu(inv_d2, 1).sum()
    
    inv_d4 = inv_d2**2
    np.fill_diagonal(inv_d4, 0.0)
    
    grad_x = -2.0 * (dx * inv_d4).sum(axis=1)
    grad_y = -2.0 * (dy * inv_d4).sum(axis=1)
    
    eps = 1e-8
    dx_w = np.clip(0.5 - x, eps, None)
    dx_w_neg = np.clip(0.5 + x, eps, None)
    dy_w = np.clip(0.5 - y, eps, None)
    dy_w_neg = np.clip(0.5 + y, eps, None)
    
    E_wall = (1.0/dx_w**2 + 1.0/dx_w_neg**2 + 1.0/dy_w**2 + 1.0/dy_w_neg**2).sum()
    
    grad_x += 2.0 / dx_w**3 - 2.0 / dx_w_neg**3
    grad_y += 2.0 / dy_w**3 - 2.0 / dy_w_neg**3
    
    total_E = E_pair + E_wall
    total_grad = np.concatenate([grad_x, grad_y])
    return total_E, total_grad

def _initialize_hexagonal():
    """Initialize with hexagonal pattern, trying multiple spacings."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.085, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.495, 0.495)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _initialize_ring():
    """Ring-based initialization for diversity."""
    rng = np.random.default_rng(42)
    c = np.zeros((N, 2))
    rings = [8, 12, 16, 20, 24, 28, 32, 36]
    radii = [0.08, 0.16, 0.24, 0.32, 0.40, 0.45, 0.48, 0.49]
    
    idx = 0
    c[idx] = [0, 0]
    idx += 1
    
    for n_ring, radius in zip(rings, radii):
        if idx >= N:
            break
        angles = np.linspace(0, 2*np.pi, n_ring, endpoint=False) + rng.uniform(0, 2*np.pi, 1)
        for a in angles:
            if idx >= N:
                break
            c[idx] = [radius * np.cos(a), radius * np.sin(a)]
            idx += 1
    
    # Fill remaining with jittered positions
    while idx < N:
        c[idx] = rng.uniform(-0.48, 0.48, 2)
        idx += 1
    
    c = np.clip(c, -0.495, 0.495)
    return c

def _optimize_energy(c0, time_budget=20.0):
    """Run L-BFGS-B energy minimization with tight tolerances."""
    start = time.time()
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    try:
        res = minimize(
            _energy_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
        r_opt = _compute_radius(c_opt)
        elapsed = time.time() - start
        return c_opt, r_opt, elapsed
    except Exception:
        return c0, _compute_radius(c0), time.time() - start

def _local_polish(c, time_limit=30.0):
    """
    Multi-stage local polish: coarse-to-fine step sizes.
    Stage 1: step ~1e-3, identify and resolve conflicts
    Stage 2: step ~1e-4, refine
    Stage 3: step ~1e-5, high-precision
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    stages = [
        (1e-3, 0.4),   # (step, fraction of time)
        (3e-4, 0.3),
        (1e-4, 0.2),
        (3e-5, 0.1),
    ]
    
    for step, frac in stages:
        stage_deadline = start_time + time_limit * frac
        while time.time() < stage_deadline:
            if time.time() - start_time > time_limit:
                break
            
            # Compute full distance matrix
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, np.inf)
            min_pair_dist = np.min(dist_matrix)
            
            bdists = np.minimum(
                np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
            )
            min_boundary_dist = np.min(bdists)
            
            current_r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            tol = 1e-7
            forces = np.zeros((N, 2))
            
            # Boundary forces
            d_left = 0.5 - c[:, 0]
            d_right = c[:, 0] + 0.5
            d_down = 0.5 - c[:, 1]
            d_up = c[:, 1] + 0.5
            min_d_vec = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
            
            tight_b = np.where(min_d_vec < min_boundary_dist + tol)[0]
            for i in tight_b:
                if d_left[i] == min_d_vec[i]:
                    forces[i, 0] += 1.0
                elif d_right[i] == min_d_vec[i]:
                    forces[i, 0] -= 1.0
                elif d_down[i] == min_d_vec[i]:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0
            
            # Pair forces
            tight_pairs = np.where(dist_matrix < (min_pair_dist + tol))
            i_idx = tight_pairs[0]
            j_idx = tight_pairs[1]
            mask = i_idx < j_idx
            i_idx = i_idx[mask]
            j_idx = j_idx[mask]
            
            if len(i_idx) > 0:
                vecs = c[i_idx] - c[j_idx]
                norms = np.linalg.norm(vecs, axis=1)
                safe_norms = np.where(norms > 1e-10, norms, 1.0)
                unit_vecs = vecs / safe_norms[:, np.newaxis]
                np.add.at(forces, i_idx, unit_vecs)
                np.add.at(forces, j_idx, -unit_vecs)
            
            # Apply forces
            improved = False
            for i in range(N):
                norm_f = np.linalg.norm(forces[i])
                if norm_f > 1e-10:
                    new_c_i = c[i] + (forces[i] / norm_f) * step
                    if np.all(np.abs(new_c_i) < 0.499):
                        old_c_i = c[i].copy()
                        c[i] = new_c_i
                        new_r = _compute_radius(c)
                        if new_r > current_r:
                            current_r = new_r
                            improved = True
                        else:
                            c[i] = old_c_i
            
            if not improved:
                break
    
    return c, current_r

def _final_polish(c, time_limit=15.0):
    """High-precision final polish with very small steps."""
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    step = 1e-5
    stall_count = 0
    
    while time.time() - start_time < time_limit:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_boundary_dist = np.min(bdists)
        
        current_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        tol = 1e-8
        forces = np.zeros((N, 2))
        
        d_left = 0.5 - c[:, 0]
        d_right = c[:, 0] + 0.5
        d_down = 0.5 - c[:, 1]
        d_up = c[:, 1] + 0.5
        min_d_vec = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        
        tight_b = np.where(min_d_vec < min_boundary_dist + tol)[0]
        for i in tight_b:
            if d_left[i] == min_d_vec[i]:
                forces[i, 0] += 1.0
            elif d_right[i] == min_d_vec[i]:
                forces[i, 0] -= 1.0
            elif d_down[i] == min_d_vec[i]:
                forces[i, 1] += 1.0
            else:
                forces[i, 1] -= 1.0
        
        tight_pairs = np.where(dist_matrix < (min_pair_dist + tol))
        i_idx = tight_pairs[0]
        j_idx = tight_pairs[1]
        mask = i_idx < j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        if len(i_idx) > 0:
            vecs = c[i_idx] - c[j_idx]
            norms = np.linalg.norm(vecs, axis=1)
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            unit_vecs = vecs / safe_norms[:, np.newaxis]
            np.add.at(forces, i_idx, unit_vecs)
            np.add.at(forces, j_idx, -unit_vecs)
        
        improved = False
        for i in range(N):
            norm_f = np.linalg.norm(forces[i])
            if norm_f > 1e-10:
                new_c_i = c[i] + (forces[i] / norm_f) * step
                if np.all(np.abs(new_c_i) < 0.499):
                    old_c_i = c[i].copy()
                    c[i] = new_c_i
                    new_r = _compute_radius(c)
                    if new_r > current_r:
                        current_r = new_r
                        improved = True
                    else:
                        c[i] = old_c_i
        
        if not improved:
            stall_count += 1
            step *= 0.5
            if step < 1e-9:
                break
        else:
            stall_count = 0
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_c = None
    best_r = 0.0
    
    # Generate diverse initializations
    inits = [_initialize_hexagonal()]
    inits.append(_initialize_ring())
    
    # Add a jittered hexagonal variant
    hex_c = _initialize_hexagonal()
    rng = np.random.default_rng(123)
    jittered = hex_c + rng.uniform(-0.005, 0.005, (N, 2))
    jittered = np.clip(jittered, -0.495, 0.495)
    inits.append(jittered)
    
    for c0 in inits:
        if time.time() > deadline - 60:
            break
        
        # Energy optimization
        c_opt, r_opt, elapsed = _optimize_energy(c0, time_budget=25.0)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        # Local polish
        remaining = deadline - time.time() - 20
        if remaining > 5 and best_c is not None:
            c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 40.0))
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()
        
        # Final high-precision polish
        remaining = deadline - time.time() - 5
        if remaining > 3 and best_c is not None:
            c_final, r_final = _final_polish(best_c, time_limit=min(remaining, 15.0))
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()
    
    # Final safety check
    if best_c is None:
        best_c = _initialize_hexagonal()
    
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
