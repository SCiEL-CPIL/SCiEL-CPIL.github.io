import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_objective_and_grad(x, beta, iu):
    """
    Objective: soft-min approximation of the radius constraints.
    G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    As beta -> inf, G(x) -> max_k (-c_k(x)) = -min_k c_k(x).
    Minimizing G(x) maximizes min_k c_k(x).
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: c_w = 0.5 - |c_i|
    wall_vals = 0.5 - np.abs(c)
    wall_vals_flat = wall_vals.flatten()
    
    # Pair constraints: c_p = dist_ij / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    pair_vals_flat = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals_flat, pair_vals_flat])
    
    # Compute weights w_k = exp(-beta * c_k) / S
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Wall gradient: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient: dG/dc_i = - sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3*N:]
    
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]
    
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights
    
    grad_pairs = -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    grad += grad_pairs
    
    return obj, grad.ravel()


def _potential_energy_and_grad(x, r_target, iu):
    """
    Smooth potential energy that penalizes overlaps and wall violations.
    Uses a smooth approximation to avoid non-differentiability at the constraint boundary.
    """
    c = x.reshape(-1, 3)
    c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    
    # Pair interactions: smooth overlap penalty
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Smooth overlap: use a smooth approximation to max(0, 2r - d)^2
    # E_pair = sum_{i<j} smooth_max(0, 2r - d_ij)^2
    # We use: for d < 2r, penalty = (2r - d)^2; for d >= 2r, penalty = 0
    # Smooth version: penalty = 0.5 * (2r - d + sqrt((2r-d)^2 + eps^2))^2 - 0.5 * eps^2
    # But simpler: just use the hard penalty with gradient, it works well with L-BFGS-B
    overlap = np.maximum(0, 2 * r_target - dists)
    E_pair = np.sum(overlap ** 2)
    
    # Gradient of pair term
    grad_pair = np.zeros_like(c)
    mask = overlap > 1e-10
    if np.any(mask):
        unit_diff = diff / dists[:, :, np.newaxis]
        # dE/dx_i = 2 * sum_j overlap_ij * (x_i - x_j) / dist_ij
        unit_diff_masked = np.where(mask[:, :, np.newaxis], diff / dists[:, :, np.newaxis], 0)
        grad_pair = 2 * np.sum(overlap[:, :, np.newaxis] * unit_diff_masked, axis=1)
    
    # Wall penalty
    wall_dist = 0.5 - np.abs(c)
    wall_viol = np.maximum(0, r_target - wall_dist)
    E_wall = np.sum(wall_viol ** 2)
    
    # Gradient of wall term
    grad_wall = np.zeros_like(c)
    wall_mask = wall_viol > 1e-10
    grad_wall[wall_mask] = -2 * wall_viol[wall_mask] * np.sign(c[wall_mask])
    
    E = E_pair + E_wall
    grad = grad_pair + grad_wall
    
    return E, grad.ravel()


def _optimize_potential_energy(centers, deadline, n_outer=40):
    """
    Minimize potential energy with adaptive target radius.
    """
    c = centers.copy()
    r = _compute_radius(c)
    r_target = r * 1.002
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for outer in range(n_outer):
        if time.time() > deadline:
            break
        
        # Run L-BFGS-B on potential energy
        try:
            res = minimize(
                _potential_energy_and_grad,
                c.flatten(),
                args=(r_target, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 25}
            )
            c = np.clip(res.x.reshape(-1, 3), -0.5, 0.5)
        except Exception:
            pass
        
        # Update r
        new_r = _compute_radius(c)
        if new_r > r:
            r = new_r
            r_target = r * 1.002
        else:
            # No improvement; slightly increase target to encourage exploration
            r_target = r * 1.001
    
    return c, r


def _optimize_soft_min(x0, deadline, beta_start=20.0, beta_end=20000.0, n_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                best_x,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass
    
    return best_x, best_r


def _force_directed_refinement(centers, n_iters=80, deadline=None):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Strong repulsion when close
        mask = dists < 2.5 * r_current
        force_mag = np.zeros_like(dists)
        force_mag[mask] = 1.0 / (dists[mask] ** 2 + 1e-8)
        
        unit_diff = diff / dists[:, :, np.newaxis]
        unit_diff[~mask] = 0
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        # Attraction to center (weak)
        forces -= 0.001 * c
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        wall_force = np.zeros_like(c)
        wall_force[wall_mask] = 0.5 / (wall_dists[wall_mask] + 1e-8)
        forces -= wall_force * np.sign(c)
        
        # Adaptive step size
        max_force = np.max(np.abs(forces))
        if max_force > 1e-12:
            step = 0.01 / max_force
        else:
            step = 0.01
        
        c += step * forces
        c = np.clip(c, -0.499, 0.499)
        
        # Check if improved
        r_new = _compute_radius(c)
        if r_new > r_current:
            r_current = r_new
    
    return c


def _init_grid(nx, ny, nz, rng):
    """Create a perturbed grid initialization."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    noise_scale = 0.008
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_fcc_like(rng):
    """FCC-like initialization."""
    fcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0, 0, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0]
    ]
    
    nx = ny = nz = 4
    sub_size = 1.0 / nx
    
    centers = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = np.array([-0.5 + ix * sub_size, -0.5 + iy * sub_size, -0.5 + iz * sub_size])
                for offset in fcc_offsets:
                    pos = base + np.array(offset) * sub_size
                    if -0.5 <= pos[0] <= 0.5 and -0.5 <= pos[1] <= 0.5 and -0.5 <= pos[2] <= 0.5:
                        centers.append(pos)
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        remaining = N - len(centers)
        random_pts = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        centers = np.vstack([centers, random_pts])
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_bcc_like(rng):
    """BCC-like initialization."""
    bcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0.5]
    ]
    
    nx = ny = nz = 5
    sub_size = 1.0 / nx
    
    centers = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = np.array([-0.5 + ix * sub_size, -0.5 + iy * sub_size, -0.5 + iz * sub_size])
                for offset in bcc_offsets:
                    pos = base + np.array(offset) * sub_size
                    if -0.5 <= pos[0] <= 0.5 and -0.5 <= pos[1] <= 0.5 and -0.5 <= pos[2] <= 0.5:
                        centers.append(pos)
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        remaining = N - len(centers)
        random_pts = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        centers = np.vstack([centers, random_pts])
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_halton(rng):
    """Halton sequence initialization."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    centers = np.zeros((N, 3))
    for i in range(1, N + 1):
        centers[i - 1, 0] = -0.5 + halton(i, 2)
        centers[i - 1, 1] = -0.5 + halton(i, 3)
        centers[i - 1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_random(rng):
    """Random initialization."""
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(12345)
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate diverse initializations
    inits = [
        _init_grid(5, 5, 4, rng),
        _init_grid(4, 5, 5, rng),
        _init_grid(5, 4, 5, rng),
        _init_fcc_like(rng),
        _init_bcc_like(rng),
        _init_halton(rng),
        _init_random(rng),
    ]
    
    # Phase 1: Force-directed refinement + potential energy + soft-min optimization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        
        c = init_centers.copy()
        
        # Force-directed refinement
        c = _force_directed_refinement(c, n_iters=40, deadline=deadline - 20.0)
        
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Potential energy minimization
        if time.time() > deadline - 20.0:
            break
        
        pe_centers, pe_r = _optimize_potential_energy(c, deadline - 15.0, n_outer=25)
        
        if pe_r > best_radius:
            best_radius = pe_r
            best_centers = pe_centers.copy()
        
        # Soft-min optimization
        if time.time() > deadline - 15.0:
            break
        
        opt_centers, opt_r = _optimize_soft_min(best_centers.flatten(), deadline - 10.0)
        opt_centers = opt_centers.reshape(-1, 3)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_centers.copy()
    
    # Phase 2: Intensive refinement on best result
    if best_centers is not None and time.time() < deadline - 25.0:
        c = best_centers.copy()
        
        # More force-directed iterations
        c = _force_directed_refinement(c, n_iters=60, deadline=deadline - 20.0)
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        # Potential energy with more outer iterations
        if time.time() < deadline - 20.0:
            pe_centers, pe_r = _optimize_potential_energy(c, deadline - 15.0, n_outer=30)
            if pe_r > best_radius:
                best_radius = pe_r
                best_centers = pe_centers.copy()
        
        # Soft-min with higher beta
        if time.time() < deadline - 15.0:
            opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 10.0, beta_start=100.0, beta_end=50000.0, n_steps=6)
            opt_centers = opt_centers.reshape(-1, 3)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers.copy()
    
    # Phase 3: Final high-precision polish
    if best_centers is not None and time.time() < deadline - 10.0:
        c = best_centers.copy()
        
        # Very high beta soft-min for final polish
        if time.time() < deadline - 8.0:
            opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 5.0, beta_start=5000.0, beta_end=100000.0, n_steps=4)
            opt_centers = opt_centers.reshape(-1, 3)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers.copy()
        
        # Final force-directed touch-up
        if time.time() < deadline - 5.0:
            c = _force_directed_refinement(best_centers, n_iters=30, deadline=deadline - 3.0)
            r = _compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_centers = c
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
