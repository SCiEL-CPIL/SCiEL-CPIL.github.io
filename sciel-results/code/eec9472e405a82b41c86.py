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
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)


def _feasibility_penalty(centers, r_target):
    """
    Compute a penalty for infeasibility at target radius r_target.
    Returns (penalty, min_violation) where penalty is sum of squared violations
    and min_violation is the worst constraint violation.
    """
    if not np.all(np.isfinite(centers)):
        return np.inf, np.inf
    
    wall_clear = 0.5 - np.abs(centers)
    wall_viol = np.maximum(0, r_target - wall_clear)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_clear = dists / 2.0
    pair_viol = np.maximum(0, r_target - pair_clear)
    
    penalty = np.sum(wall_viol ** 2) + np.sum(pair_viol ** 2)
    min_viol = max(np.max(wall_viol), np.max(pair_viol))
    
    return penalty, min_viol


def _simulated_annealing(centers, r_target, n_iters=5000, deadline=None):
    """
    Simulated annealing to find a feasible packing at target radius r_target.
    """
    c = centers.copy()
    penalty, _ = _feasibility_penalty(c, r_target)
    best_c = c.copy()
    best_penalty = penalty
    
    T_start = 0.05
    T_end = 0.001
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        frac = it / max(1, n_iters - 1)
        T = T_start * (T_end / T_start) ** frac
        
        idx = np.random.randint(N)
        delta = np.random.normal(0, T * 0.5, size=3)
        new_c = c.copy()
        new_c[idx] = np.clip(c[idx] + delta, -0.5, 0.5)
        
        new_penalty, _ = _feasibility_penalty(new_c, r_target)
        delta_p = new_penalty - penalty
        
        if delta_p < 0 or np.random.random() < np.exp(-delta_p / max(T, 1e-12)):
            c = new_c
            penalty = new_penalty
            if penalty < best_penalty:
                best_penalty = penalty
                best_c = c.copy()
                if best_penalty < 1e-10:
                    break
    
    return best_c, best_penalty


def _local_search_penalty(centers, r_target, n_iters=200, deadline=None):
    """
    Greedy local search: repeatedly move the most-violating sphere.
    """
    c = centers.copy()
    penalty, _ = _feasibility_penalty(c, r_target)
    best_c = c.copy()
    best_penalty = penalty
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        wall_clear = 0.5 - np.abs(c)
        wall_viol = np.maximum(0, r_target - wall_clear)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        pair_clear = dists / 2.0
        pair_viol = np.maximum(0, r_target - pair_clear)
        
        sphere_wall_viol = np.max(wall_viol, axis=1)
        sphere_pair_viol = np.max(pair_viol, axis=1)
        sphere_viol = np.maximum(sphere_wall_viol, sphere_pair_viol)
        
        if np.max(sphere_viol) < 1e-10:
            break
        
        worst_idx = np.argmax(sphere_viol)
        
        best_move_penalty = penalty
        best_move = None
        for _ in range(20):
            direction = np.random.normal(0, 1, size=3)
            direction /= np.linalg.norm(direction) + 1e-12
            for ss in step_sizes[:2]:
                trial = c.copy()
                trial[worst_idx] = np.clip(c[worst_idx] + ss * direction, -0.5, 0.5)
                tp, _ = _feasibility_penalty(trial, r_target)
                if tp < best_move_penalty:
                    best_move_penalty = tp
                    best_move = trial
        
        if best_move is not None:
            c = best_move
            penalty = best_move_penalty
            if penalty < best_penalty:
                best_penalty = penalty
                best_c = c.copy()
        else:
            kick_idx = np.random.randint(N)
            c[kick_idx] = np.clip(c[kick_idx] + np.random.normal(0, 0.03, 3), -0.5, 0.5)
            penalty, _ = _feasibility_penalty(c, r_target)
    
    return best_c, best_penalty


def _binary_search_radius(centers, r_lo, r_hi, deadline, n_sa_iters=3000):
    """
    Binary search for the maximum feasible radius starting from initial centers.
    """
    best_c = centers.copy()
    best_r = _compute_radius(centers)
    
    c = centers.copy()
    penalty, _ = _feasibility_penalty(c, best_r * 0.95)
    if penalty < 1e-8:
        r_lo = best_r * 0.95
    else:
        r_test = best_r * 0.9
        c, pen = _simulated_annealing(c, r_test, n_iters=n_sa_iters, deadline=deadline)
        if pen < 1e-8:
            r_lo = r_test
            best_c = c
            best_r = r_test
        else:
            r_lo = r_test * 0.95
            best_c = c
            best_r = r_test * 0.95
    
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        
        r_mid = (r_lo + r_hi) / 2
        c, pen = _simulated_annealing(best_c, r_mid, n_iters=n_sa_iters, deadline=deadline)
        
        if pen < 1e-8:
            r_lo = r_mid
            best_c = c
            best_r = max(best_r, r_mid)
        else:
            r_hi = r_mid
    
    if time.time() < deadline - 2:
        c, pen = _local_search_penalty(best_c, best_r, n_iters=100, deadline=deadline)
        if pen < 1e-8:
            best_c = c
    
    return best_c, best_r


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


def _force_directed_refinement(centers, n_iters=50, deadline=None):
    """
    Force-directed method: each sphere repels others and is attracted to center.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(c)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        mask = dists < 2.5 * r_current
        force_mag = np.zeros_like(dists)
        force_mag[mask] = 1.0 / (dists[mask] ** 2 + 1e-8)
        
        unit_diff = diff / dists[:, :, np.newaxis]
        unit_diff[~mask] = 0
        
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_diff, axis=1)
        
        forces -= 0.001 * c
        
        wall_dists = 0.5 - np.abs(c)
        wall_mask = wall_dists < r_current
        wall_force = np.zeros_like(c)
        wall_force[wall_mask] = 0.5 / (wall_dists[wall_mask] + 1e-8)
        forces -= wall_force * np.sign(c)
        
        max_force = np.max(np.abs(forces))
        if max_force > 1e-12:
            step = 0.01 / max_force
        else:
            step = 0.01
        
        c += step * forces
        c = np.clip(c, -0.499, 0.499)
        
        r_new = _compute_radius(c)
        if r_new > r_current:
            r_current = r_new
    
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    inits = [
        _init_grid(5, 5, 4, rng),
        _init_grid(4, 5, 5, rng),
        _init_grid(5, 4, 5, rng),
        _init_fcc_like(rng),
        _init_bcc_like(rng),
        _init_halton(rng),
    ]
    
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        c = init_centers.copy()
        
        c = _force_directed_refinement(c, n_iters=30, deadline=deadline - 15.0)
        
        r = _compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
        
        if time.time() > deadline - 15.0:
            break
        
        r_hi = min(r * 1.3, 0.5)
        r_lo = r * 0.9
        
        opt_c, opt_r = _binary_search_radius(c, r_lo, r_hi, deadline - 5.0, n_sa_iters=2000)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_c.copy()
    
    if best_centers is not None and time.time() < deadline - 10.0:
        c = best_centers.copy()
        
        r_current = _compute_radius(c)
        c, pen = _simulated_annealing(c, r_current, n_iters=5000, deadline=deadline - 5.0)
        if pen < 1e-8:
            r_new = _compute_radius(c)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = c.copy()
        
        if time.time() < deadline - 5.0:
            r_target = best_radius * 1.02
            c2, pen2 = _simulated_annealing(best_centers, r_target, n_iters=3000, deadline=deadline - 2.0)
            if pen2 < 1e-8:
                r_new2 = _compute_radius(c2)
                if r_new2 > best_radius:
                    best_radius = r_new2
                    best_centers = c2.copy()
    
    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
