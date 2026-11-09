"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.clip(d, 1e-15, None)
    terms = [-math.log(di) for di in d]
    return math.fsum(terms)


def _log_energy_and_grad(pts):
    """Compute log(1/d) energy and gradient."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    dist = np.clip(dist, 1e-15, None)
    
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -math.fsum([math.log(di) for di in d])
    
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = ~np.eye(N, dtype=bool)
    inv_dist_sq[mask] = 1.0 / (dist_sq[mask] + 1e-30)
    
    weighted_sum = inv_dist_sq @ pts
    row_sums = inv_dist_sq.sum(axis=1)
    
    grad = weighted_sum - pts * row_sums[:, None]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - dot * pts


def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_random(seed):
    """Random points on sphere with given seed."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _initial_perturbed(base, seed, magnitude=0.05):
    """Perturb base configuration in tangent space."""
    rng = np.random.default_rng(seed)
    pts = base.copy()
    for i in range(N):
        v = rng.standard_normal(3)
        v = v - np.dot(v, pts[i]) * pts[i]
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            v = v / nv
            pts[i] = pts[i] + magnitude * v
            pts[i] = pts[i] / np.linalg.norm(pts[i])
    return pts


def _optimize_lbfgs(pts, max_iter=5000, ftol=1e-12, gtol=1e-10):
    """Optimize using L-BFGS-B with tangent-projected gradient."""
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _log_energy_and_grad(p)
        pg = _tangent_project(p, g)
        return e, pg.flatten()
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if result.fun < best_energy:
            result_pts = _project_to_sphere(result.x.reshape(N, 3))
            result_energy = _log_energy(result_pts)
            if result_energy < best_energy:
                best_pts = result_pts
                best_energy = result_energy
    except Exception:
        pass
    
    return best_pts, best_energy


def _optimize_bfgs(pts, max_iter=2000):
    """Optimize using BFGS with tangent-projected gradient."""
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _log_energy_and_grad(p)
        pg = _tangent_project(p, g)
        return e, pg.flatten()
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    try:
        result = minimize(
            objective,
            x0,
            jac=True,
            method='BFGS',
            options={'maxiter': max_iter, 'gtol': 1e-10, 'disp': False}
        )
        if result.fun < best_energy:
            result_pts = _project_to_sphere(result.x.reshape(N, 3))
            result_energy = _log_energy(result_pts)
            if result_energy < best_energy:
                best_pts = result_pts
                best_energy = result_energy
    except Exception:
        pass
    
    return best_pts, best_energy


def _basin_hopping(pts, n_iter=50, stepsize=0.1):
    """Basin hopping with L-BFGS polish at each step."""
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    current_pts = pts.copy()
    current_energy = best_energy
    
    rng = np.random.default_rng(12345)
    
    for it in range(n_iter):
        # Generate perturbation in tangent space
        perturbed = current_pts.copy()
        for i in range(N):
            v = rng.standard_normal(3)
            v = v - np.dot(v, perturbed[i]) * perturbed[i]
            nv = np.linalg.norm(v)
            if nv > 1e-12:
                v = v / nv
                perturbed[i] = perturbed[i] + stepsize * v
                perturbed[i] = perturbed[i] / np.linalg.norm(perturbed[i])
        
        # Polish the perturbed configuration
        polished, polished_energy = _optimize_lbfgs(perturbed, max_iter=500, ftol=1e-10, gtol=1e-8)
        
        # Metropolis acceptance
        delta_e = polished_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / 0.5):
            current_pts = polished
            current_energy = polished_energy
        
        if current_energy < best_energy:
            best_pts = current_pts.copy()
            best_energy = current_energy
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Phase 1: L-BFGS from Fibonacci
    if time.time() < deadline - 30:
        pts_opt, energy_opt = _optimize_lbfgs(pts, max_iter=5000, ftol=1e-12, gtol=1e-10)
        if energy_opt < best_energy:
            best_pts = pts_opt
            best_energy = energy_opt
    
    # Phase 2: Multiple diverse restarts
    restart_configs = []
    
    # Perturbed Fibonacci variants
    for seed in [42, 123, 456, 789, 101112]:
        if time.time() >= deadline - 40:
            break
        cfg = _initial_perturbed(best_pts, seed, magnitude=0.05)
        restart_configs.append(cfg)
    
    # Random starts
    for seed in [7, 17, 27, 37, 47]:
        if time.time() >= deadline - 40:
            break
        cfg = _initial_random(seed)
        restart_configs.append(cfg)
    
    for idx, cfg in enumerate(restart_configs):
        if time.time() >= deadline - 20:
            break
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # L-BFGS optimization
        pts_opt, energy_opt = _optimize_lbfgs(cfg, max_iter=min(3000, int(remaining * 50)), ftol=1e-11, gtol=1e-9)
        if energy_opt < best_energy:
            best_pts = pts_opt
            best_energy = energy_opt
    
    # Phase 3: Basin hopping from best configuration
    if time.time() < deadline - 25:
        remaining = deadline - time.time()
        n_bh = min(30, int(remaining / 1.5))
        if n_bh > 5:
            pts_bh, energy_bh = _basin_hopping(best_pts, n_iter=n_bh, stepsize=0.1)
            if energy_bh < best_energy:
                best_pts = pts_bh
                best_energy = energy_bh
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining > 15:
            pts_polish, energy_polish = _optimize_lbfgs(best_pts, max_iter=10000, ftol=1e-15, gtol=1e-14)
            if energy_polish < best_energy:
                best_pts = pts_polish
                best_energy = energy_polish
    
    # Additional polish with BFGS if time permits
    if time.time() < deadline - 8:
        remaining = deadline - time.time()
        if remaining > 10:
            pts_bfgs, energy_bfgs = _optimize_bfgs(best_pts, max_iter=min(500, int(remaining * 30)))
            if energy_bfgs < best_energy:
                best_pts = pts_bfgs
                best_energy = energy_bfgs
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += 1e-8 * best_pts[i]
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
