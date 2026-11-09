"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _energy_and_grad_spherical(theta_phi):
    """
    theta_phi: (n, 2) array of (theta, phi) angles
    Returns (energy, gradient in (theta, phi) space)
    Combined computation for efficiency.
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    n = len(theta)
    
    # Convert to Cartesian
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    pts = np.column_stack([x, y, z])
    
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-15)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(n, k=1)
    energy = float(math.fsum(-np.log(dists[iu])))
    
    # Gradient in Cartesian: grad[i] = sum_j (x_i - x_j) / d_ij^2
    inv_d2 = 1.0 / (dists ** 2)
    inv_d2[np.diag_indices(n)] = 0.0
    grad_cart = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Chain rule to spherical coordinates
    grad_theta = grad_cart[:, 0] * cos_t * cos_p + grad_cart[:, 1] * cos_t * sin_p + grad_cart[:, 2] * (-sin_t)
    grad_phi = grad_cart[:, 0] * (-sin_t * sin_p) + grad_cart[:, 1] * (sin_t * cos_p)
    
    grad = np.column_stack([grad_theta, grad_phi])
    return energy, grad


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _pts_to_spherical(pts):
    """Convert Cartesian points to (theta, phi) spherical coordinates."""
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(pts[:, 1], pts[:, 0])
    return np.column_stack([theta, phi_s])


def _spherical_to_pts(tp):
    """Convert (theta, phi) to Cartesian points on unit sphere."""
    theta = tp[:, 0]
    phi = tp[:, 1]
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _random_sphere_points(seed):
    """Generate random points on sphere with given seed."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    pts = v / norms
    return pts


def _ring_initialization():
    """Generate ring-based initialization for diversity."""
    pts_list = []
    n_rings = 10
    pts_per_ring = N // n_rings
    remainder = N - pts_per_ring * n_rings
    
    for ring_idx in range(n_rings):
        z = 1.0 - 2.0 * (ring_idx + 0.5) / n_rings
        r = np.sqrt(max(0.0, 1.0 - z * z))
        n_in_ring = pts_per_ring + (1 if ring_idx < remainder else 0)
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False) + ring_idx * 0.3
        for a in angles:
            pts_list.append([r * np.cos(a), r * np.sin(a), z])
    
    pts = np.array(pts_list[:N])
    return pts


def _perturbed_fibonacci(scale, seed):
    """Fibonacci points with random perturbation."""
    fib = _fibonacci_spiral()
    rng = np.random.default_rng(seed)
    perturbed = fib + scale * rng.standard_normal((N, 3))
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    return perturbed / norms


def _antipodal_pairs(seed):
    """Start with pairs of near-antipodal points, then perturb."""
    rng = np.random.default_rng(seed)
    half = N // 2
    base = rng.standard_normal((half, 3))
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    base = base / norms
    
    if N % 2 == 0:
        pts = np.vstack([base, -base])
    else:
        extra = rng.standard_normal((1, 3))
        extra = extra / np.linalg.norm(extra)
        pts = np.vstack([base, -base, extra])
    
    # Small perturbation to break symmetry
    pts = pts + 0.01 * rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _hexagonal_like(seed):
    """Generate points inspired by hexagonal packing on sphere."""
    pts = _fibonacci_spiral()
    rng = np.random.default_rng(seed)
    # Random rotation
    theta = rng.uniform(0, 2 * np.pi)
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    pts = pts @ rot.T
    # Small perturbation
    pts = pts + 0.02 * rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _clustered_random(n_clusters, seed):
    """Clustered random initialization."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, 3))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    pts = []
    per_cluster = N // n_clusters
    for i in range(n_clusters):
        cluster = centers[i:i+1] + 0.3 * rng.standard_normal((per_cluster, 3))
        pts.append(cluster)
    
    pts = np.vstack(pts)
    if len(pts) < N:
        extra = rng.standard_normal((N - len(pts), 3))
        pts = np.vstack([pts, extra])
    pts = pts[:N]
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _basin_hopping_optimize(tp_init, deadline, max_outer=50, max_inner=300, 
                            noise_scale=0.05, noise_decay=0.95):
    """
    Basin hopping with L-BFGS-B inner optimizer.
    Uses simulated annealing acceptance to escape local minima.
    """
    def objective(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _energy_and_grad_spherical(tp_2d)[0]
    
    def gradient(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _energy_and_grad_spherical(tp_2d)[1].ravel()
    
    best_tp = tp_init.copy()
    best_energy = objective(best_tp.ravel())
    current_tp = tp_init.copy()
    current_energy = best_energy
    
    temperature = 1.0
    rng = np.random.default_rng(12345)
    
    for outer in range(max_outer):
        if time.time() > deadline - 15.0:
            break
        
        # Inner L-BFGS-B optimization
        try:
            result = minimize(
                objective,
                current_tp.ravel(),
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': max_inner,
                    'maxfun': max_inner * 5,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxls': 40
                }
            )
            
            if result.success or result.fun < current_energy:
                new_tp = result.x.reshape(-1, 2)
                new_energy = result.fun
            else:
                new_tp = current_tp
                new_energy = current_energy
        except Exception:
            new_tp = current_tp
            new_energy = current_energy
        
        # Accept/reject based on simulated annealing
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
            current_tp = new_tp
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_tp = current_tp.copy()
        
        # Generate perturbation for next iteration
        noise = noise_scale * noise_decay ** outer
        perturbed = current_tp + noise * rng.standard_normal((N, 2))
        # Wrap phi to [0, 2pi)
        perturbed[:, 1] = perturbed[:, 1] % (2 * np.pi)
        current_tp = perturbed
        
        temperature *= 0.98
    
    return best_tp, best_energy


def _high_precision_polish(tp, deadline):
    """
    Final high-precision polish with progressively tighter tolerances.
    Multiple passes with decreasing step sizes.
    """
    def objective(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _energy_and_grad_spherical(tp_2d)[0]
    
    def gradient(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _energy_and_grad_spherical(tp_2d)[1].ravel()
    
    current_tp = tp.copy()
    current_energy = objective(current_tp.ravel())
    
    # Progressive tightening
    tolerances = [
        (1e-12, 1e-10, 2000),
        (1e-13, 1e-11, 3000),
        (1e-14, 1e-12, 5000),
        (1e-15, 1e-13, 10000),
    ]
    
    for ftol, gtol, maxiter in tolerances:
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(
                objective,
                current_tp.ravel(),
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'maxfun': maxiter * 3,
                    'ftol': ftol,
                    'gtol': gtol,
                    'maxls': 100
                }
            )
            
            if result.fun < current_energy:
                current_energy = result.fun
                current_tp = result.x.reshape(-1, 2)
        except Exception:
            pass
    
    return current_tp, current_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_energy = np.inf
    best_tp = None
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    pts_fib = _fibonacci_spiral()
    tp_fib = _pts_to_spherical(pts_fib)
    initializations.append(tp_fib)
    
    # 2. Ring-based
    pts_ring = _ring_initialization()
    tp_ring = _pts_to_spherical(pts_ring)
    initializations.append(tp_ring)
    
    # 3. Perturbed Fibonacci with various scales
    for scale in [0.02, 0.05, 0.1, 0.15, 0.25]:
        for seed in [1, 7, 42]:
            pts = _perturbed_fibonacci(scale, seed)
            tp = _pts_to_spherical(pts)
            initializations.append(tp)
    
    # 4. Antipodal pairs
    for seed in [1, 42, 99]:
        pts = _antipodal_pairs(seed)
        tp = _pts_to_spherical(pts)
        initializations.append(tp)
    
    # 5. Hexagonal-like
    for seed in [1, 42]:
        pts = _hexagonal_like(seed)
        tp = _pts_to_spherical(pts)
        initializations.append(tp)
    
    # 6. Clustered random
    for n_clusters in [3, 5, 7]:
        for seed in [1, 42]:
            pts = _clustered_random(n_clusters, seed)
            tp = _pts_to_spherical(pts)
            initializations.append(tp)
    
    # 7. Random with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        pts_rand = _random_sphere_points(seed)
        tp_rand = _pts_to_spherical(pts_rand)
        initializations.append(tp_rand)
    
    # Sort initializations by initial energy (best first)
    init_energies = []
    for tp in initializations:
        e, _ = _energy_and_grad_spherical(tp)
        init_energies.append((e, tp))
    init_energies.sort(key=lambda x: x[0])
    
    # Optimize each initialization with basin hopping
    for idx, (e_init, tp_init) in enumerate(init_energies):
        if time.time() > deadline - 30.0:
            break
        
        # Quick L-BFGS first to get a good starting point
        def objective(tp_flat):
            tp_2d = tp_flat.reshape(-1, 2)
            return _energy_and_grad_spherical(tp_2d)[0]
        
        def gradient(tp_flat):
            tp_2d = tp_flat.reshape(-1, 2)
            return _energy_and_grad_spherical(tp_2d)[1].ravel()
        
        try:
            result = minimize(
                objective,
                tp_init.ravel(),
                jac=gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxfun': 2500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 40
                }
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_tp = result.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Basin hopping for top initializations
        if idx < 8 and time.time() < deadline - 20.0:
            bh_tp, bh_energy = _basin_hopping_optimize(
                best_tp if best_tp is not None else tp_init,
                deadline - 10.0,
                max_outer=30,
                max_inner=200,
                noise_scale=0.03,
                noise_decay=0.97
            )
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_tp = bh_tp
    
    # Final high-precision polish on the best result
    if best_tp is not None and time.time() < deadline - 5.0:
        polished_tp, polished_energy = _high_precision_polish(best_tp, deadline)
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_tp = polished_tp
    
    # Convert back to Cartesian
    if best_tp is None:
        # Fallback to Fibonacci if something went wrong
        pts = _fibonacci_spiral()
    else:
        pts = _spherical_to_pts(best_tp)
    
    # Ensure all points are on unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
