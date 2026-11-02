"""Pack 50 equal circles in the unit circle using multi-start optimization.

Combines:
- Soft-min surrogate with stabilized log-sum-exp (vectorized gradient)
- Multi-start with diverse initializations
- Vectorized force-directed relaxation as a fast local refiner
- Multi-stage beta annealing (50→100000) for high-precision convergence
- Binary search on radius with gradient-based position optimization
- High-precision Nelder-Mead polish
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 115.0  # Safety margin for 150s limit

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    dists = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary < 0:
        return 0.0
    pairwise = cdist(centers, centers)
    np.fill_diagonal(pairwise, np.inf)
    min_pair = np.min(pairwise)
    score = min(min_boundary, min_pair / 2.0)
    return max(score, 0.0)

def _objective_and_gradient(x, beta):
    """
    Objective: -soft_min(constraints) with stabilized log-sum-exp.
    Returns (value, gradient). Vectorized for speed.
    """
    n = N
    centers = x.reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    
    # Combine all constraints
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # Stabilized soft min
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    soft_min_val = m - (1.0 / beta) * np.log(S)
    obj_val = -soft_min_val
    
    # Gradient (vectorized)
    weights = exp_terms / S
    grad_centers = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    pair_weights = weights[:n*(n-1)//2]
    mask = pair_weights > 1e-12
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        w_m = pair_weights[mask]
        d_m = dists[i_m, j_m]
        d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
        vecs = (centers[i_m] - centers[j_m]) / d_safe[:, np.newaxis]
        contrib = w_m[:, np.newaxis] * 0.5 * vecs
        np.add.at(grad_centers, i_m, -contrib)
        np.add.at(grad_centers, j_m, contrib)
    
    # Boundary contributions (vectorized)
    bound_weights = weights[n*(n-1)//2:]
    mask_b = bound_weights > 1e-12
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        w_b = bound_weights[idx_b]
        norm_b = norms[idx_b]
        norm_safe = np.where(norm_b > 1e-10, norm_b, 1e-10)
        vecs_b = centers[idx_b] / norm_safe[:, np.newaxis]
        contrib_b = w_b[:, np.newaxis] * vecs_b
        np.add.at(grad_centers, idx_b, contrib_b)
    
    return obj_val, grad_centers.flatten()

def _optimize_from_start(start_centers, beta_start=50.0, beta_end=100000.0, time_limit=10.0):
    """Optimize using L-BFGS-B with multi-stage beta annealing for high-precision convergence."""
    start = time.time()
    x0 = start_centers.flatten().astype(np.float64)
    best_x = x0.copy()
    best_score = _compute_score(start_centers)
    n_betas = 16
    betas = np.geomspace(beta_start, beta_end, n_betas)
    for beta in betas:
        if time.time() - start > time_limit - 2.0:
            break
        try:
            def obj_func(x):
                val, _ = _objective_and_gradient(x, beta)
                return val
            def grad_func(x):
                _, g = _objective_and_gradient(x, beta)
                return g
            result = minimize(obj_func, best_x, jac=grad_func,
                            method='L-BFGS-B',
                            options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 300})
            candidate = result.x.reshape(N, 2)
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_x = candidate.flatten()
        except Exception:
            break
    return best_x.reshape(N, 2), best_score

def _force_directed_relax(centers, time_limit=3.0, target_radius=None):
    """Vectorized force-directed relaxation with adaptive damping and step size."""
    start = time.time()
    centers = centers.copy().astype(np.float64)
    if target_radius is None:
        target_radius = _compute_score(centers)
    r = target_radius * 1.003
    dt = 0.002
    velocities = np.zeros_like(centers)
    
    def _compute_forces(pos, r):
        # Vectorized pairwise forces
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = 2.0 * r
        overlaps = np.maximum(min_dist - dists, 0.0)
        dist_safe = np.where(dists > 1e-12, dists, 1.0)
        force_mags = 80.0 * overlaps * overlaps / dist_safe
        # Only apply forces where there's overlap
        force_mags = np.where(overlaps > 0, force_mags, 0.0)
        np.fill_diagonal(force_mags, 0.0)
        forces = np.sum(force_mags[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces (vectorized)
        norms = np.sqrt(np.sum(pos * pos, axis=1))
        max_dist = 1.0 - r
        bound_overlaps = np.maximum(norms - max_dist, 0.0)
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        bound_force_mags = 150.0 * bound_overlaps * bound_overlaps / norms_safe
        bound_force_mags = np.where(bound_overlaps > 0, bound_force_mags, 0.0)
        forces += bound_force_mags[:, np.newaxis] * (-pos / norms_safe[:, np.newaxis])
        return forces
    
    forces = _compute_forces(centers, r)
    damping = 0.9
    while time.time() - start < time_limit - 1.0:
        centers += velocities * dt + 0.5 * forces * dt * dt
        new_forces = _compute_forces(centers, r)
        velocities = damping * velocities + 0.5 * (forces + new_forces) * dt
        forces = new_forces
        # Clip to boundary
        norms = np.sqrt(np.sum(centers * centers, axis=1))
        max_r = 1.0 - r
        mask = norms > max_r
        if np.any(mask):
            centers[mask] = centers[mask] * (max_r / norms[mask])[:, np.newaxis]
    return centers

def _get_hex_lattice(n, radius, angle=0.0):
    """Generate hexagonal lattice points within unit circle."""
    spacing = 2 * radius
    sqrt3 = np.sqrt(3)
    points = []
    max_k = int(np.ceil(1.0 / spacing)) + 5
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.sqrt(x**2 + y**2) <= 1.0 - radius + 1e-9:
                points.append([x, y])
    if abs(angle) > 1e-9:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        points = [np.dot(rot, p) for p in points]
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return np.array(points[:n])

def _get_ring_initialization(n, radii=None):
    """Initialize with concentric rings."""
    if radii is None:
        rings = [(1, 0.0), (6, 0.22), (12, 0.44), (12, 0.66), (9, 0.82), (10, 0.92)]
    else:
        rings = radii
    points = []
    for count, r in rings:
        if len(points) >= n:
            break
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                if len(points) >= n:
                    break
                points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:n])

def _get_fibonacci_initialization(n):
    """Initialize using Fibonacci spiral."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = 0.95 * np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _get_random_initialization(n, seed):
    """Random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    points = []
    for i in range(n):
        r = np.sqrt(rng.random()) * 0.95
        theta = rng.random() * 2 * np.pi
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _nelder_mead_polish(centers, time_limit=8.0):
    """High-precision polish with Nelder-Mead on the exact score."""
    start = time.time()
    x0 = centers.flatten()
    def neg_score(x):
        c = x.reshape(N, 2)
        return -_compute_score(c)
    best_x = x0.copy()
    best_val = neg_score(x0)
    for _ in range(8):
        if time.time() - start > time_limit - 1.0:
            break
        try:
            result = minimize(neg_score, best_x, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17,
                                   'adaptive': True, 'maxfev': 8000})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            break
    return best_x.reshape(N, 2), -best_val

def _binary_search_refine(centers, time_limit=15.0):
    """Binary search on radius with gradient-based position optimization."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Initial radius estimate
    r_low = best_score
    r_high = best_score * 1.02
    
    # Binary search iterations
    for iteration in range(8):
        if time.time() - start_time > time_limit - 2.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 2.5:
            break
        
        # Try to optimize positions for radius r_mid
        try:
            opt_centers, opt_score = _optimize_from_start(
                best_centers, beta_start=500.0, beta_end=200000.0, 
                time_limit=min(3.5, remaining_time - 1.5)
            )
            
            if opt_score >= r_mid - 1e-10:
                r_low = r_mid
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = opt_centers
            else:
                r_high = r_mid
        except Exception:
            break
    
    return best_centers, best_score

def _local_search(centers, time_limit=8.0):
    """Local search: try moving each point slightly in random directions."""
    start = time.time()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.RandomState(42)
    step_sizes = [0.003, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    for step in step_sizes:
        if time.time() - start > time_limit - 1.0:
            break
        for _ in range(800):
            if time.time() - start > time_limit - 1.0:
                break
            idx = rng.randint(N)
            direction = rng.randn(2)
            direction /= np.linalg.norm(direction)
            candidate = best_centers.copy()
            candidate[idx] += step * direction
            d = np.linalg.norm(candidate[idx])
            if d > 1.0:
                candidate[idx] = candidate[idx] * (0.999 / d)
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_centers = candidate
    return best_centers, best_score

def construct():
    """Construct the best packing found within the time budget."""
    deadline = time.time() + DEADLINE
    best_centers = None
    best_score = 0.0
    
    # Generate multiple initializations
    initializations = []
    
    # Ring initialization
    initializations.append(_get_ring_initialization(N))
    
    # Hexagonal lattice with various angles and radii
    for angle in [0.0, np.pi/6, np.pi/3, np.pi/4, np.pi/2, 2*np.pi/3, np.pi/12, np.pi/8, np.pi/5, 3*np.pi/10]:
        for radius in [0.12, 0.125, 0.128, 0.13, 0.132]:
            try:
                init = _get_hex_lattice(N, radius, angle)
                if len(init) == N:
                    initializations.append(init)
            except Exception:
                pass
    
    # Fibonacci spiral
    initializations.append(_get_fibonacci_initialization(N))
    
    # Random initializations
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        initializations.append(_get_random_initialization(N, seed))
    
    # Process each initialization
    for init_idx, init in enumerate(initializations):
        if time.time() > deadline - 50.0:
            break
        
        # Step 1: Force-directed relaxation (vectorized, faster)
        try:
            relaxed = _force_directed_relax(init, time_limit=2.5)
            score = _compute_score(relaxed)
            if score > best_score:
                best_score = score
                best_centers = relaxed
        except Exception:
            relaxed = init
            score = _compute_score(relaxed)
            if score > best_score:
                best_score = score
                best_centers = relaxed
        
        # Step 2: Soft-min optimization with multi-stage beta annealing
        if time.time() > deadline - 35.0:
            break
        try:
            remaining = min(14.0, deadline - time.time() - 35.0)
            if remaining > 3.0:
                optimized, opt_score = _optimize_from_start(relaxed, beta_start=50.0, 
                                                            beta_end=100000.0, 
                                                            time_limit=remaining)
                if opt_score > best_score:
                    best_score = opt_score
                    best_centers = optimized
        except Exception:
            pass
    
    # Final refinement on the best solution
    if best_centers is not None:
        # Binary search refinement with extended beta range
        if time.time() < deadline - 28.0:
            try:
                remaining = min(22.0, deadline - time.time() - 6.0)
                refined, ref_score = _binary_search_refine(best_centers, time_limit=remaining)
                if ref_score > best_score:
                    best_score = ref_score
                    best_centers = refined
            except Exception:
                pass
        
        # Nelder-Mead polish (high precision)
        if time.time() < deadline - 15.0:
            try:
                remaining = min(12.0, deadline - time.time() - 5.0)
                polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
                if pol_score > best_score:
                    best_score = pol_score
                    best_centers = polished
            except Exception:
                pass
        
        # Local search
        if time.time() < deadline - 8.0:
            try:
                remaining = min(5.0, deadline - time.time() - 3.0)
                refined, ref_score = _local_search(best_centers, time_limit=remaining)
                if ref_score > best_score:
                    best_score = ref_score
                    best_centers = refined
            except Exception:
                pass
        
        # Final high-precision Nelder-Mead
        if time.time() < deadline - 3.0:
            try:
                remaining = max(3.0, deadline - time.time() - 1.0)
                polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
                if pol_score > best_score:
                    best_score = pol_score
                    best_centers = polished
            except Exception:
                pass
    
    if best_centers is None:
        best_centers = _get_hex_lattice(N, 0.12)
    
    return best_centers
