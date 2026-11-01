"""Pack 50 equal circles in the unit circle using multi-start optimization.

Combines:
- Soft-min surrogate with stabilized log-sum-exp (vectorized gradient)
- Multi-start with diverse initializations
- Force-directed relaxation as a fast local refiner
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

def _optimize_from_start(start_centers, beta_start=50.0, beta_end=2000.0, time_limit=10.0):
    """Optimize using L-BFGS-B with increasing beta."""
    start = time.time()
    x0 = start_centers.flatten().astype(np.float64)
    best_x = x0.copy()
    best_score = _compute_score(start_centers)
    n_betas = 10
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
                            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
            candidate = result.x.reshape(N, 2)
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_x = candidate.flatten()
        except Exception:
            break
    return best_x.reshape(N, 2), best_score

def _force_directed_relax(centers, time_limit=3.0, target_radius=None):
    """Force-directed relaxation with velocity Verlet."""
    start = time.time()
    centers = centers.copy().astype(np.float64)
    if target_radius is None:
        target_radius = _compute_score(centers)
    r = target_radius * 1.005
    dt = 0.002
    damping = 0.85
    velocities = np.zeros_like(centers)
    
    def _compute_forces(pos, r):
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in range(i + 1, N):
                diff = pos[i] - pos[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = 2 * r
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force_mag = 50.0 * overlap * overlap
                    force_dir = diff / dist
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
        for i in range(N):
            dist_from_center = np.sqrt(np.sum(pos[i]**2))
            max_dist = 1.0 - r
            if dist_from_center > max_dist and dist_from_center > 1e-12:
                overlap = dist_from_center - max_dist
                force_mag = 100.0 * overlap * overlap
                force_dir = -pos[i] / dist_from_center
                forces[i] += force_mag * force_dir
        return forces
    
    forces = _compute_forces(centers, r)
    while time.time() - start < time_limit - 1.0:
        centers += velocities * dt + 0.5 * forces * dt * dt
        new_forces = _compute_forces(centers, r)
        velocities = damping * velocities + 0.5 * (forces + new_forces) * dt
        forces = new_forces
        for i in range(N):
            d = np.sqrt(np.sum(centers[i]**2))
            if d > 1.0 - r:
                centers[i] = centers[i] * (1.0 - r) / d
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
    """Polish with Nelder-Mead on the exact score."""
    start = time.time()
    x0 = centers.flatten()
    def neg_score(x):
        c = x.reshape(N, 2)
        return -_compute_score(c)
    best_x = x0.copy()
    best_val = neg_score(x0)
    for _ in range(4):
        if time.time() - start > time_limit - 1.0:
            break
        try:
            result = minimize(neg_score, best_x, method='Nelder-Mead',
                            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
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
        if time.time() - start_time > time_limit - 3.0:
            break
        
        r_mid = (r_low + r_high) / 2.0
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 3.0:
            break
        
        # Try to optimize positions for radius r_mid
        try:
            # Use soft-min with target radius r_mid
            # We want all pair distances >= 2*r_mid and all boundary distances >= r_mid
            # So we maximize min(pair_dists/2, 1-|c_i|) subject to it being >= r_mid
            
            # Simple approach: use the existing optimizer with a high beta
            # and check if the result achieves radius r_mid
            opt_centers, opt_score = _optimize_from_start(
                best_centers, beta_start=100.0, beta_end=3000.0, 
                time_limit=min(5.0, remaining_time - 2.0)
            )
            
            if opt_score >= r_mid - 1e-9:
                # We can achieve this radius
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
        if time.time() > deadline - 40.0:
            break
        
        # Step 1: Force-directed relaxation
        try:
            relaxed = _force_directed_relax(init, time_limit=3.0)
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
        
        # Step 2: Soft-min optimization with L-BFGS-B
        if time.time() > deadline - 25.0:
            break
        try:
            optimized, opt_score = _optimize_from_start(relaxed, beta_start=50.0, 
                                                        beta_end=2000.0, 
                                                        time_limit=10.0)
            if opt_score > best_score:
                best_score = opt_score
                best_centers = optimized
        except Exception:
            pass
    
    # Final refinement on the best solution
    if best_centers is not None:
        # Binary search refinement
        if time.time() < deadline - 20.0:
            try:
                remaining = min(18.0, deadline - time.time() - 5.0)
                refined, ref_score = _binary_search_refine(best_centers, time_limit=remaining)
                if ref_score > best_score:
                    best_score = ref_score
                    best_centers = refined
            except Exception:
                pass
        
        # Nelder-Mead polish
        if time.time() < deadline - 10.0:
            try:
                remaining = min(8.0, deadline - time.time() - 4.0)
                polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
                if pol_score > best_score:
                    best_score = pol_score
                    best_centers = polished
            except Exception:
                pass
        
        # Local search
        if time.time() < deadline - 5.0:
            try:
                remaining = min(4.0, deadline - time.time() - 2.0)
                refined, ref_score = _local_search(best_centers, time_limit=remaining)
                if ref_score > best_score:
                    best_score = ref_score
                    best_centers = refined
            except Exception:
                pass
        
        # Final Nelder-Mead
        if time.time() < deadline - 2.0:
            try:
                remaining = max(2.0, deadline - time.time() - 1.0)
                polished, pol_score = _nelder_mead_polish(best_centers, time_limit=remaining)
                if pol_score > best_score:
                    best_score = pol_score
                    best_centers = polished
            except Exception:
                pass
    
    if best_centers is None:
        best_centers = _get_hex_lattice(N, 0.12)
    
    return best_centers
