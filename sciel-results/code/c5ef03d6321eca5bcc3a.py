"""Hybrid Force-Directed + L-BFGS-B Soft-Min optimizer for 50 circle packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 130.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values."""
    v = np.array(values)
    if len(v) == 0:
        return 0.0
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    """
    centers = params.reshape(n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    obj = -_soft_min(vals, beta)
    
    # Handle case where vals might be empty or beta is extreme
    min_val = np.min(vals)
    exp_vals = np.exp(-beta * (vals - min_val))
    S = np.sum(exp_vals)
    if S == 0 or not np.isfinite(S):
        return obj, np.zeros_like(centers)
        
    weights = exp_vals / S
    grad = np.zeros_like(centers)
    
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-10:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
            
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-10:
            continue
        if norms[i] > 1e-8:
            grad[i] -= w * centers[i] / norms[i]
            
    return obj, -grad

def _optimize_from_start(centers, beta, maxiter=80, gtol=1e-6):
    """Optimize from a given starting point using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient,
            centers.flatten(),
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return centers

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.12
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    points = np.array(points)
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, num_rings=3):
    """Generate ring-based initialization."""
    centers = []
    centers.append([0.0, 0.0])
    remaining = n - 1
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = 0.3 + ring_idx * 0.25
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.85)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _force_directed_relax(centers, start_time, max_steps=2000, cooling=0.999):
    """
    Force-directed relaxation with hard constraint projection.
    Uses repulsive forces between all pairs to maximize min distance.
    """
    current = centers.copy()
    n = len(current)
    step_size = 0.05
    min_dist = _compute_score(current) * 2.0
    
    # Pre-compute indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for step in range(max_steps):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        # 1. Calculate Forces
        # We want to push circles apart. 
        # Force magnitude ~ 1/d^2 is standard, but for packing, we want to push until they touch.
        # A simple repulsive potential U = 1/d^2 gives F = 2/d^3.
        # To ensure stability, we limit the force and use a small step size.
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        # Vectorized distance calculation
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We only care about pairs that are close, or all pairs if we want global expansion.
        # For packing, global expansion is good.
        # Avoid division by zero
        safe_dists = np.where(dists < 1e-6, 1e-6, dists)
        
        # Force vector F_ij = k * (c_i - c_j) / |c_i - c_j|^3
        # Sum of forces on i from j
        # Using vectorized ops:
        # inv_d3 = 1 / safe_dists**3
        # force_matrix = diff * inv_d3[:, :, np.newaxis]
        # forces = np.sum(force_matrix, axis=1)
        
        # This is O(N^2) memory. For N=50, it's small.
        inv_d3 = 1.0 / (safe_dists ** 3)
        # Multiply each vector by its corresponding scalar
        force_contrib = diff * inv_d3[:, :, np.newaxis]
        forces += np.sum(force_contrib, axis=1)
        
        # Boundary repulsion: Push back if outside? No, we project.
        # But we can add a soft boundary force to keep them inside during relaxation.
        norms = np.linalg.norm(current, axis=1)
        # If close to boundary, push in
        boundary_forces = np.zeros_like(current)
        for i in range(n):
            if norms[i] > 0.8:
                # Push towards center
                boundary_forces[i] = -current[i] / norms[i] * (norms[i] - 0.8) * 2.0
        forces += boundary_forces

        # 2. Update Positions
        # Limit step size
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = 1.0
        scale = np.where(force_norms > max_force, max_force / force_norms, 1.0)
        forces = forces * scale[:, np.newaxis]
        
        current = current + step_size * forces
        
        # 3. Projection
        # a) Boundary: Keep inside unit circle
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            current[mask] = current[mask] / norms[mask][np.newaxis, :]
            
        # b) Overlap: Resolve overlaps by pushing apart
        # We do a few iterations of overlap resolution
        for _ in range(3):
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Find pairs that are overlapping (dist < 2*r_min_est)
            # We don't know r_min_est exactly, so we assume any overlap (dist < 1e-6) or use a target radius.
            # Better: Just resolve any pair with dist < 1.0 (arbitrary small threshold) to ensure no coincident points.
            # Actually, for packing, we want to maximize min dist. 
            # The force-directed method naturally expands. 
            # The projection should just ensure validity (no overlap, inside boundary).
            # Since we are maximizing, we don't have a fixed r. 
            # We just ensure d_ij > 0.
            
            # To be safe, if d_ij < 1e-4, push them apart.
            mask_overlap = dists < 1e-4
            # Iterate over overlapping pairs
            overlap_i, overlap_j = np.where(mask_overlap)
            for i, j in zip(overlap_i, overlap_j):
                if i >= j: continue
                d = dists[i, j]
                if d < 1e-8:
                    # Random push
                    current[i] += np.random.normal(0, 1e-4, 2)
                    current[j] -= np.random.normal(0, 1e-4, 2)
                else:
                    # Push apart by half the overlap
                    # Overlap is 1e-4 - d
                    push = (1e-4 - d) / 2.0
                    vec = (current[i] - current[j]) / d
                    current[i] += vec * push
                    current[j] -= vec * push
        
        # Cool down
        step_size *= cooling
        
        # Stop if converged
        if step_size < 1e-6:
            break
            
    return current

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta."""
    current = centers.copy()
    best_score = _compute_score(current)
    betas = [50.0, 100.0, 200.0, 400.0, 800.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 15:
            break
        polished = _optimize_from_start(current, beta, maxiter=50, gtol=1e-7)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    return current, best_score

def _final_high_precision_polish(centers, start_time):
    """Final high-precision polish with very tight tolerances and high beta."""
    current = centers.copy()
    best_score = _compute_score(current)
    for beta in [1000.0, 2000.0, 5000.0]:
        if time.time() - start_time > DEADLINE - 5:
            break
        polished = _optimize_from_start(current, beta, maxiter=100, gtol=1e-9)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    # Additional local search with very small steps
    rng = np.random.default_rng(777)
    for it in range(10):
        if time.time() - start_time > DEADLINE - 3:
            break
        noisy = current + rng.normal(0, 0.005, current.shape)
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            noisy[mask] *= 0.98 / norms[mask]
        optimized = _optimize_from_start(noisy, beta=2000.0, maxiter=60, gtol=1e-9)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                current = optimized.copy()
    return current, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
        _get_random_initialization(N, seed=555),
    ]
    
    # Phase 1: Force-Directed Relaxation
    # This phase is robust to local minima and ensures a valid, expanded packing.
    for init in initializations:
        if time.time() - start_time > DEADLINE - 50:
            break
        
        # Ensure initial points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # Run force-directed relaxation
        relaxed = _force_directed_relax(init, start_time, max_steps=1500)
        
        if np.all(np.isfinite(relaxed)):
            r = _compute_score(relaxed)
            if r > best_score:
                best_score = r
                best_centers = relaxed.copy()
                
    # If no valid center found (shouldn't happen), fallback
    if best_centers is None:
        best_centers = initializations[0]
        best_score = _compute_score(best_centers)
        
    # Phase 2: L-BFGS-B Polish
    # Hand off the best force-directed result to the precise optimizer.
    if time.time() - start_time < DEADLINE - 30:
        best_centers, best_score = _polish(best_centers, start_time)
        
    # Phase 3: Local Search / Perturbation
    # Try to escape local minima with small perturbations + re-optimization
    if time.time() - start_time < DEADLINE - 20:
        current = best_centers.copy()
        current_score = best_score
        rng = np.random.default_rng(12345)
        max_iters = 10
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 20:
                break
            noisy = current + rng.normal(0, 0.01, current.shape)
            norms = np.linalg.norm(noisy, axis=1)
            mask = norms > 0.95
            if np.any(mask):
                noisy[mask] *= 0.95 / norms[mask]
            
            # Quick polish from perturbed state
            optimized = _optimize_from_start(noisy, beta=100.0, maxiter=40, gtol=1e-7)
            if np.all(np.isfinite(optimized)):
                r = _compute_score(optimized)
                if r > current_score:
                    current_score = r
                    current = optimized.copy()
        if current_score > best_score:
            best_score = current_score
            best_centers = current
            
    # Phase 4: Final High Precision Polish
    if time.time() - start_time < DEADLINE - 10:
        best_centers, best_score = _final_high_precision_polish(best_centers, start_time)
        
    return best_centers
# EVOLVE-BLOCK-END
