"""Packing 50 equal circles in a unit circle using L-BFGS-B with Simulated Annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    # Use a slightly larger estimated radius for better coverage
    r_est = 0.13
    s = 2.0 * r_est
    sqrt3 = np.sqrt(3.0)
    
    points = []
    max_range = int(2.0 / s) + 5
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) <= 0.97:
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
    
    # Select points that maximize minimum distance (greedy)
    selected = []
    remaining = points.copy()
    for _ in range(n):
        if len(remaining) == 0:
            break
        # Find point that maximizes minimum distance to already selected
        if len(selected) == 0:
            # Pick point closest to center first
            dists_to_center = np.linalg.norm(remaining, axis=1)
            idx = np.argmin(dists_to_center)
        else:
            sel = np.array(selected)
            min_dists = []
            for p in remaining:
                d = np.min(np.linalg.norm(sel - p, axis=1))
                min_dists.append(d)
            min_dists = np.array(min_dists)
            idx = np.argmax(min_dists)
        
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    
    return np.array(selected[:n])

def _get_ring_initialization(n, num_rings=5):
    """Generate ring-based initialization with better spacing."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    # Distribute points across rings with increasing radius
    ring_radii = np.linspace(0.25, 0.9, num_rings)
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = ring_radii[ring_idx]
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + ring_idx * 0.3
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.88)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    S = np.sum(exp_terms)
    
    obj = -(m - (1.0 / beta) * np.log(S))
    
    weights = exp_terms / S
    
    grad = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    pair_weights = weights[:n*(n-1)//2]
    d_vecs = centers[i_idx] - centers[j_idx]
    d_mags = pair_dists
    safe_mags = np.where(d_mags > 1e-10, d_mags, 1.0)
    unit_vecs = d_vecs / safe_mags[:, np.newaxis]
    
    # For each pair k: grad[i] += w_k * unit_vec[k] / 2, grad[j] -= w_k * unit_vec[k] / 2
    # Vectorize using np.add.at
    half_w = pair_weights / 2.0
    np.add.at(grad, i_idx, half_w[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, -half_w[:, np.newaxis] * unit_vecs)
    
    # Boundary contributions
    bound_weights = weights[n*(n-1)//2:]
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad -= (bound_weights / safe_norms)[:, np.newaxis] * centers
    
    return obj, -grad

def _optimize_from_start(centers, beta, maxiter=100, gtol=1e-8):
    """Optimize from a given starting point using L-BFGS-B."""
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _nelder_mead_polish(centers, maxiter=200, tol=1e-10):
    """Polish using Nelder-Mead on the raw (nonsmooth) objective."""
    def raw_obj(params):
        c = params.reshape(N, 2)
        return -_compute_score(c)
    
    try:
        res = minimize(
            raw_obj,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta, force-directed, and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 2500.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=80, gtol=1e-10)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
        
        # Add force-directed refinement after each beta stage
        if time.time() - start_time < _DEADLINE - 20:
            refined, refined_score = _force_directed_refine(current, start_time, max_iter=50, step_size=0.0003)
            if refined_score > best_score:
                best_score = refined_score
                current = refined
    
    # Nelder-Mead polish on the nonsmooth objective
    if time.time() - start_time < _DEADLINE - 12:
        nm_result = _nelder_mead_polish(current, maxiter=500)
        if np.all(np.isfinite(nm_result)):
            r = _compute_score(nm_result)
            if r > best_score:
                best_score = r
                current = nm_result
    
    # Final high-precision polish
    if time.time() - start_time < _DEADLINE - 5:
        final = _optimize_from_start(current, beta=5000.0, maxiter=100, gtol=1e-12)
        if np.all(np.isfinite(final)):
            r = _compute_score(final)
            if r > best_score:
                best_score = r
                current = final
    
    return current, best_score

def _force_directed_refine(centers, start_time, max_iter=100, step_size=0.001):
    """
    Force-directed refinement: iteratively push overlapping circles apart
    and pull boundary circles inward.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(max_iter):
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion for overlapping or close circles
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * 0.13 and dist > 1e-10:
                    # Strong repulsion when too close
                    force_mag = (2 * 0.13 - dist) / dist
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
        
        # Boundary attraction: pull circles that are too close to edge
        norms = np.linalg.norm(current, axis=1)
        for i in range(N):
            if norms[i] > 0.85:
                pull = (norms[i] - 0.85) * 0.5
                forces[i] -= pull * current[i] / norms[i]
        
        # Apply forces with adaptive step size
        new_centers = current + step_size * forces
        
        # Clamp to container
        new_norms = np.linalg.norm(new_centers, axis=1)
        mask = new_norms > 0.98
        if np.any(mask):
            new_centers[mask] *= 0.98 / new_norms[mask]
        
        new_score = _compute_score(new_centers)
        if new_score > best_score:
            best_score = new_score
            best = new_centers.copy()
        
        # Always move if it improves, otherwise keep current
        if new_score >= current_score - 1e-6:
            current = new_centers
            current_score = new_score
        else:
            # Reduce step size when stuck
            step_size *= 0.9
    
    return best, best_score

def _simulated_annealing(centers, start_time, max_iter=200):
    """
    Simulated annealing with gradient steps and force-directed refinement
    to escape local minima.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.015  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.97
    
    rng = np.random.default_rng(12345)
    
    for it in range(max_iter):
        if time.time() - start_time > _DEADLINE - 25:
            break
        
        # Generate perturbation with adaptive noise
        noise_scale = 0.008 + 0.025 * T
        noisy = current + rng.normal(0, noise_scale, current.shape)
        
        # Keep inside container
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 0.97
        if np.any(mask):
            noisy[mask] *= 0.97 / norms[mask]
        
        # Gradient step with moderate beta
        optimized = _optimize_from_start(noisy, beta=150.0, maxiter=30, gtol=1e-6)
        if not np.all(np.isfinite(optimized)):
            continue
        
        # Force-directed refinement on the optimized result
        refined, refined_score = _force_directed_refine(optimized, start_time, max_iter=30, step_size=0.0005)
        
        new_score = refined_score
        delta = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, T_min)):
            current = refined
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N, num_rings=5),
        _get_ring_initialization(N, num_rings=6),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
        _get_random_initialization(N, seed=7777),
    ]
    
    # First pass: optimize each initialization with moderate beta
    for init in initializations:
        if time.time() - start_time > _DEADLINE - 60:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.97
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.97 / norms[mask]
        
        optimized = _optimize_from_start(init, beta=50.0, maxiter=120, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    # Second pass: simulated annealing to escape local minima
    if best_centers is not None and time.time() - start_time < _DEADLINE - 50:
        sa_centers, sa_score = _simulated_annealing(best_centers, start_time, max_iter=180)
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers
    
    # Third pass: polish the best solution
    if best_centers is not None and time.time() - start_time < _DEADLINE - 30:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Fourth pass: try to improve with perturbations of the best
    if best_centers is not None and time.time() - start_time < _DEADLINE - 20:
        rng = np.random.default_rng(7)
        for attempt in range(10):
            if time.time() - start_time > _DEADLINE - 20:
                break
            
            noise_scale = 0.004 + 0.008 * attempt
            noisy = best_centers + rng.normal(0, noise_scale, best_centers.shape)
            norms = np.linalg.norm(noisy, axis=1)
            mask = norms > 0.97
            if np.any(mask):
                noisy[mask] *= 0.97 / norms[mask]
            
            optimized = _optimize_from_start(noisy, beta=250.0, maxiter=70, gtol=1e-9)
            if np.all(np.isfinite(optimized)):
                r = _compute_score(optimized)
                if r > best_score:
                    best_score = r
                    best_centers = optimized.copy()
            
            # Force-directed refinement on perturbed result
            if time.time() - start_time < _DEADLINE - 20:
                refined, refined_score = _force_directed_refine(optimized, start_time, max_iter=40, step_size=0.0004)
                if refined_score > best_score:
                    best_score = refined_score
                    best_centers = refined.copy()
        
        # Final polish
        if time.time() - start_time < _DEADLINE - 8:
            best_centers, best_score = _polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
# EVOLVE-BLOCK-END
