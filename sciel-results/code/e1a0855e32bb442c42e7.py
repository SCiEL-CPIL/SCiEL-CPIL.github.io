import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: Minimize the log-sum-exp soft minimum of clearances.
    This is equivalent to maximizing the approximate minimum clearance.
    Vectorized gradient for speed.
    """
    centers = params.reshape(n, 2)
    
    # Calculate all clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    pair_clearances = pair_dists / 2.0
    
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_clearances = 1.0 - norms
    
    all_clearances = np.concatenate([pair_clearances, bound_clearances])
    
    # Numerical stability: shift by min clearance
    c_min = np.min(all_clearances)
    shifted = -beta * (all_clearances - c_min)
    
    exp_terms = np.exp(shifted)
    S = np.sum(exp_terms)
    
    if S == 0 or not np.isfinite(S):
        return 0.0, np.zeros_like(params)
        
    # Objective value: -c_min + (1/beta)*ln(S)
    obj_val = -c_min + (1.0 / beta) * np.log(S)
    
    # Weights w_k = exp(-beta * c_k) / S
    weights = exp_terms / S
    
    # Gradient calculation (vectorized)
    grad = np.zeros_like(centers)
    
    # Gradient of soft-min: ∇f = -(1/S) * Σ exp(-βc_k) * ∇c_k
    # For pair (i,j): c_ij = d_ij/2, ∇_{c_i} c_ij = (c_i - c_j)/(2*d_ij)
    # So contribution to grad[i] from pair (i,j): -w_ij * (c_i-c_j)/(2*d_ij)
    # And contribution to grad[j]: -w_ij * (c_j-c_i)/(2*d_ij) = +w_ij * (c_i-c_j)/(2*d_ij)
    pair_weights = weights[:len(i_idx)]
    
    # Filter out negligible weights and tiny distances for numerical safety
    valid = (pair_weights > 1e-12) & (pair_dists > 1e-10)
    if np.any(valid):
        vi = i_idx[valid]
        vj = j_idx[valid]
        vw = pair_weights[valid]
        vd = pair_dists[valid]
        
        # (c_i - c_j) / d for each valid pair
        vecs = (centers[vi] - centers[vj]) / vd[:, np.newaxis]
        weighted_vecs = vw[:, np.newaxis] * 0.5 * vecs
        
        # grad[i] += -weighted_vec, grad[j] += +weighted_vec
        np.add.at(grad, vi, -weighted_vecs)
        np.add.at(grad, vj, weighted_vecs)
    
    # Boundary: c_i = 1 - |c_i|, ∇c_i = -c_i/|c_i|
    # Contribution: -w_i * (-c_i/|c_i|) = +w_i * c_i/|c_i|
    bound_weights = weights[len(i_idx):]
    valid_b = (bound_weights > 1e-12) & (norms > 1e-10)
    if np.any(valid_b):
        vb = bound_weights[valid_b]
        vnorms = norms[valid_b]
        vidx = np.where(valid_b)[0]
        grad[vidx] += (vb / vnorms)[:, np.newaxis] * centers[vidx]
    
    return obj_val, grad

def _force_directed_relax(centers, n_steps=300, dt=0.001):
    """
    Rigid-body dynamics with discrete wall particles modeling the unit circle boundary.
    Uses short-range repulsive potentials for both inter-circle and wall interactions.
    """
    centers = centers.copy()
    
    # Create discrete wall particles on the unit circle
    n_wall = 200
    wall_angles = np.linspace(0, 2 * np.pi, n_wall, endpoint=False)
    wall_positions = np.column_stack([np.cos(wall_angles), np.sin(wall_angles)])
    
    # Interaction range for wall particles (short-range)
    wall_range = 0.15
    wall_strength = 50.0
    
    # Interaction range for circle-circle
    circle_range = 0.3
    circle_strength = 1.0
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Vectorized pairwise repulsion between circles
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Force magnitude: short-range repulsion
        f_mag = np.zeros_like(dists)
        close_mask = dists < circle_range
        # Inverse-square law within range, zero outside
        f_mag[close_mask] = circle_strength / (dists[close_mask] ** 2)
        np.fill_diagonal(f_mag, 0.0)
        
        # Force vector = f_mag * diff / dist
        forces = np.sum(f_mag[:, :, np.newaxis] * diff / dists[:, :, np.newaxis], axis=1)
        
        # Wall particle interactions
        # For each circle, compute forces from all wall particles
        for i in range(len(centers)):
            to_wall = wall_positions - centers[i]
            wall_dists = np.sqrt(np.sum(to_wall ** 2, axis=1))
            
            # Only interact with wall particles within range
            in_range = wall_dists < wall_range
            if np.any(in_range):
                # Force magnitude: strong repulsion when too close
                f_wall_mag = np.zeros_like(wall_dists)
                f_wall_mag[in_range] = wall_strength / (wall_dists[in_range] ** 2)
                
                # Force vector from wall particles
                wall_force = np.sum(f_wall_mag[:, np.newaxis] * to_wall / wall_dists[:, np.newaxis], axis=0)
                forces[i] += wall_force
        
        # Damping to help convergence
        forces *= 0.95
        
        # Update positions with velocity Verlet-like integration
        centers += dt * forces
        
        # Soft projection: gently pull back if significantly outside
        norms = np.linalg.norm(centers, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            # Gradual pull-back rather than hard projection
            pull_factor = 1.0 - 0.1 * (norms[mask] - 0.98)
            centers[mask] *= pull_factor / norms[mask]
    
    return centers

def _get_hex_lattice(n):
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
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, seed=0):
    """Generate initial points on concentric rings."""
    rng = np.random.default_rng(seed)
    points = []
    
    points.append([0.0, 0.0])
    
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1)
        r = 0.9
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    return np.array(points[:n])

def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _optimize_from_start(centers, start_time, deadline):
    """
    Optimize from a single start using:
    1. Rigid-body dynamics with wall particles for initial relaxation
    2. Annealed beta soft-min gradient optimization
    3. Simulated annealing to escape local minima
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.default_rng(42)
    
    # Phase 1: Rigid-body dynamics with wall particles
    if time.time() < deadline - 25:
        relaxed = _force_directed_relax(best_centers, n_steps=300)
        relaxed_score = _compute_score(relaxed)
        if relaxed_score > best_score:
            best_score = relaxed_score
            best_centers = relaxed
    
    # Phase 2: Gradient-based optimization with annealed beta
    current_beta = 10.0
    sa_temp = 0.01  # Simulated annealing temperature
    beta_multiplier = 1.5
    temp_decay = 0.8
    
    while time.time() < deadline - 5:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, current_beta),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 150, 'gtol': 1e-9, 'ftol': 1e-12}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_centers)):
                break
                
            new_score = _compute_score(new_centers)
            
            if new_score >= best_score:
                best_score = new_score
                best_centers = new_centers.copy()
            else:
                # Simulated annealing: accept worse solutions with prob exp(-delta/T)
                delta = best_score - new_score
                if sa_temp > 1e-6 and rng.random() < np.exp(-delta / sa_temp):
                    best_centers = new_centers.copy()
                else:
                    # Add perturbation and re-relax with dynamics
                    perturbed = best_centers + rng.normal(0, 0.005, size=best_centers.shape)
                    # Re-apply dynamics to settle
                    perturbed = _force_directed_relax(perturbed, n_steps=50)
                    perturbed_score = _compute_score(perturbed)
                    if perturbed_score > best_score:
                        best_score = perturbed_score
                        best_centers = perturbed.copy()
                    else:
                        best_centers = perturbed.copy()
                    
        except Exception:
            break
            
        # Anneal: increase beta, decrease temperature
        current_beta *= beta_multiplier
        sa_temp *= temp_decay
        if current_beta > 10000:
            current_beta = 10000
            # Reset temperature occasionally to allow further exploration
            if sa_temp < 0.0005:
                sa_temp = 0.003
    
    return best_centers, best_score

def _final_polish(centers, start_time, deadline):
    """
    Final high-precision polish with very high beta, dynamics-assisted restarts,
    and multiple refinement passes.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.default_rng(12345)
    
    betas = [5000.0, 10000.0, 20000.0, 50000.0, 100000.0]
    
    for beta in betas:
        # Try multiple perturbations at this beta level
        for attempt in range(6):
            if time.time() > deadline - 3:
                break
            try:
                start_c = best_centers.copy()
                if attempt > 0:
                    # Small perturbation to try different local minima
                    start_c += rng.normal(0, 0.001, size=start_c.shape)
                    # Use dynamics to settle the perturbed configuration
                    start_c = _force_directed_relax(start_c, n_steps=30)
                
                res = minimize(
                    _soft_min_obj_and_grad,
                    start_c.flatten(),
                    args=(N, beta),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'gtol': 1e-13, 'ftol': 1e-15}
                )
                new_centers = res.x.reshape(N, 2)
                
                # Final dynamics relaxation after gradient step
                new_centers = _force_directed_relax(new_centers, n_steps=20)
                
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
            except Exception:
                pass
    
    return best_centers, best_score

def _construct():
    """Main construction function."""
    start_time = time.time()
    
    # Generate diverse initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_ring_initialization(N, seed=1),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=2),
        _get_random_initialization(N, seed=456),
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Allocate time budget for each initialization
    n_inits = len(initializations)
    time_per_init = (DEADLINE - 30) / n_inits
    
    for i, init in enumerate(initializations):
        remaining = DEADLINE - time.time()
        if remaining < 15:
            break
            
        # Use remaining time or allocated time, whichever is smaller
        alloc_time = min(time_per_init, remaining - 15)
        if alloc_time < 5:
            break
            
        init_deadline = time.time() + alloc_time
        
        centers, score = _optimize_from_start(init, start_time, init_deadline)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    if best_centers is None:
        return initializations[0]
        
    # Final polish with remaining time
    if time.time() < DEADLINE - 5:
        best_centers, best_score = _final_polish(best_centers, start_time, DEADLINE)

    return best_centers

def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
