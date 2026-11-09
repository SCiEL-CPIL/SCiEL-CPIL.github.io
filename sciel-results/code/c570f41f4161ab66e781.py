"""Direct non-smooth optimization for 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    d_boundary = np.min(0.5 - np.abs(centres), axis=1)
    min_boundary = np.min(d_boundary)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_boundary)

def _soft_min_value_and_grad(centres, beta=200.0):
    """
    Compute soft-min of all constraint values (pairwise distances/2 and boundary clearances)
    and its gradient using log-sum-exp trick.
    """
    n = len(centres)
    
    # Boundary constraints
    d_boundary = 0.5 - np.abs(centres)  # shape (n, 3)
    b_vals = d_boundary.ravel()  # shape (3n,)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    p_vals = dists / 2.0
    np.fill_diagonal(p_vals, np.inf)
    
    # Find global minimum for numerical stability
    b_min = np.min(b_vals)
    p_min = np.min(p_vals)
    all_min = min(b_min, p_min)
    
    # Compute weights using log-sum-exp
    b_shifted = b_vals - all_min
    p_shifted = p_vals.ravel() - all_min
    
    # Clip to avoid overflow
    b_weights = np.exp(-beta * np.clip(b_shifted, 0, 50))
    p_weights = np.exp(-beta * np.clip(p_shifted, 0, 50))
    
    total_weight = b_weights.sum() + p_weights.sum()
    
    if total_weight < 1e-300:
        return all_min, np.zeros_like(centres)
    
    # Soft-min value
    soft_min_val = all_min + np.log(total_weight) / beta
    
    # Compute gradient from boundary constraints
    grad = np.zeros_like(centres)
    b_weights_reshaped = b_weights.reshape(n, 3)
    signs = np.sign(centres)
    grad_boundary = -b_weights_reshaped * signs / total_weight
    grad += grad_boundary
    
    # Compute gradient from pairwise constraints
    p_weights_reshaped = p_weights.reshape(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            w = p_weights_reshaped[i, j] / total_weight
            if w < 1e-15:
                continue
            d = dists[i, j]
            if d < 1e-12:
                continue
            unit = (centres[i] - centres[j]) / d
            grad[i] += w * unit / 2.0
            grad[j] -= w * unit / 2.0
    
    return soft_min_val, grad


def _fcc_lattice_init():
    """
    Generate FCC lattice points within the cube and select the best 100.
    """
    a1 = np.array([0.5, 0.5, 0.0])
    a2 = np.array([0.5, 0.0, 0.5])
    a3 = np.array([0.0, 0.5, 0.5])
    
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5])
    ]
    
    points = []
    for offset in offsets:
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    p = i * a1 + j * a2 + k * a3 + offset
                    if np.all(p >= -0.5 - 1e-9) and np.all(p <= 0.5 + 1e-9):
                        p_clamped = np.clip(p, -0.49, 0.49)
                        points.append(p_clamped)
    
    points = np.array(points)
    
    if len(points) < N:
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Compute clearance for each point
    d_boundary = np.min(0.5 - np.abs(points), axis=1)
    n_pts = len(points)
    min_pair_dist = np.full(n_pts, np.inf)
    for i in range(n_pts):
        if n_pts - i - 1 > 0:
            diffs = points[i+1:] - points[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                min_pair_dist[i] = np.min(dists)
                min_pair_dist[i+1:] = np.minimum(min_pair_dist[i+1:], dists)
    
    clearance = np.minimum(d_boundary, min_pair_dist / 2.0)
    
    if n_pts > N:
        idx = np.argsort(-clearance)[:N]
        points = points[idx]
    
    return points[:N].astype(np.float64)


def _gradient_step(centres, step_size, beta=200.0):
    """
    One step of gradient ascent on the soft-min surrogate objective.
    """
    n = len(centres)
    
    soft_val, grad = _soft_min_value_and_grad(centres, beta=beta)
    
    # Compute hard radius for tracking
    d_boundary = 0.5 - np.abs(centres)
    min_boundary_val = np.min(d_boundary)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_val = np.min(dists)
    r = min(min_pair_val / 2.0, min_boundary_val)
    
    norm_grad = np.linalg.norm(grad)
    if norm_grad < 1e-12:
        return centres, r
    
    # Adaptive step sizing
    max_move = 0.01
    scaled_step = min(step_size, max_move / norm_grad)
    
    new_centres = centres + scaled_step * grad
    new_centres = np.clip(new_centres, -0.4999, 0.4999)
    
    return new_centres, r


def _active_set_polish(centres, radius, max_iter=200):
    """
    Polish by identifying active constraints and solving a local QP-like problem.
    """
    n = len(centres)
    best_centres = centres.copy()
    best_radius = radius
    
    for it in range(max_iter):
        # Identify active constraints
        d_boundary = 0.5 - np.abs(best_centres)
        diff = best_centres[:, np.newaxis, :] - best_centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum constraint
        min_boundary_idx = np.argmin(d_boundary)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        # Try to push the most constrained point away
        # Boundary constraint
        i, k = divmod(min_boundary_idx, 3)
        direction = -np.sign(best_centres[i, k])
        move = 0.001 * direction
        new_c = best_centres.copy()
        new_c[i, k] += move
        new_c = np.clip(new_c, -0.4999, 0.4999)
        new_r = _compute_radius(new_c)
        if new_r > best_radius + 1e-12:
            best_centres = new_c
            best_radius = new_r
            continue
        
        # Pairwise constraint
        i, j = min_pair_idx
        if i == j:
            continue
        direction = (best_centres[i] - best_centres[j])
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        direction = direction / norm
        move = 0.001 * direction
        new_c = best_centres.copy()
        new_c[i] += move
        new_c[j] -= move
        new_c = np.clip(new_c, -0.4999, 0.4999)
        new_r = _compute_radius(new_c)
        if new_r > best_radius + 1e-12:
            best_centres = new_c
            best_radius = new_r
            continue
        
        # No improvement, try small random perturbations
        for _ in range(10):
            perturb = 0.002 * np.random.randn(*best_centres.shape)
            new_c = np.clip(best_centres + perturb, -0.4999, 0.4999)
            new_r = _compute_radius(new_c)
            if new_r > best_radius + 1e-12:
                best_centres = new_c
                best_radius = new_r
                break
    
    return best_centres, best_radius


def _multi_start_refine(centres, best_centres, best_radius, num_starts=5, time_budget=20.0):
    """
    Try multiple random restarts with local refinement.
    """
    start_time = time.time()
    
    for s in range(num_starts):
        if time.time() - start_time > time_budget:
            break
        
        # Create perturbed version
        perturbation = 0.03 * np.random.randn(*centres.shape)
        new_centres = np.clip(centres + perturbation, -0.4999, 0.4999)
        
        # Local optimization
        temp_centres = new_centres.copy()
        temp_radius = _compute_radius(temp_centres)
        
        for step in range(100):
            if time.time() - start_time > time_budget:
                break
            temp_centres, _ = _gradient_step(temp_centres, 0.005, beta=150.0)
            temp_radius = _compute_radius(temp_centres)
        
        # Active set polish
        temp_centres, temp_radius = _active_set_polish(temp_centres, temp_radius, max_iter=50)
        
        if temp_radius > best_radius + 1e-10:
            best_radius = temp_radius
            best_centres = temp_centres.copy()
    
    return best_centres, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Initialize with FCC lattice
    centres = _fcc_lattice_init()
    
    if centres.shape != (N, 3):
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Phase 1: Coarse optimization
    step_size = 0.05
    beta = 50.0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 60.0:
        new_centres, r = _gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            step_size *= 0.999
            beta *= 1.005
        else:
            centres = new_centres
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-5:
            break
    
    # Phase 2: Fine optimization
    step_size = 0.01
    beta = 200.0
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline - 40.0:
        new_centres, r = _gradient_step(centres, step_size, beta=beta)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            centres = new_centres
            no_improve_count = 0
            step_size *= 0.998
            beta *= 1.002
        else:
            centres = new_centres
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        if step_size < 1e-6:
            break
    
    # Phase 3: Multi-start refinement
    best_centres, best_radius = _multi_start_refine(
        best_centres, best_centres, best_radius,
        num_starts=8,
        time_budget=30.0
    )
    
    # Phase 4: Final high-precision polish
    step_size = 0.003
    beta_final = 1000.0
    no_improve_count = 0
    while time.time() < deadline - 5.0:
        new_centres, r = _gradient_step(best_centres, step_size, beta=beta_final)
        new_r = _compute_radius(new_centres)
        
        if new_r > best_radius + 1e-12:
            best_radius = new_r
            best_centres = new_centres.copy()
            no_improve_count = 0
            step_size *= 0.99
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                break
    
    # Final active set polish
    best_centres, best_radius = _active_set_polish(best_centres, best_radius, max_iter=100)
    
    # Final validation
    best_centres = np.clip(best_centres, -0.499999, 0.499999)
    final_radius = _compute_radius(best_centres)
    if final_radius > best_radius:
        best_radius = final_radius
    
    return best_centres
# EVOLVE-BLOCK-END
