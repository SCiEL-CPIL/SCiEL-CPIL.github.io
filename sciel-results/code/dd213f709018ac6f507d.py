"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    """
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq)) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj(centers, r, beta=50.0):
    """
    Smooth surrogate: soft-min of (pairwise distances/2, wall distances) - r.
    Lower is better. Negative means feasible.
    Uses log-sum-exp for numerical stability.
    """
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq) / 2.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    
    # Collect all constraints: pairwise (i<j) and wall distances
    # For efficiency, use the full matrix but ignore diagonal
    all_dists = np.concatenate([
        dists[np.triu_indices(n, k=1)],
        wall_dists
    ])
    
    # Soft min: -log(sum(exp(-beta * d))) / beta
    # = - (1/beta) * log(sum(exp(-beta * d)))
    # For numerical stability:
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(np.sum(exp_terms)) / beta
    
    return soft_min - r

def _soft_min_grad(centers, r, beta=50.0):
    """
    Gradient of the soft-min surrogate objective.
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq) / 2.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    
    all_dists = np.concatenate([
        dists[np.triu_indices(n, k=1)],
        wall_dists
    ])
    
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    weights = exp_terms / np.sum(exp_terms)
    
    # Pairwise gradients
    iu, ju = np.triu_indices(n, k=1)
    for idx in range(len(iu)):
        i, j = iu[idx], ju[idx]
        d = dists[i, j]
        if d > 1e-12:
            w = weights[idx]
            # d = |c_i - c_j|/2, grad d/dc_i = (c_i - c_j)/(2*d)
            factor = w / (2.0 * d)
            grad[i] += factor * (centers[i] - centers[j])
            grad[j] -= factor * (centers[i] - centers[j])
    
    # Wall gradients
    for idx in range(n):
        w = weights[len(iu) + idx]
        if wall_dists_x[idx] <= wall_dists_y[idx]:
            if centers[idx, 0] >= 0:
                grad[idx, 0] -= w
            else:
                grad[idx, 0] += w
        else:
            if centers[idx, 1] >= 0:
                grad[idx, 1] -= w
            else:
                grad[idx, 1] += w
    
    return grad

def _soft_min_polish(centers, max_iters=200, time_limit=15.0, beta_start=20.0, beta_end=200.0):
    """
    Polish using soft-min surrogate with increasing beta.
    Uses line search for more reliable convergence.
    """
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    r_est = _compute_radius(centers)
    lr = 0.01
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Anneal beta
        beta = beta_start * (beta_end / beta_start) ** (it / max_iters)
        
        obj = _soft_min_obj(centers, r_est, beta)
        if obj < -1e-10:
            # Already feasible, try to increase r slightly
            r_est *= 1.001
            continue
        
        grad = _soft_min_grad(centers, r_est, beta)
        
        # Line search: try a few step sizes and pick the best
        best_obj = obj
        best_centers = centers
        for trial_lr in [lr, lr * 0.5, lr * 0.25, lr * 2.0]:
            trial_centers = centers - trial_lr * grad
            trial_centers = np.clip(trial_centers, -0.4999, 0.4999)
            trial_obj = _soft_min_obj(trial_centers, r_est, beta)
            if trial_obj < best_obj:
                best_obj = trial_obj
                best_centers = trial_centers
        
        centers = best_centers
        obj = best_obj
        
        # Adapt learning rate based on progress
        if obj < -1e-10:
            r_est *= 1.0005
        else:
            # Scale lr based on gradient magnitude
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                lr = max(0.001, min(0.05, lr * (1.0 if obj < 0 else 0.8)))
    
    return centers

def _violation(centers, r):
    """
    Compute total violation for radius r.
    Violation = sum of max(0, r - d/2) for pairs + sum of max(0, r - wall_dist) for walls.
    Lower is better. Zero means feasible.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_violation = np.sum(np.maximum(0, r - dists / 2.0))
    
    # Wall violations
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    wall_violation = np.sum(np.maximum(0, r - wall_dists))
    
    return pair_violation + wall_violation

def _grad_violation(centers, r):
    """
    Compute gradient of violation with respect to centers.
    """
    centers = np.asarray(centers)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # For each pair (i,j) where dist < 2*r, add force pushing them apart
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            if d < 2.0 * r and d > 1e-12:
                # Gradient of max(0, r - d/2) w.r.t. centers
                # d = |c_i - c_j|, so grad_d/dc_i = (c_i - c_j)/d
                # grad_violation = -0.5 * (c_i - c_j)/d for c_i, +0.5 * (c_i - c_j)/d for c_j
                factor = -0.5 / d
                grad[i] += factor * (centers[i] - centers[j])
                grad[j] -= factor * (centers[i] - centers[j])
    
    # Wall gradient
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_dists_x, wall_dists_y)
    
    for i in range(n):
        if wall_dists[i] < r:
            # Push away from nearest wall
            if wall_dists_x[i] <= wall_dists_y[i]:
                # Nearest wall is x-direction
                if centers[i, 0] >= 0:
                    grad[i, 0] -= 1.0  # Push left
                else:
                    grad[i, 0] += 1.0  # Push right
            else:
                # Nearest wall is y-direction
                if centers[i, 1] >= 0:
                    grad[i, 1] -= 1.0  # Push down
                else:
                    grad[i, 1] += 1.0  # Push up
    
    return grad

def _force_directed_layout(centers, max_iters=5000, time_limit=30.0):
    """
    Force-directed layout using repulsive forces between circles
    and attractive forces toward the square center.
    """
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    # Estimate initial radius
    r_est = _compute_radius(centers)
    
    # Force parameters
    repulsion_strength = 1.0
    attraction_strength = 0.1
    damping = 0.9
    max_step = 0.05
    
    velocities = np.zeros_like(centers)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        forces = np.zeros_like(centers)
        
        # Repulsive forces between circles
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    # Random small perturbation for coincident points
                    diff = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(diff)
                
                # Force inversely proportional to distance^3
                force_mag = repulsion_strength / (dist ** 3)
                force = force_mag * diff / dist
                forces[i] += force
                forces[j] -= force
        
        # Attractive force toward center
        for i in range(n):
            forces[i] -= attraction_strength * centers[i]
        
        # Wall repulsion: push away from walls if too close
        for i in range(n):
            # Distance to each wall
            dist_left = centers[i, 0] + 0.5
            dist_right = 0.5 - centers[i, 0]
            dist_bottom = centers[i, 1] + 0.5
            dist_top = 0.5 - centers[i, 1]
            
            min_wall_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            if min_wall_dist < 0.1:
                # Push away from nearest wall
                if min_wall_dist == dist_left:
                    forces[i, 0] += repulsion_strength / (min_wall_dist ** 2 + 1e-6)
                elif min_wall_dist == dist_right:
                    forces[i, 0] -= repulsion_strength / (min_wall_dist ** 2 + 1e-6)
                elif min_wall_dist == dist_bottom:
                    forces[i, 1] += repulsion_strength / (min_wall_dist ** 2 + 1e-6)
                else:
                    forces[i, 1] -= repulsion_strength / (min_wall_dist ** 2 + 1e-6)
        
        # Update velocities with damping
        velocities = damping * velocities + 0.1 * forces
        
        # Limit step size
        for i in range(n):
            step = np.linalg.norm(velocities[i])
            if step > max_step:
                velocities[i] *= max_step / step
        
        # Update positions
        centers += velocities
        
        # Clip to square with small margin
        centers = np.clip(centers, -0.499, 0.499)
        
        # Every 100 iterations, re-estimate radius and adjust forces
        if it % 100 == 0:
            r_est = _compute_radius(centers)
            # Scale repulsion to be meaningful relative to radius
            repulsion_strength = max(0.01, r_est * 10)
    
    return centers

def _gradient_polish(centers, max_iters=200, time_limit=10.0):
    """
    Gradient-based polish using violation minimization.
    """
    start_time = time.time()
    centers = centers.copy()
    n = len(centers)
    
    # Binary search for maximum feasible radius
    r_low = _compute_radius(centers)
    r_high = 0.1  # Upper bound guess
    
    # First, find a feasible radius via binary search
    for _ in range(30):
        if time.time() - start_time > time_limit:
            break
        r_mid = (r_low + r_high) / 2
        v = _violation(centers, r_mid)
        if v < 1e-10:
            r_low = r_mid
        else:
            r_high = r_mid
    
    r_target = r_low
    
    # Now polish to minimize violation at r_target
    lr = 0.01
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        v = _violation(centers, r_target)
        if v < 1e-12:
            break
        
        grad = _grad_violation(centers, r_target)
        centers -= lr * grad
        centers = np.clip(centers, -0.499, 0.499)
        
        # Adapt learning rate
        v_new = _violation(centers, r_target)
        if v_new > v:
            lr *= 0.5
            centers = centers + lr * grad  # Undo step
        else:
            lr = min(lr * 1.1, 0.05)
    
    return centers

def _sa_search(centers, time_limit=30.0, rng=None):
    """
    Simulated Annealing to maximize radius.
    Uses adaptive step size, occasional multi-point moves, and
    periodic gradient polish to escape local optima.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    start_time = time.time()
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    T_start = 0.03
    T_end = 1e-8
    max_steps = 200000
    polish_every = 5000
    
    # Adaptive step size
    base_step = 0.015
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Occasionally move multiple points
        if rng.random() < 0.15:
            n_move = rng.integers(2, 6)
            idxs = rng.choice(N, size=n_move, replace=False)
            step_size = base_step * (1.0 + 0.5 * np.cos(np.pi * step / max_steps))
            new_centers = current_centers.copy()
            for idx in idxs:
                delta = rng.normal(0, step_size, 2)
                new_centers[idx] += delta
            new_centers = np.clip(new_centers, -0.499, 0.499)
        else:
            idx = rng.integers(0, N)
            step_size = base_step * (1.0 + 0.5 * np.cos(np.pi * step / max_steps))
            delta = rng.normal(0, step_size, 2)
            new_centers = current_centers.copy()
            new_centers[idx] += delta
            new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = new_centers.copy()
        else:
            delta_r = new_radius - current_radius
            if delta_r > -5 * T:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current_centers = new_centers
                    current_radius = new_radius
        
        # Periodic gradient polish to refine current position
        if step > 0 and step % polish_every == 0:
            remaining = time_limit - (time.time() - start_time)
            if remaining > 2.0:
                polished = _soft_min_polish(current_centers, max_iters=50, time_limit=min(2.0, remaining - 1.0), beta_start=100.0, beta_end=300.0)
                pol_radius = _compute_radius(polished)
                if pol_radius > current_radius:
                    current_centers = polished
                    current_radius = pol_radius
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current_centers.copy()
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    best_centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
    best_centers = np.clip(best_centers, -0.49, 0.49)
    
    best_radius = _compute_radius(best_centers)
    
    # Phase 1: Force-directed layout (25s)
    elapsed = time.time() - start_time
    if elapsed < deadline - 55:
        fd_centers = _force_directed_layout(best_centers, max_iters=5000, time_limit=25.0)
        fd_radius = _compute_radius(fd_centers)
        if fd_radius > best_radius:
            best_centers = fd_centers
            best_radius = fd_radius
    
    # Phase 2: Gradient polish (8s)
    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        polished = _gradient_polish(best_centers, max_iters=200, time_limit=8.0)
        pol_radius = _compute_radius(polished)
        if pol_radius > best_radius:
            best_centers = polished
            best_radius = pol_radius
    
    # Phase 3: Soft-min polish (10s)
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        soft_polished = _soft_min_polish(best_centers, max_iters=300, time_limit=10.0)
        sp_radius = _compute_radius(soft_polished)
        if sp_radius > best_radius:
            best_centers = soft_polished
            best_radius = sp_radius
    
    # Phase 4: Simulated annealing (remaining time, up to 55s)
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        sa_time = min(55.0, deadline - elapsed - 5)
        sa_centers, sa_radius = _sa_search(best_centers, time_limit=sa_time, rng=rng)
        if sa_radius > best_radius:
            best_centers = sa_centers
            best_radius = sa_radius
    
    # Phase 5: Final high-precision polish with higher beta for tighter convergence
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        final_polish = _soft_min_polish(best_centers, max_iters=300, time_limit=min(10.0, deadline - elapsed - 2), beta_start=200.0, beta_end=1000.0)
        fp_radius = _compute_radius(final_polish)
        if fp_radius > best_radius:
            best_centers = final_polish
            best_radius = fp_radius
    
    # Final check
    if not np.all(np.isfinite(best_centers)):
        return grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
