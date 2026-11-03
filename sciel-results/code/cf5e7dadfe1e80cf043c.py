import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit


def _true_score(centers):
    """Compute the true radius for a given configuration (vectorized)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    
    # Boundary constraint
    boundary = (0.5 - np.abs(centers)).min()
    
    # Pairwise distances - vectorized
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    
    r = min(boundary, min_dist / 2.0)
    return r


def _soft_min(values, beta=50.0):
    """Differentiable approximation of the minimum."""
    # log-sum-exp trick for numerical stability
    scaled = -values * beta
    max_scaled = scaled.max()
    if not np.isfinite(max_scaled):
        return 0.0
    scaled = scaled - max_scaled
    log_sum = np.log(np.sum(np.exp(scaled)))
    return -(max_scaled + log_sum) / beta


def _soft_radius(centers, beta=50.0):
    """Differentiable approximation of the packing radius."""
    if centers.shape != (N, 2):
        return 0.0
    
    # Boundary constraint (soft)
    boundary = (0.5 - np.abs(centers))
    soft_boundary = _soft_min(boundary, beta)
    
    # Pairwise distances (soft)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    soft_min_dist = _soft_min(dists, beta)
    
    r = min(soft_boundary, soft_min_dist / 2.0)
    return r


def _negative_soft_radius(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft radius."""
    centers = x.reshape(N, 2)
    return -_soft_radius(centers, beta)


def _init_grid():
    """Create a 7x7 grid initialization for 50 circles with one extra point."""
    m = 7
    grid_pts = []
    for i in range(m):
        for j in range(m):
            x = (i + 0.5) / m - 0.5
            y = (j + 0.5) / m - 0.5
            grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    
    # Add 50th point near center with small offset
    np.random.seed(42)
    extra_pt = np.array([[0.03, 0.03]])
    all_pts = np.vstack([grid_pts, extra_pt])
    return all_pts


def _init_hexagonal():
    """Create a hexagonal lattice initialization for 50 circles."""
    # Hexagonal lattice: rows offset by half spacing
    # For 50 circles, try different arrangements
    # 7 rows with varying counts: 7,8,7,8,7,8,5 = 50
    points = []
    s = 1.0 / 7.0  # spacing
    
    row_counts = [7, 8, 7, 8, 7, 8, 5]
    y_positions = np.linspace(-0.5 + s/2, 0.5 - s/2, len(row_counts))
    
    for idx, (count, y) in enumerate(zip(row_counts, y_positions)):
        offset = (idx % 2) * s / 2
        x_start = -0.5 + s/2 + offset
        for j in range(count):
            x = x_start + j * s
            if -0.5 <= x <= 0.5:
                points.append([x, y])
    
    points = np.array(points[:N])
    if len(points) < N:
        # Fill remaining with random points
        np.random.seed(42)
        while len(points) < N:
            points = np.vstack([points, np.random.uniform(-0.4, 0.4, size=(1, 2))])
    return points[:N]


def _init_delaunay_based():
    """Initialize using a perturbed hexagonal pattern."""
    # Start with hexagonal and add small random perturbations
    pts = _init_hexagonal()
    np.random.seed(42)
    perturb = np.random.uniform(-0.01, 0.01, size=pts.shape)
    pts = pts + perturb
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def _init_random():
    """Random initialization."""
    np.random.seed(42)
    return np.random.uniform(-0.45, 0.45, size=(N, 2))


def _force_relax(centers, n_steps=100, step_size=0.01, time_budget=5.0):
    """Simple force-directed relaxation to resolve overlaps."""
    start_time = time.time()
    centers = centers.copy()
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 0.01:
                    diff = np.array([0.01, 0.0])
                    dist = 0.01
                # Repulsive force if too close
                if dist < 0.15:
                    force_mag = (0.15 - dist) / dist
                    forces[i] += diff * force_mag
                    forces[j] -= diff * force_mag
        
        # Boundary repulsion
        for i in range(N):
            for k in range(2):
                if centers[i, k] > 0.45:
                    forces[i, k] -= (centers[i, k] - 0.45) * 10
                elif centers[i, k] < -0.45:
                    forces[i, k] += (-0.45 - centers[i, k]) * 10
        
        centers += step_size * forces
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decrease step size
        step_size *= 0.95
    
    return centers


def _optimize_lbfgsb(centers, time_budget=20.0, beta=50.0):
    """Optimize using L-BFGS-B with soft radius objective."""
    start_time = time.time()
    x0 = centers.flatten()
    
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    try:
        bounds = [(-0.5, 0.5)] * (2 * N)
        res = minimize(_negative_soft_radius, x0, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50, 'maxfun': 10000})
        
        new_centers = res.x.reshape(N, 2)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        new_score = _true_score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return best_centers, best_score, elapsed


def _optimize_sa(centers, time_budget=20.0, n_iter=5000):
    """Simulated annealing optimization."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    current_centers = centers.copy()
    current_score = best_score
    
    T = 0.01  # Initial temperature
    cooling = 0.999
    
    for it in range(n_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Random perturbation
        idx = np.random.randint(N)
        perturbation = np.random.normal(0, T, size=2)
        new_centers = current_centers.copy()
        new_centers[idx] += perturbation
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_score = _true_score(new_centers)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        
        T *= cooling
        if T < 1e-6:
            break
    
    return best_centers, best_score


def _local_refine(centers, time_budget=15.0):
    """Local refinement with coordinate descent and small perturbations."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    step_sizes = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break
            
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for i in range(N):
                # Try moving in each direction
                for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                               (-step, -step), (-step, step), (step, -step), (step, step)]:
                    new_c = best_centers.copy()
                    new_c[i, 0] += dx
                    new_c[i, 1] += dy
                    if np.any(np.abs(new_c) > 0.5):
                        continue
                    s = _true_score(new_c)
                    if s > best_score + 1e-12:
                        best_centers = new_c
                        best_score = s
                        improved = True
            if time.time() - start_time > time_budget:
                break
    
    return best_centers, best_score


def _radius_expansion(centers, time_budget=10.0):
    """Try to expand the minimum radius by moving the most constrained points."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    for _ in range(50):
        if time.time() - start_time > time_budget:
            break
        
        # Find the most constrained pair or boundary
        r = best_score
        boundary = (0.5 - np.abs(best_centers)).min()
        diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        
        if boundary < min_dist / 2:
            # Boundary is limiting
            idx = np.argmin((0.5 - np.abs(best_centers)).min(axis=1))
            # Move point away from boundary
            for axis in range(2):
                if best_centers[idx, axis] > 0:
                    best_centers[idx, axis] -= 0.001
                else:
                    best_centers[idx, axis] += 0.001
        else:
            # Pair is limiting
            min_dist_idx = np.argmin(dists)
            i, j = np.unravel_index(min_dist_idx, (N, N))
            if i > j:
                i, j = j, i
            diff = best_centers[i] - best_centers[j]
            dist = np.linalg.norm(diff)
            if dist > 0:
                direction = diff / dist
                # Move apart slightly
                best_centers[i] += direction * 0.0005
                best_centers[j] -= direction * 0.0005
        
        best_centers = np.clip(best_centers, -0.5, 0.5)
        new_score = _true_score(best_centers)
        if new_score > best_score:
            best_score = new_score
        else:
            # Revert if it didn't help
            # Keep trying different moves
            pass
    
    return best_centers, best_score


def construct():
    """Main entry point."""
    start_time = time.time()
    
    np.random.seed(42)
    
    # Try multiple initializations and keep the best
    initializations = [
        _init_grid(),
        _init_hexagonal(),
        _init_delaunay_based(),
    ]
    
    best_centers = None
    best_score = -1
    
    for init in initializations:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Force relax to resolve initial overlaps
        relaxed = _force_relax(init, n_steps=50, step_size=0.01, time_budget=3.0)
        
        # L-BFGS-B optimization
        elapsed_so_far = time.time() - start_time
        remaining = DEADLINE - elapsed_so_far
        if remaining > 20:
            t_lbfgs = min(remaining * 0.3, 20.0)
            centers_lbfgs, score_lbfgs, _ = _optimize_lbfgsb(relaxed, time_budget=t_lbfgs)
            
            # Local refinement
            elapsed_so_far = time.time() - start_time
            remaining = DEADLINE - elapsed_so_far
            if remaining > 10:
                t_lr = min(remaining * 0.2, 10.0)
                centers_lr, score_lr = _local_refine(centers_lbfgs, time_budget=t_lr)
                if score_lr > score_lbfgs:
                    centers_lbfgs = centers_lr
                    score_lbfgs = score_lr
            
            # Radius expansion
            elapsed_so_far = time.time() - start_time
            remaining = DEADLINE - elapsed_so_far
            if remaining > 5:
                t_re = min(remaining * 0.1, 5.0)
                centers_re, score_re = _radius_expansion(centers_lbfgs, time_budget=t_re)
                if score_re > score_lbfgs:
                    centers_lbfgs = centers_re
                    score_lbfgs = score_re
            
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_centers = centers_lbfgs.copy()
    
    # If we have time, do simulated annealing from the best solution
    if best_centers is not None:
        elapsed_so_far = time.time() - start_time
        remaining = DEADLINE - elapsed_so_far
        if remaining > 15:
            t_sa = min(remaining * 0.4, 15.0)
            centers_sa, score_sa = _optimize_sa(best_centers, time_budget=t_sa)
            if score_sa > best_score:
                best_score = score_sa
                best_centers = centers_sa.copy()
    
    # Final local refinement
    if best_centers is not None:
        elapsed_so_far = time.time() - start_time
        remaining = DEADLINE - elapsed_so_far
        if remaining > 5:
            t_final = min(remaining * 0.5, 10.0)
            centers_final, score_final = _local_refine(best_centers, time_budget=t_final)
            if score_final > best_score:
                best_score = score_final
                best_centers = centers_final.copy()
    
    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid()
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final check
    final_score = _true_score(best_centers)
    if final_score <= 0:
        # Fallback to a known valid configuration
        best_centers = _init_grid()
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers


if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
