import numpy as np
from scipy.spatial.distance import pdist
import time
import scipy.optimize as so

def _compute_score(points):
    """Compute R = (d_min / d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _compute_dists_matrix(points):
    """Compute pairwise distance matrix efficiently."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists

def _sqp_maximize_ratio(points, max_iter=200, tol=1e-9):
    """
    Directly maximize R = (d_min/d_max)^2 using SLSQP.
    Variables: 2n point coordinates + 1 auxiliary r (min dist) + 1 auxiliary M (max dist)
    Objective: maximize r/M (equivalently minimize -r/M)
    Constraints: d_ij >= r for all pairs, d_ij <= M for all pairs, r > 0, M > 0
    """
    n = len(points)
    m = n * (n - 1) // 2
    idx_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    x0 = points.flatten()
    d0 = pdist(points)
    r0 = max(d0.min() * 0.99, 1e-6)
    M0 = d0.max() * 1.01
    v0 = np.append(x0, [r0, M0])
    
    def objective(v):
        r = v[2*n]
        M = v[2*n + 1]
        if M < 1e-12:
            return 1e6
        return -(r / M)
    
    def jac(v):
        grad = np.zeros(2*n + 2)
        r = v[2*n]
        M = v[2*n + 1]
        if M > 1e-12:
            grad[2*n] = -1.0 / M
            grad[2*n + 1] = r / (M * M)
        return grad
    
    constraints = []
    
    # Constraints: d_ij >= r for all pairs
    for k, (i, j) in enumerate(idx_pairs):
        def min_dist_con(v, i=i, j=j):
            pts = v[:2*n].reshape(n, 2)
            d = np.sqrt(np.sum((pts[i] - pts[j])**2))
            return d - v[2*n]
        constraints.append({'type': 'ineq', 'fun': min_dist_con})
    
    # Constraints: d_ij <= M for all pairs
    for k, (i, j) in enumerate(idx_pairs):
        def max_dist_con(v, i=i, j=j):
            pts = v[:2*n].reshape(n, 2)
            d = np.sqrt(np.sum((pts[i] - pts[j])**2))
            return v[2*n + 1] - d
        constraints.append({'type': 'ineq', 'fun': max_dist_con})
    
    bounds = [(-10, 10)] * (2*n) + [(1e-8, 10), (1e-8, 10)]
    
    try:
        res = so.minimize(objective, v0, method='SLSQP', jac=jac, bounds=bounds,
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def _sqp_smooth_ratio(points, beta=100.0, max_iter=200, tol=1e-9):
    """
    Smooth version using log-sum-exp for min/max distances.
    Maximizes r/M where r ≈ soft_min(d), M ≈ soft_max(d).
    """
    n = len(points)
    m = n * (n - 1) // 2
    
    def soft_min(x, beta):
        x_min = np.min(x)
        return x_min - np.log(np.mean(np.exp(-beta * (x - x_min)))) / beta
    
    def soft_max(x, beta):
        x_max = np.max(x)
        return x_max + np.log(np.mean(np.exp(beta * (x - x_max)))) / beta
    
    x0 = points.flatten()
    d0 = pdist(points)
    r0 = max(d0.min() * 0.99, 1e-6)
    M0 = d0.max() * 1.01
    v0 = np.append(x0, [r0, M0])
    
    def objective(v):
        pts = v[:2*n].reshape(n, 2)
        d = pdist(pts)
        r = v[2*n]
        M = v[2*n + 1]
        if M < 1e-12:
            return 1e6
        return -(r / M)
    
    def jac(v):
        grad = np.zeros(2*n + 2)
        r = v[2*n]
        M = v[2*n + 1]
        if M > 1e-12:
            grad[2*n] = -1.0 / M
            grad[2*n + 1] = r / (M * M)
        return grad
    
    constraints = []
    
    # Soft constraints: soft_min(d) >= r - epsilon
    def soft_min_con(v):
        pts = v[:2*n].reshape(n, 2)
        d = pdist(pts)
        r = v[2*n]
        sm = soft_min(d, beta)
        return sm - r + 0.01  # small slack
    
    constraints.append({'type': 'ineq', 'fun': soft_min_con})
    
    # Soft constraints: soft_max(d) <= M + epsilon
    def soft_max_con(v):
        pts = v[:2*n].reshape(n, 2)
        d = pdist(pts)
        M = v[2*n + 1]
        smax = soft_max(d, beta)
        return M - smax + 0.01  # small slack
    
    constraints.append({'type': 'ineq', 'fun': soft_max_con})
    
    bounds = [(-10, 10)] * (2*n) + [(1e-8, 10), (1e-8, 10)]
    
    try:
        res = so.minimize(objective, v0, method='SLSQP', jac=jac, bounds=bounds,
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def _fast_local_search(points, time_budget=50.0, seed=42):
    """
    Fast local search using precomputed distance matrix updates.
    Much faster than simulated annealing with full pdist recomputation.
    """
    n = len(points)
    rng = np.random.RandomState(seed)
    
    # Normalize
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        current = points / d_max
    else:
        current = points.copy()
    
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 0.02
    alpha = 0.9995
    step_size = 0.03
    iterations = 0
    
    # Precompute pair indices for efficient distance updates
    pair_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pair_idx)
    
    while time.time() - start_time < time_budget:
        if T < 1e-10:
            break
        
        # Pick a point to move
        idx = rng.randint(0, n)
        delta = rng.randn(2) * step_size
        
        new_pt = current[idx].copy()
        new_pt += delta
        
        # Compute new distances involving point idx
        # Only need to recompute distances from idx to all other points
        other_pts = np.delete(current, idx, axis=0)
        new_dists = np.sqrt(np.sum((other_pts - new_pt)**2, axis=1))
        
        # Get old distances from idx
        old_dists = np.sqrt(np.sum((other_pts - current[idx])**2, axis=1))
        
        # Find which pairs are affected
        # Pairs involving idx: (idx, j) for j != idx
        # We need to update the global distance set
        
        # Quick score estimate: just check if min/max changed significantly
        # For speed, compute full score every 10th iteration, estimate otherwise
        if iterations % 5 == 0:
            new_pts = current.copy()
            new_pts[idx] = new_pt
            new_score = _compute_score(new_pts)
        else:
            # Fast estimate: use old min/max and check if new point creates new extremes
            d_min_old = np.min(old_dists)
            d_max_old = np.max(old_dists)
            d_min_new = np.min(new_dists)
            d_max_new = np.max(new_dists)
            
            # Rough estimate assuming other distances unchanged
            # This is approximate but fast
            new_score = current_score * (1 + 0.5 * (d_min_new - d_min_old) / max(d_min_old, 1e-12))
        
        delta_score = new_score - current_score
        
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            current[idx] = new_pt
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        T *= alpha
        step_size *= 0.9999
        iterations += 1
    
    return best, best_score

def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)
    
    # 7. Random with larger spread
    candidates.append(rng.randn(n, 2) * 2.0)
    
    # 8. Fibonacci sphere projection to 2D (golden angle spiral)
    angles = 2 * np.pi * np.arange(n) * 0.6180339887
    radii = np.sqrt(np.arange(n) / n)
    fib = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(fib)
    
    # 9. 3x3 grid + 7 extra points
    xs3 = np.linspace(-1, 1, 3)
    ys3 = np.linspace(-1, 1, 3)
    grid3 = np.column_stack(np.meshgrid(xs3, ys3, indexing='ij')).reshape(-1, 2)
    extra_angles = 2 * np.pi * np.arange(7) / 7
    extra = 1.3 * np.column_stack([np.cos(extra_angles), np.sin(extra_angles)])
    candidates.append(np.vstack([grid3, extra]))

    return candidates

def _polish(points, time_budget=30.0):
    """High-precision polish pass with tighter tolerances and more iterations."""
    n = len(points)
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Multiple rounds with different methods and decreasing tolerance
    for tol in [1e-7, 1e-9, 1e-11]:
        if time.time() - start_time > time_budget:
            break
        
        # Direct ratio SQP
        result = _sqp_maximize_ratio(best_points, max_iter=150, tol=tol)
        if result is not None and len(result) == 2*n + 2:
            new_points = result[:2*n].reshape(n, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        # Smooth ratio SQP
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3.0:
            result_smooth = _sqp_smooth_ratio(best_points, beta=50.0, max_iter=100, tol=tol)
            if result_smooth is not None and len(result_smooth) == 2*n + 2:
                new_points = result_smooth[:2*n].reshape(n, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
    
    # Final fast local search polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5.0:
        ls_points, ls_score = _fast_local_search(best_points, time_budget=min(remaining - 2.0, 20.0), seed=777)
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_points.copy()
    
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 610.0  # Safety margin before 630s kill
    
    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)
    
    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)
    
    # Phase 1: Direct ratio SQP from each initial configuration
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 60.0:
            break
        
        time_budget = min(20.0, deadline - time.time() - 30.0)
        if time_budget < 5.0:
            break
        
        try:
            result = _sqp_maximize_ratio(init_pts, max_iter=100, tol=1e-7)
            if result is not None and len(result) == 2*n + 2:
                new_points = result[:2*n].reshape(n, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_points = new_points.copy()
                    best_score = new_score
        except Exception as e:
            pass

    # Phase 2: Fast local search from the best point found
    remaining = deadline - time.time() - 10.0
    if remaining > 20.0:
        try:
            ls_points, ls_score = _fast_local_search(best_points, time_budget=min(remaining - 10.0, 80.0), seed=123)
            if ls_score > best_score:
                best_points = ls_points.copy()
                best_score = ls_score
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        try:
            polished_points, polished_score = _polish(best_points, time_budget=min(remaining - 5.0, 60.0))
            if polished_score > best_score:
                best_points = polished_points.copy()
                best_score = polished_score
        except Exception:
            pass

    return best_points
