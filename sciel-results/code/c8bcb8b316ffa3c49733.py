import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def compute_radius(points):
    """
    Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Use cdist for efficient pairwise distance calculation
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def _softmin_logsumexp(values, beta):
    """
    Soft-min via log-sum-exp for numerical stability.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    # Clip to avoid overflow in exp
    neg_clipped = np.clip(neg - neg_max, -500, 500)
    exp_sum = np.sum(np.exp(neg_clipped))
    if exp_sum <= 0:
        return np.min(values)
    return -(neg_max + np.log(exp_sum)) / beta

def _smooth_objective_and_grad(x, beta=50.0):
    """
    Smooth objective with analytical gradient for L-BFGS-B.
    Objective: -soft_min(all_margins)
    Margins: 
    - Boundary: f_i = 1 - |c_i|
    - Pairwise: g_ij = |c_i - c_j| / 2
    """
    pts = x.reshape(N, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise margins
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    pair_half = dists / 2.0
    
    # Combine all margins into a single vector
    # Boundary: N values
    # Pairwise: N*N values (including inf on diagonal, which will have 0 weight)
    all_vals = np.concatenate([boundary.ravel(), pair_half.ravel()])
    
    # Filter out non-finite values (diagonal is inf)
    finite_mask = np.isfinite(all_vals)
    all_vals = all_vals[finite_mask]
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft-min computation
    neg = -all_vals * beta
    neg_max = np.max(neg)
    neg_clipped = np.clip(neg - neg_max, -500, 500)
    exp_vals = np.exp(neg_clipped)
    exp_sum = np.sum(exp_vals)
    
    if exp_sum <= 0:
        soft_min = np.min(all_vals)
        # In this degenerate case, gradient is not well-defined, return 0
        return -soft_min, np.zeros_like(x)
    
    soft_min = -(neg_max + np.log(exp_sum)) / beta
    weights = exp_vals / exp_sum
    
    # Gradient calculation
    # d(soft_min)/dx = sum_k weights[k] * d(margin_k)/dx
    # d(-soft_min)/dx = - sum_k weights[k] * d(margin_k)/dx
    
    grad = np.zeros_like(x)
    
    # We need to map weights back to the original indices
    # The 'all_vals' vector was formed by concatenating boundary and pair_half
    # and then filtering finite values.
    # It's easier to work with the unfiltered arrays and apply the finite mask logic.
    
    # Let's re-calculate weights on the full arrays to avoid index mapping errors
    # Boundary weights
    neg_b = -boundary * beta
    neg_b_max = np.max(neg_b)
    neg_b_clipped = np.clip(neg_b - neg_b_max, -500, 500)
    exp_b = np.exp(neg_b_clipped)
    
    # Pairwise weights
    # Replace inf with a large negative number in the exponent to make weight 0
    # Or just handle the diagonal separately.
    # dists has inf on diagonal. 
    # We want weights for all finite margins.
    
    # A simpler approach: calculate the gradient contribution from each type separately
    # using the global normalization factor exp_sum.
    
    # Note: exp_sum calculated above is for the concatenated finite values.
    # We must be careful. The weights 'weights' correspond to 'all_vals' which is filtered.
    # Let's reconstruct the weights for the full arrays.
    
    # Global normalization
    # log_sum_exp of all -beta * margins
    # We already have exp_sum for the finite part.
    
    # Let's just iterate over the finite values and accumulate gradients.
    # This is O(N^2) which is fine for N=50.
    
    # Indices in all_vals:
    # 0 to N-1: boundary
    # N to N*N-1: pairwise
    
    # We need to know which indices in the original arrays are finite.
    # Boundary: all finite if pts are finite.
    # Pairwise: diagonal is inf.
    
    # Let's create a weight array for the full concatenated vector before filtering.
    full_neg = -np.concatenate([boundary.ravel(), pair_half.ravel()]) * beta
    # Handle inf: exp(-beta * inf) = 0.
    # np.exp(-inf) is 0. So we can just compute exp on the full vector, 
    # but we must avoid overflow.
    
    # Shift by max for stability
    full_neg_max = np.max(full_neg)
    full_neg_clipped = np.clip(full_neg - full_neg_max, -500, 500)
    full_exp = np.exp(full_neg_clipped)
    full_exp_sum = np.sum(full_exp)
    
    if full_exp_sum <= 0:
        return -soft_min, np.zeros_like(x)
        
    full_weights = full_exp / full_exp_sum
    
    # Boundary weights: full_weights[0:N]
    w_b = full_weights[:N]
    
    # Pairwise weights: full_weights[N:N+N*N]
    w_p = full_weights[N:].reshape(N, N)
    
    # Gradient for boundary:
    # d(1 - |c_i|)/d(c_i) = -c_i / |c_i|
    # Contribution to grad: -w_b[i] * (-c_i / |c_i|) = w_b[i] * c_i / |c_i|
    # Wait, objective is -soft_min.
    # d(-soft_min)/dx = - sum w_k d(margin_k)/dx
    # d(boundary_i)/dx_i = -c_i / |c_i|
    # So term is -w_b[i] * (-c_i / |c_i|) = w_b[i] * c_i / |c_i|
    
    for i in range(N):
        if w_b[i] > 1e-15:
            if norms[i] > 1e-10:
                grad[2*i:2*i+2] += w_b[i] * (pts[i] / norms[i])
    
    # Gradient for pairwise:
    # d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(|c_i - c_j|/2)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # Contribution to grad:
    # -w_p[i,j] * (c_i - c_j) / (2 * dist_ij) for point i
    # -w_p[i,j] * (c_j - c_i) / (2 * dist_ij) for point j
    
    # Vectorized approach for pairwise gradient
    # dists is (N, N)
    # vecs is (N, N, 2) where vecs[i,j] = c_i - c_j
    vecs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    
    # Avoid division by zero on diagonal
    # dists is (N, N)
    # Safe dists
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # Unit vectors
    unit_vecs = vecs / safe_dists[:, :, np.newaxis]
    
    # Gradient contribution from point i to itself due to pair (i, j)
    # sum_j w_p[i,j] * unit_vecs[i,j] / 2
    # Gradient contribution from point j to itself due to pair (i, j)
    # sum_i w_p[i,j] * (-unit_vecs[i,j]) / 2  (since vecs[j,i] = -vecs[i,j])
    
    # grad_pair_i = sum_j w_p[i,j] * unit_vecs[i,j] / 2
    # grad_pair_j = sum_i w_p[i,j] * (-unit_vecs[i,j]) / 2
    
    # Let's accumulate:
    # grad[2*i:2*i+2] -= 0.5 * sum_j (w_p[i,j] * unit_vecs[i,j])
    # grad[2*j:2*j+2] -= 0.5 * sum_i (w_p[i,j] * (-unit_vecs[i,j]))
    #                 = 0.5 * sum_i (w_p[i,j] * unit_vecs[i,j])
    
    # Note: w_p is symmetric? No, weights depend on distance, which is symmetric.
    # w_p[i,j] = w_p[j,i].
    
    # Sum for each i:
    # S_i = sum_j w_p[i,j] * unit_vecs[i,j]
    # This is a matrix multiplication-like operation.
    # S_i = (w_p * unit_vecs)_i ? No, unit_vecs is 3D.
    
    # Let's just loop, it's fast enough for N=50.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = w_p[i, j]
            if w > 1e-15:
                # Contribution to i
                grad[2*i:2*i+2] -= 0.5 * w * unit_vecs[i, j]
                # Contribution to j
                # unit_vecs[j, i] = -unit_vecs[i, j]
                # Term for j is -0.5 * w_p[j, i] * unit_vecs[j, i]
                # = -0.5 * w * (-unit_vecs[i, j]) = 0.5 * w * unit_vecs[i, j]
                grad[2*j:2*j+2] += 0.5 * w * unit_vecs[i, j]
                
    obj = -soft_min
    return obj, grad

def initial_rings():
    """Generate initial points using concentric rings matching known optimal topology."""
    points = np.zeros((N, 2))
    
    # Center point
    points[0] = [0, 0]
    count = 1
    
    # Ring structure: 1 center, 6, 12, 18, 13 = 50 total
    # Radii are approximations, will be optimized
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.92]
    ring_counts = [1, 6, 12, 18, 13]
    
    for idx, (r, n) in enumerate(zip(ring_radii, ring_counts)):
        for k in range(n):
            if count >= N:
                break
            angle = 2 * np.pi * k / n
            # Offset alternate rings to interlock
            if idx % 2 == 1:
                angle += np.pi / n
            points[count, 0] = r * np.cos(angle)
            points[count, 1] = r * np.sin(angle)
            count += 1
    
    # Add small random perturbation to break symmetry and avoid local minima
    np.random.seed(42)
    points += np.random.normal(0, 0.01, size=(N, 2))
    
    # Ensure inside unit circle
    norms = np.linalg.norm(points, axis=1)
    for i in range(N):
        if norms[i] > 0.98:
            points[i] *= 0.98 / norms[i]
    
    return points

def _lbfbsb_optimize(x0, time_limit=25.0, beta=50.0, maxiter=200):
    """Run L-BFGS-B optimization on the smooth objective with gradient."""
    start_time = time.time()
    
    def fun_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _smooth_objective_and_grad(x, beta=beta)
    
    try:
        result = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                         options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxfun': 5000, 'maxls': 50})
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x
    except Exception:
        pass
    
    return x0

def _multi_stage_anneal(pts, deadline):
    """
    Multi-stage annealing: start with low beta (smooth landscape) and increase
    to high beta (sharp minimum) for precise convergence.
    """
    current_pts = pts.copy()
    current_score = compute_radius(current_pts)
    
    # Stages: low beta for exploration, high beta for exploitation
    betas = [10.0, 30.0, 80.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        time_left = deadline - time.time() - 8
        if time_left < 3:
            break
        
        x0 = current_pts.flatten()
        optimized = _lbfbsb_optimize(x0, time_limit=min(time_left, 20.0), beta=beta, maxiter=150)
        
        opt_pts = optimized.reshape(N, 2)
        opt_score = compute_radius(opt_pts)
        
        if opt_score > current_score:
            current_pts = opt_pts
            current_score = opt_score
    
    return current_pts, current_score

def _push_apart(points, target_r, iterations=200):
    """
    Iterative constraint projection: push overlapping circles apart and
    boundary-violating circles inward until all constraints are satisfied.
    """
    points = points.copy()
    
    for _ in range(iterations):
        moved = False
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                min_dist = 2 * target_r
                
                if dist < min_dist:
                    if dist > 1e-10:
                        push = (min_dist - dist) / 2.0
                        direction = diff / dist
                        points[i] += direction * push
                        points[j] -= direction * push
                    else:
                        # Coincident points: separate randomly
                        rand_dir = np.random.normal(0, 1, 2)
                        rand_dir /= np.linalg.norm(rand_dir)
                        points[i] += rand_dir * target_r
                        points[j] -= rand_dir * target_r
                    moved = True
        
        # Boundary constraints
        for i in range(N):
            norm = np.linalg.norm(points[i])
            max_norm = 1.0 - target_r
            if norm > max_norm:
                if norm > 1e-10:
                    points[i] = points[i] * max_norm / norm
                else:
                    points[i] = np.array([max_norm, 0.0])
                moved = True
        
        if not moved:
            break
    
    # Final boundary enforcement
    for i in range(N):
        norm = np.linalg.norm(points[i])
        max_norm = 1.0 - target_r
        if norm > max_norm and norm > 1e-10:
            points[i] = points[i] * max_norm / norm
    
    return points

def _binary_search_polish(pts, deadline):
    """
    Binary search on radius with constraint projection to find the maximum
    feasible radius.
    """
    current_pts = pts.copy()
    current_r = compute_radius(current_pts)
    
    # Binary search bounds
    r_low = current_r * 0.95
    r_high = current_r * 1.05 + 0.01
    
    # First, try to expand
    for _ in range(15):
        if time.time() > deadline - 5:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to achieve radius r_mid
        trial_pts = _push_apart(current_pts, r_mid, iterations=300)
        trial_r = compute_radius(trial_pts)
        
        if trial_r >= r_mid - 1e-10:
            # Success: r_mid is feasible
            current_pts = trial_pts
            current_r = trial_r
            r_low = r_mid
        else:
            # Fail: r_mid is not feasible
            r_high = mid
        
        if r_high - r_low < 1e-12:
            break
    
    # Final polish with very high beta
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 3
        if time_left > 3:
            x0 = current_pts.flatten()
            optimized = _lbfbsb_optimize(x0, time_limit=min(time_left, 15.0), beta=2000.0, maxiter=200)
            opt_pts = optimized.reshape(N, 2)
            opt_r = compute_radius(opt_pts)
            if opt_r > current_r:
                current_pts = opt_pts
                current_r = opt_r
    
    return current_pts, current_r

def construct():
    deadline = time.time() + DEADLINE
    
    np.random.seed(42)
    
    best_r = 0.0
    best_points = None
    
    def update_best(pts, r):
        nonlocal best_r, best_points
        if r > best_r:
            best_r = r
            best_points = pts.copy()
    
    # Phase 1: Generate diverse initializations
    initializations = [
        initial_rings(),
    ]
    
    # Add a hexagonal initialization
    spacing = 0.24
    hex_points = []
    max_k = 8
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 0.92:
                hex_points.append([x, y])
    hex_points = np.array(hex_points)
    if len(hex_points) >= N:
        dists = np.linalg.norm(hex_points, axis=1)
        indices = np.argsort(dists)
        selected = hex_points[indices[:N]]
    else:
        needed = N - len(hex_points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([hex_points, random_pts])
    initializations.append(selected)
    
    # Phase 2: For each initialization, run multi-stage annealing
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        current_pts = init_pts.copy()
        current_r = compute_radius(current_pts)
        update_best(current_pts, current_r)
        
        # Multi-stage annealing
        annealed_pts, annealed_r = _multi_stage_anneal(current_pts, deadline)
        update_best(annealed_pts, annealed_r)
        
        # Push-apart refinement
        if time.time() < deadline - 15:
            target_r = max(annealed_r, 0.12)
            pushed = _push_apart(annealed_pts, target_r, iterations=300)
            pushed_r = compute_radius(pushed)
            update_best(pushed, pushed_r)
            
            # Binary search polish on the best so far
            if pushed_r > annealed_r:
                annealed_pts = pushed
                annealed_r = pushed_r
    
    # Phase 3: Final binary search polish on the best solution
    if best_points is not None and time.time() < deadline - 10:
        polished_pts, polished_r = _binary_search_polish(best_points, deadline)
        update_best(polished_pts, polished_r)
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if best_points is not None and time.time() < deadline - 5:
        time_left = deadline - time.time() - 3
        if time_left > 3:
            x0 = best_points.flatten()
            optimized = _lbfbsb_optimize(x0, time_limit=min(time_left, 12.0), beta=3000.0, maxiter=300)
            opt_pts = optimized.reshape(N, 2)
            opt_r = compute_radius(opt_pts)
            update_best(opt_pts, opt_r)
    
    # Ensure validity
    if best_points is None:
        best_points = initial_rings()
    
    # Final validation and boundary enforcement
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.9999:
            best_points[i] *= 0.9999 / norms[i]
    
    # Verify no NaN or Inf
    if not np.all(np.isfinite(best_points)):
        best_points = initial_rings()
        norms = np.linalg.norm(best_points, axis=1)
        for i in range(N):
            if norms[i] > 0.9999:
                best_points[i] *= 0.9999 / norms[i]
    
    return best_points


if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")

