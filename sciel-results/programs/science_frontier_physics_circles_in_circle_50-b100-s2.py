"""Pack 50 equal circles in the unit disk: soft-min L-BFGS-B + simulated annealing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _raw_score(centers):
    """
    Calculate the maximum radius r such that all circles fit.
    r = min( min_pair_dist/2, min_edge_dist )
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    centers = centers.reshape(N, 2)
    norms = np.sqrt(np.sum(centers**2, axis=1))
    if np.any(norms > 1.0 + 1e-9):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_edge_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_edge_dist)
    return max(0.0, r)

def _soft_min_objective(centers, beta=100.0):
    """
    Objective function for L-BFGS-B: minimize -soft_min(constraints)
    """
    centers = centers.reshape(N, 2)
    norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
    edge_constraints = 1.0 - norms
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    neg_beta_x = -beta * all_constraints
    max_val = np.max(neg_beta_x)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    return -soft_min_val

def _soft_min_gradient(centers, beta=100.0):
    """
    Gradient of the soft-min objective: d/dx [-soft_min(constraints)]
    soft_min(x) = -(1/beta) * log(sum(exp(-beta*x_i)))
    d(soft_min)/dx_j = sum_i w_i * (d x_i / dx_j) where w_i = exp(-beta*x_i)/sum(exp(-beta*x_k))
    So d(-soft_min)/dx_j = -sum_i w_i * (d x_i / dx_j)
    """
    centers = centers.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
    edge_constraints = 1.0 - norms
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    neg_beta_x = -beta * all_constraints
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # w_i = exp(-beta*x_i) / sum(exp(-beta*x_k))
    
    # Edge constraint: x_i = 1 - |c_i|, d x_i / d c_i = -c_i / |c_i|
    # d(-soft_min)/d c_i = -w_i * (-c_i/|c_i|) = w_i * c_i/|c_i|
    edge_w = weights[:N]
    grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
    
    # Pair constraint: x_ij = |c_i - c_j|/2
    # d x_ij / d c_i = (c_i - c_j) / |c_i - c_j|
    # d x_ij / d c_j = (c_j - c_i) / |c_i - c_j| = -(c_i - c_j) / |c_i - c_j|
    # d(-soft_min)/d c_i = -w_ij * (c_i - c_j)/|c_i - c_j|
    # d(-soft_min)/d c_j = -w_ij * (-(c_i - c_j)/|c_i - c_j|) = w_ij * (c_i - c_j)/|c_i - c_j|
    
    n_pairs = len(pair_constraints)
    i_idx = iu[0]
    j_idx = iu[1]
    w_pair = weights[N:N+n_pairs]
    
    diff_pairs = centers[i_idx] - centers[j_idx]
    dist_pairs = dists[i_idx, j_idx]
    dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
    
    np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
    np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
    
    return grad.reshape(-1)

def _newton_polish(centers, max_iter=50, tol=1e-12):
    """
    Newton-Raphson contact-graph refinement: find the largest r such that
    all circles are tangent to neighbors or boundary.
    Uses the contact graph from the current configuration.
    """
    centers = centers.copy().reshape(N, 2)
    r = _raw_score(centers)
    
    if r <= 0:
        return centers, r
    
    for iteration in range(max_iter):
        # Compute current minimum distances
        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_dists = 1.0 - norms  # distance to boundary
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0  # half-distance = gap/2
        
        # Find contact pairs (within tolerance of current r)
        contact_tol = max(r * 0.05, 1e-6)
        contact_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if abs(pair_dists[i,j] - r) < contact_tol:
                    contact_pairs.append((i, j))
        
        # Contact with boundary
        contact_boundary = []
        for i in range(N):
            if abs(edge_dists[i] - r) < contact_tol:
                contact_boundary.append(i)
        
        # Build constraint system: we want to adjust centers so that
        # all contact constraints are satisfied at radius r_new
        # For now, do a simple gradient step toward increasing r
        
        # Compute gradient of r with respect to each center
        # r = min(min_pair_dist/2, min_edge_dist)
        # Use subgradient: average of gradients of active constraints
        
        grad = np.zeros_like(centers)
        
        # Edge gradient contribution (from active boundary contacts)
        if contact_boundary:
            for i in contact_boundary:
                if norms[i] > 1e-10:
                    grad[i] += centers[i] / norms[i]
        
        # Pair gradient contribution
        if contact_pairs:
            for i, j in contact_pairs:
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    direction = d / dist
                    grad[i] -= direction
                    grad[j] += direction
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        # Step size: try to increase r
        step_size = 0.01
        new_centers = centers + step_size * grad
        
        # Keep inside unit circle
        new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
        out_mask = new_norms > 0.9999
        if np.any(out_mask):
            new_centers[out_mask] *= 0.9999 / new_norms[out_mask, np.newaxis]
        
        new_r = _raw_score(new_centers)
        
        if new_r > r + 1e-15:
            centers = new_centers
            r = new_r
        else:
            # Try smaller step
            step_size *= 0.5
            if step_size < 1e-10:
                break
            new_centers = centers + step_size * grad
            new_norms = np.sqrt(np.sum(new_centers**2, axis=1))
            out_mask = new_norms > 0.9999
            if np.any(out_mask):
                new_centers[out_mask] *= 0.9999 / new_norms[out_mask, np.newaxis]
            new_r = _raw_score(new_centers)
            if new_r > r + 1e-15:
                centers = new_centers
                r = new_r
            else:
                break
        
        if abs(r - _raw_score(centers)) < tol:
            break
    
    return centers, r

def _refine(centers, beta=300.0, maxiter=200):
    """
    Local refinement using L-BFGS-B with soft-min objective.
    """
    try:
        result = minimize(
            _soft_min_objective,
            centers.flatten(),
            args=(beta,),
            jac=lambda x, b: _soft_min_gradient(x, b),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        refined = result.x.reshape(N, 2)
        # Ensure points stay inside
        norms = np.sqrt(np.sum(refined**2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            refined[mask] *= (1.0 - 1e-6) / norms[mask, np.newaxis]
        return refined
    except Exception:
        return centers

def _simulated_annealing(centers, steps=1500, T_init=0.015, T_final=0.0003, deadline=None):
    """
    Simulated Annealing for global exploration.
    """
    current_centers = centers.copy()
    current_score = _raw_score(current_centers)
    best_centers = current_centers.copy()
    best_score = current_score
    
    iu = np.triu_indices(N, 1)
    start_time = time.time()
    
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        if time.time() - start_time > 25.0:
            break
            
        T = T_init * (T_final / T_init) ** (i / steps)
        
        # Move 1-3 random points
        num_moves = np.random.randint(1, 4)
        indices = np.random.choice(N, num_moves, replace=False)
        
        new_centers = current_centers.copy()
        step_size = max(T * 0.5, 0.003)
        moves = np.random.normal(0, step_size, size=(num_moves, 2))
        new_centers[indices] += moves
        
        # Keep inside unit circle
        norms = np.sqrt(np.sum(new_centers**2, axis=1))
        out_mask = norms > 0.9995
        if np.any(out_mask):
            new_centers[out_mask] *= 0.9995 / norms[out_mask, np.newaxis]
            
        new_score = _raw_score(new_centers)
        delta = new_score - current_score
        
        if delta > 0:
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = new_centers.copy()
        else:
            clamped_delta = max(delta, -10.0 * T)
            if np.random.rand() < np.exp(clamped_delta / T):
                current_centers = new_centers
                current_score = new_score
    
    return best_centers, best_score

def _make_rings(radii, counts, offsets=None):
    """Generate ring-based initial configuration."""
    centers = []
    centers.append([0.0, 0.0])
    start_idx = 1
    for r, c in zip(radii, counts):
        base_offset = offsets[start_idx-1] if offsets else 0.0
        for k in range(c):
            theta = 2 * np.pi * k / c + base_offset
            centers.append([r * np.cos(theta), r * np.sin(theta)])
        start_idx += 1
    return np.array(centers[:N])

def _hexagonal_seed(n, jitter=0.0):
    """Generate hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    
    if len(points) >= N:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:N]
        pts = points[idx]
        if jitter > 0:
            pts += np.random.normal(0, jitter, pts.shape)
            # Project back inside
            norms = np.sqrt(np.sum(pts**2, axis=1))
            out_mask = norms > 0.999
            if np.any(out_mask):
                pts[out_mask] *= 0.999 / norms[out_mask, np.newaxis]
        return pts
    return None

def construct():
    """Main entry point: find best packing of 50 circles in unit disk."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Ring-based initializations - more systematic exploration
    # Best known packing has rings with counts approximately: 1, 6, 12, 18, 13
    base_r1 = [0.0, 0.24, 0.46, 0.68, 0.88]
    base_c1 = [1, 6, 12, 18, 13]
    initializations.append(_make_rings(base_r1, base_c1))
    
    # Perturbed rings with various seeds
    for seed in range(15):
        np.random.seed(seed)
        perturbed_radii = [base_r1[0]] + [r + np.random.uniform(-0.03, 0.03) for r in base_r1[1:]]
        offsets = [0.0] * len(base_r1)
        for s in range(1, len(base_r1)):
            offsets[s] = np.random.uniform(-0.2, 0.2)
        initializations.append(_make_rings(perturbed_radii, base_c1, offsets))
    
    # Different ring count distributions - explore combinatorial space
    ring_configs = [
        ([0.0, 0.25, 0.5, 0.75, 0.9], [1, 6, 12, 18, 13]),
        ([0.0, 0.22, 0.44, 0.66, 0.85], [1, 7, 14, 14, 14]),
        ([0.0, 0.2, 0.4, 0.6, 0.8, 0.95], [1, 5, 10, 15, 10, 9]),
        ([0.0, 0.18, 0.38, 0.58, 0.78, 0.92], [1, 6, 11, 16, 10, 6]),
        ([0.0, 0.2, 0.45, 0.7, 0.9], [1, 8, 16, 14, 11]),
        ([0.0, 0.26, 0.52, 0.78, 0.94], [1, 6, 12, 18, 13]),
        ([0.0, 0.2, 0.42, 0.64, 0.82, 0.96], [1, 6, 10, 14, 10, 9]),
        ([0.0, 0.24, 0.48, 0.72, 0.92], [1, 6, 12, 18, 13]),
    ]
    for radii, counts in ring_configs:
        total = sum(counts)
        if total < N:
            counts = list(counts)
            counts[-1] += N - total
        elif total > N:
            counts = list(counts)
            counts[-1] -= total - N
            if counts[-1] <= 0:
                continue
        cfg = _make_rings(radii, counts)
        if cfg is not None and len(cfg) == N:
            initializations.append(cfg)
    
    # 2. Hexagonal lattice initializations
    hex_pts = _hexagonal_seed(N)
    if hex_pts is not None:
        initializations.append(hex_pts)
        # With jitter
        for j in [0.01, 0.02, 0.05]:
            hex_jit = _hexagonal_seed(N, jitter=j)
            if hex_jit is not None:
                initializations.append(hex_jit)
    
    # 3. Random initializations
    for seed in range(8):
        np.random.seed(100 + seed)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)
    
    # Filter valid initializations
    valid_inits = []
    for init in initializations:
        if init is not None and len(init) == N and np.all(np.isfinite(init)):
            norms = np.sqrt(np.sum(init**2, axis=1))
            if np.all(norms <= 1.0):
                valid_inits.append(init)
    
    if not valid_inits:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        radii = np.sqrt(np.linspace(0, 1, N))
        valid_inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Optimize each initialization
    best_points = None
    best_score = 0.0
    
    # Allocate time per initialization based on number of inits
    n_inits = len(valid_inits)
    time_per_init = (deadline - start_time - 15.0) / max(n_inits, 1)
    
    for idx, init in enumerate(valid_inits):
        if time.time() > deadline - 15.0:
            break
        
        # Simulated annealing phase with deadline
        sa_deadline = min(time.time() + time_per_init * 0.5, deadline - 10.0)
        sa_steps = int(time_per_init * 0.5 * 80)  # ~80 steps/sec estimate
        sa_steps = max(100, min(sa_steps, 3000))
        sa_points, sa_score = _simulated_annealing(init, steps=sa_steps, 
                                                    T_init=0.015, T_final=0.0003,
                                                    deadline=sa_deadline)
        
        # Refine with L-BFGS-B
        refined_points = _refine(sa_points, beta=300.0, maxiter=150)
        refined_score = _raw_score(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
        
        # Try another refinement with different beta
        if time.time() < deadline - 12.0:
            refined_points2 = _refine(refined_points, beta=500.0, maxiter=100)
            refined_score2 = _raw_score(refined_points2)
            
            if refined_score2 > best_score:
                best_score = refined_score2
                best_points = refined_points2.copy()
    
    # Final high-precision polish on best
    if best_points is not None and time.time() < deadline - 8.0:
        # Multiple L-BFGS passes with increasing beta
        current = best_points.copy()
        for beta_val in [200, 500, 1000, 2000]:
            if time.time() > deadline - 5.0:
                break
            polished = _refine(current, beta=beta_val, maxiter=200)
            polished_score = _raw_score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
                current = polished
        
        # Newton polish
        if time.time() < deadline - 3.0:
            newton_pts, newton_score = _newton_polish(best_points, max_iter=100, tol=1e-13)
            if newton_score > best_score:
                best_score = newton_score
                best_points = newton_pts
    
    # Ensure we return a valid result
    if best_points is None or len(best_points) != N:
        angles = 2 * np.pi * np.arange(N) / N
        radii = np.sqrt(np.linspace(0, 1, N))
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Final validity check
    norms = np.sqrt(np.sum(best_points**2, axis=1))
    out_mask = norms > 1.0
    if np.any(out_mask):
        best_points[out_mask] *= (1.0 - 1e-6) / norms[out_mask, np.newaxis]
    
    # Clamp to ensure no point is exactly on boundary (strictly inside)
    norms = np.sqrt(np.sum(best_points**2, axis=1))
    boundary_mask = norms > 1.0 - 1e-10
    if np.any(boundary_mask):
        best_points[boundary_mask] *= (1.0 - 1e-8) / norms[boundary_mask, np.newaxis]
    
    return best_points
