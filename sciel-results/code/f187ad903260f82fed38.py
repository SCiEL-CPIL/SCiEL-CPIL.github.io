"""
Packing 50 equal circles in a unit disk using a combined approach:
1. Soft-min surrogate objective with proper log-sum-exp stability (from inspiration)
2. Contact-graph weighted gradient for identifying critical constraints (from current)
3. Multi-stage beta annealing for convergence
4. Force-directed pre-conditioning
5. L-BFGS-B refinement with binary search on radius

Key structural combination:
- The soft-min provides a smooth, differentiable objective
- The contact graph weighting focuses optimization effort on binding constraints
- Binary search on radius with gradient-based position optimization enables precise convergence
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = None
DEADLINE = 130.0  # safety margin before 150s kill


def _score(points):
    """Compute packing radius: min(min pairwise dist/2, min (1-|c_i|))."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    boundary_margin = np.min(1.0 - dists)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    return min(min_pair_dist / 2.0, boundary_margin)


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _hex_lattice_init(r_guess):
    """Hexagonal lattice initialization."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _build_contact_weights(points, r_target, n_iter=8):
    """
    Build contact weights using simplified message passing.
    Returns (N, N) weight matrix where higher values indicate more critical contacts.
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    contact_threshold = 2.0 * r_target
    is_contact = dists < contact_threshold
    np.fill_diagonal(is_contact, False)
    
    # Proximity-based initial weights
    proximity = np.clip((contact_threshold - dists) / (contact_threshold + 1e-10), 0, 1)
    np.fill_diagonal(proximity, 0)
    
    # Simple iterative refinement: amplify weights of tightly constrained pairs
    weights = proximity.copy()
    for _ in range(n_iter):
        new_weights = weights.copy()
        for i in range(n):
            neighbors = np.where(is_contact[i])[0]
            if len(neighbors) > 1:
                # Pairs with many close neighbors are more critical
                local_density = np.sum(weights[i, neighbors])
                new_weights[i, neighbors] = weights[i, neighbors] * (1.0 + 0.1 * local_density)
        weights = new_weights
        # Normalize to prevent overflow
        max_w = np.max(weights)
        if max_w > 10.0:
            weights /= max_w
    
    return weights


def _soft_min_objective_and_grad(x, r_target, contact_weights, beta):
    """
    Objective: minimize -soft_min of constraint violations.
    Constraints: (1 - |c_i|) - r_target for boundary, (|c_i - c_j|/2) - r_target for pairs.
    
    We want all constraints >= 0. Soft-min approximates the minimum.
    """
    pts = x.reshape(N, 2)
    n = N
    
    # Boundary constraints: (1 - |c_i|) - r_target
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    bound_vals = (1.0 - norms) - r_target
    
    # Pair constraints: |c_i - c_j|/2 - r_target
    iu = np.triu_indices(n, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_vals = pair_dists / 2.0 - r_target
    
    # Apply contact weights to pair constraints
    pair_weights = contact_weights[iu[0], iu[1]]
    pair_weights = np.maximum(pair_weights, 0.1)  # Ensure no constraint is fully ignored
    
    # Combine all constraint values
    all_vals = np.concatenate([bound_vals, pair_vals])
    all_weights = np.concatenate([np.ones(n), pair_weights])
    
    # Soft-min with log-sum-exp stability
    m = np.min(all_vals)
    shifted = all_vals - m
    # Weighted soft-min: use weights to emphasize critical constraints
    weighted_shifted = shifted * all_weights
    max_ws = np.max(weighted_shifted)
    exp_terms = np.exp(-(beta) * (weighted_shifted - max_ws)) * all_weights
    S = np.sum(exp_terms)
    
    if S < 1e-300:
        S = 1e-300
    
    soft_min_val = m - (1.0 / beta) * np.log(S)
    obj_val = -soft_min_val
    
    # Gradient computation
    weights = exp_terms / S
    grad = np.zeros_like(pts)
    
    # Boundary gradient
    bound_w = weights[:n]
    mask_b = bound_w > 1e-12
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        w_b = bound_w[idx_b]
        norm_b = norms[idx_b]
        norm_safe = np.where(norm_b > 1e-10, norm_b, 1.0)
        # d/dx of (1-|x|) is -x/|x|
        vecs_b = -pts[idx_b] / norm_safe[:, np.newaxis]
        contrib_b = w_b[:, np.newaxis] * vecs_b
        np.add.at(grad, idx_b, contrib_b)
    
    # Pair gradient
    pair_w = weights[n:]
    mask_p = pair_w > 1e-12
    if np.any(mask_p):
        i_m = iu[0][mask_p]
        j_m = iu[1][mask_p]
        w_m = pair_w[mask_p]
        d_m = pair_dists[mask_p]
        d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
        vecs = (pts[i_m] - pts[j_m]) / d_safe[:, np.newaxis]
        contrib = w_m[:, np.newaxis] * 0.5 * vecs
        np.add.at(grad, i_m, contrib)
        np.add.at(grad, j_m, -contrib)
    
    return obj_val, grad.flatten()


def _optimize_for_radius(pts_init, r_target, start_beta=50.0, end_beta=50000.0,
                         n_stages=5, max_iter_per_stage=800, tol=1e-12):
    """
    Optimize positions for a given target radius using multi-stage beta annealing
    with contact-weighted soft-min objective.
    """
    current_pts = pts_init.copy()
    
    for stage in range(n_stages):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        
        frac = stage / max(n_stages - 1, 1)
        beta = start_beta * (end_beta / start_beta) ** frac
        
        # Build contact weights from current configuration
        contact_weights = _build_contact_weights(current_pts, r_target, n_iter=6)
        
        def objective(x):
            val, grad = _soft_min_objective_and_grad(x, r_target, contact_weights, beta)
            return val, grad
        
        x0 = current_pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter_per_stage, 'ftol': tol,
                                   'gtol': 1e-8, 'maxls': 50})
        current_pts = result.x.reshape(N, 2)
    
    return current_pts


def _force_directed(points, max_iters=2000, dt=0.005):
    """Force-directed optimization with strong repulsion for close pairs."""
    points = points.copy()
    n = points.shape[0]
    best_score = _score(points)
    best_points = points.copy()
    
    for it in range(max_iters):
        if time.time() - START_TIME > DEADLINE:
            break
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        forces = np.zeros_like(points)
        current_r = _score(points)
        threshold = 2.5 * current_r if current_r > 0.01 else 0.4
        
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < threshold:
                    f = diff[i, j] / (d[i, j] ** 3 + 1e-8)
                    forces[i] += f
                    forces[j] -= f
        
        dists = np.linalg.norm(points, axis=1)
        for i in range(n):
            if dists[i] > 0.95:
                forces[i] -= 0.5 * points[i] / dists[i]
        
        points += dt * forces
        
        dists = np.linalg.norm(points, axis=1)
        for i in range(n):
            if dists[i] > 0.99:
                points[i] *= 0.99 / dists[i]
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        dt *= 0.999
    
    return best_points, best_score


def _lbfgs_refine(points, max_iter=2000):
    """L-BFGS-B refinement of the packing using direct score maximization."""
    def objective(flat):
        pts = flat.reshape(N, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50})
    return result.x.reshape(N, 2), -result.fun


def _full_optimize(points, budget_seconds=25.0):
    """Full optimization pipeline: force-directed + radius-targeted optimization + L-BFGS-B."""
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Phase 1: Force-directed pre-conditioning
    if time.time() - t0 < budget_seconds * 0.25:
        pts_f, score_f = _force_directed(points, max_iters=1500, dt=0.004)
        if score_f > best_score:
            best_score = score_f
            best_points = pts_f.copy()
    
    # Phase 2: Radius-targeted optimization with binary search
    r_low = best_score * 0.95
    r_high = min(0.145, best_score * 1.15)
    
    for _ in range(5):
        if time.time() - t0 > budget_seconds * 0.75:
            break
        r_mid = (r_low + r_high) / 2.0
        
        pts_opt = _optimize_for_radius(best_points, r_mid,
                                       start_beta=50.0, end_beta=50000.0,
                                       n_stages=4, max_iter_per_stage=600)
        score_opt = _score(pts_opt)
        
        if score_opt >= r_mid - 1e-8:
            r_low = r_mid
            if score_opt > best_score:
                best_score = score_opt
                best_points = pts_opt.copy()
        else:
            r_high = r_mid
        
        if r_high - r_low < 1e-6:
            break
    
    # Phase 3: L-BFGS-B refinement
    if time.time() - t0 < budget_seconds * 0.9:
        pts_lb, score_lb = _lbfgs_refine(best_points, max_iter=1000)
        if score_lb > best_score:
            best_score = score_lb
            best_points = pts_lb.copy()
    
    # Phase 4: Final fine-tune
    if time.time() - t0 < budget_seconds * 0.98:
        pts_final, score_final = _lbfgs_refine(best_points, max_iter=400)
        if score_final > best_score:
            best_score = score_final
            best_points = pts_final.copy()
    
    return best_points, best_score


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Try multiple initializations
    initializations = []
    
    # Hex lattice
    hex_pts = _hex_lattice_init(0.12)
    initializations.append(hex_pts)
    
    # Ring configurations
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        ([1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.80, 0.98]),
        ([1, 7, 12, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 13, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.20, 0.42, 0.65, 0.88]),
    ]
    
    for ring_sizes, radii in ring_configs:
        pts = _ring_init(ring_sizes, radii)
        initializations.append(pts)
    
    # Add some perturbed versions
    for i in range(3):
        base = initializations[i % len(initializations)]
        perturbed = base + rng.normal(0, 0.02, size=(N, 2))
        # Clamp to unit disk
        norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
        for j in range(N):
            if norms[j] > 0.95:
                perturbed[j] *= 0.95 / norms[j]
        initializations.append(perturbed)
    
    # Optimize each initialization
    for init_idx, pts_init in enumerate(initializations):
        if time.time() - START_TIME > DEADLINE - 30:
            break
        
        remaining = DEADLINE - (time.time() - START_TIME) - 20
        budget = min(15.0, remaining * 0.8)
        if budget < 5:
            break
        
        pts_opt, score_opt = _full_optimize(pts_init, budget_seconds=budget)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Multi-start with symmetry breaking
    n_starts = 3
    for s in range(n_starts):
        if time.time() - START_TIME > DEADLINE - 25:
            break
        
        if best_points is not None:
            r_target = max(best_score, 0.12)
        else:
            r_target = 0.12
        
        if s == 0:
            pts_init = best_points.copy() if best_points is not None else _hex_lattice_init(r_target)
        elif s == 1:
            pts_init = _hex_lattice_init(r_target)
        else:
            pts_init = best_points.copy() if best_points is not None else _hex_lattice_init(r_target)
            noise = rng.normal(0, 0.03 * (s + 1), size=pts_init.shape)
            pts_init += noise
            norms = np.hypot(pts_init[:, 0], pts_init[:, 1])
            for i in range(N):
                if norms[i] > 1.0 - r_target:
                    pts_init[i] *= (1.0 - r_target) / norms[i]
        
        remaining = DEADLINE - (time.time() - START_TIME) - 15
        if remaining < 10:
            break
            
        pts_opt = _optimize_for_radius(pts_init, r_target,
                                       start_beta=20.0, end_beta=30000.0,
                                       n_stages=4, max_iter_per_stage=700)
        score = _score(pts_opt)
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() - START_TIME < DEADLINE - 5:
        r_current = _score(best_points)
        if r_current > 0.05:
            pts_polish = _optimize_for_radius(best_points, r_current,
                                              start_beta=5000.0, end_beta=100000.0,
                                              n_stages=3, max_iter_per_stage=400)
            score_polish = _score(pts_polish)
            if score_polish > best_score:
                best_score = score_polish
                best_points = pts_polish.copy()
    
    if best_points is None:
        best_points = _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])
    
    # Ensure validity
    dists = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            best_points[i] *= 0.99 / max(dists[i], 1e-10)
    
    return best_points


# EVOLVE-BLOCK-START
# The main construct function is defined above
# EVOLVE-BLOCK-END
