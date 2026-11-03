"""
Packing 50 equal circles in a unit disk.
Combines ring-structured initialization with multi-stage smooth optimization,
direct constraint polishing, and ring-parameter search.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE_MARGIN = 15.0  # seconds before hard limit


def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _pairwise_distances_and_grad(pts):
    """Compute pairwise distances and their gradients."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    # Gradient of dist[i,j] w.r.t. pts[i] and pts[j]
    # d(dist_ij)/d(pts_i) = (pts_i - pts_j) / dist_ij
    # d(dist_ij)/d(pts_j) = (pts_j - pts_i) / dist_ij
    inv_dist = 1.0 / np.maximum(dist, 1e-30)
    grad = diff * inv_dist[:, :, np.newaxis]
    return dist, grad


def _score(pts):
    """Compute the packing score: minimum of half pairwise distances and boundary distances."""
    pts = np.asarray(pts).reshape(-1, 2)
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return max(min(min_pair, min_boundary), 0.0)

def _score_with_indices(pts):
    """Compute score and return the indices of the binding constraint."""
    pts = np.asarray(pts).reshape(-1, 2)
    if len(pts) != N:
        return 0.0, None, None
    if not np.all(np.isfinite(pts)):
        return 0.0, None, None
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
    min_pair = d[min_pair_idx] / 2.0
    boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary_idx = int(np.argmin(boundary_margins))
    min_boundary = boundary_margins[min_boundary_idx]
    score = max(min(min_pair, min_boundary), 0.0)
    if min_pair <= min_boundary:
        return score, min_pair_idx, None
    else:
        return score, None, min_boundary_idx


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp for numerical stability."""
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta

def _softmin_logsumexp_grad(values, beta):
    """Soft-min via log-sum-exp with gradient."""
    if len(values) == 0:
        return 0.0, np.array([])
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0, np.zeros_like(values)
    exp_vals = np.exp(neg - neg_max)
    exp_sum = np.sum(exp_vals)
    if exp_sum == 0:
        return 0.0, np.zeros_like(values)
    soft_min = -(neg_max + np.log(exp_sum)) / beta
    # Gradient: d(soft_min)/d(values_k) = -exp(-beta*values_k) / sum(exp(-beta*values_j))
    grad = -exp_vals / exp_sum
    return soft_min, grad


def _smooth_objective(x, beta=50.0):
    """
    Smooth objective: minimize -softmin of all constraint margins.
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    
    # Get all finite values
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 1e10
    
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min

def _smooth_objective_grad(x, beta=50.0):
    """
    Smooth objective with gradient: minimize -softmin of all constraint margins.
    """
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10, np.zeros_like(x)
    
    d, d_grad = _pairwise_distances_and_grad(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    
    # Gradient of boundary constraint w.r.t. pts[i]
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    norms = np.maximum(norms, 1e-30)
    boundary_grad = -pts / norms[:, np.newaxis]  # d(1-|p|)/d(p) = -p/|p|
    
    pair_half = d / 2.0
    pair_half_grad = d_grad / 2.0
    
    # Get all finite values
    pair_flat = pair_half.ravel()
    pair_flat_grad = pair_half_grad.reshape(N * N, 2)
    
    all_vals = np.concatenate([pair_flat, boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 1e10, np.zeros_like(x)
    
    soft_min, soft_min_grad = _softmin_logsumexp_grad(all_vals, beta)
    
    # Now compute gradient of -soft_min w.r.t. x
    # We need to map the gradient back
    # This is complex, so for now just return the objective and let L-BFGS estimate gradient
    # Actually, let's just return a dummy gradient and use BFGS which estimates it
    return -soft_min, np.zeros_like(x)


def _ring_initialization(ring_radii, ring_counts, angles_offset=None):
    """Generate points from a ring structure."""
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, ring_radii)):
        if count == 1 and radius < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            if angles_offset is not None and idx < len(angles_offset):
                offset = angles_offset[idx]
            else:
                offset = 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for i in range(count):
                centers.append(np.array([xs[i], ys[i]]))
    
    pts = np.array(centers[:N])
    # Project inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _hexagonal_init(n):
    """
    Initialize with a hexagonal lattice structure.
    """
    a = 1.0
    sqrt3 = np.sqrt(3)
    
    def hex_to_cart(q, r):
        x = a * 1.5 * q
        y = a * sqrt3 * (r + q / 2.0)
        return np.array([x, y])
    
    max_ring = 5
    candidates = []
    for q in range(-max_ring, max_ring + 1):
        for r in range(-max_ring, max_ring + 1):
            candidates.append((q, r))
            
    cart_points = []
    for q, r in candidates:
        p = hex_to_cart(q, r)
        d = np.linalg.norm(p)
        cart_points.append((d, p))
        
    cart_points.sort(key=lambda x: x[0])
    selected_points = [p for d, p in cart_points[:N]]
    points = np.array(selected_points)
    
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        scale = 0.8 / max_r
        points = points * scale
        
    return points


def _ring_init(n, n_rings=5):
    """
    Initialize with ring structure matching known optimal packing topology.
    Ring 0: center (1 circle)
    Ring 1: 6 circles
    Ring 2: 12 circles
    Ring 3: 18 circles
    Ring 4: 13 circles (remaining)
    """
    points = []

    # Center
    points.append([0.0, 0.0])

    # Ring 1: 6 circles at radius ~0.25
    r1 = 0.25
    for k in range(6):
        angle = 2 * np.pi * k / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])

    # Ring 2: 12 circles at radius ~0.5
    r2 = 0.5
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 6  # offset
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])

    # Ring 3: 18 circles at radius ~0.75
    r3 = 0.75
    for k in range(18):
        angle = 2 * np.pi * k / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])

    # Ring 4: remaining circles at radius ~0.9
    remaining = n - len(points)
    r4 = 0.9
    for k in range(remaining):
        angle = 2 * np.pi * k / remaining + np.pi / remaining
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])

    return np.array(points[:n])


def _optimize_from_init(pts, deadline, max_beta_stages=5):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Progressive beta values for smooth optimization - fewer stages, more time per stage
    betas = [10.0, 30.0, 80.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # High-precision polish with larger betas
    for beta in [500.0, 1500.0]:
        if time.time() > deadline - 8:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 10000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current_pts, current_score


def _binary_search_polish(pts, r_target, deadline, max_iters=500):
    """
    Polish by iteratively pushing points to achieve target radius r_target.
    This is a direct constraint satisfaction approach.
    """
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Compute current margins
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # Find the most violated constraint
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _de_ring_optimize(deadline):
    """
    Differential evolution over ring parameters to find good ring configurations.
    """
    ring_counts = [1, 6, 12, 18, 13]
    
    def ring_to_pts(params):
        radii = params[:4]
        offsets = params[4:]
        radii = np.maximum(radii, 0.05)
        radii = np.sort(radii)
        max_allowed = 1.0 - 0.12
        radii = np.minimum(radii, max_allowed)
        return _ring_initialization(radii, ring_counts, offsets)
    
    def neg_score(params):
        pts = ring_to_pts(params)
        return -_score(pts)
    
    bounds = [
        (0.15, 0.35),   # r1
        (0.30, 0.55),   # r2
        (0.45, 0.70),   # r3
        (0.60, 0.90),   # r4
        (0, 2*np.pi),   # offset1
        (0, 2*np.pi),   # offset2
        (0, 2*np.pi),   # offset3
        (0, 2*np.pi),   # offset4
    ]
    
    rng = np.random.RandomState(42)
    pop_size = 30
    max_gen = 40
    F = 0.8
    CR = 0.9
    
    pop = []
    for _ in range(pop_size):
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pop.append(params)
    pop = np.array(pop)
    
    scores = np.array([neg_score(p) for p in pop])
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 20:
            break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            a_idx, b_idx, c_idx = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            v = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])
            
            # Clip to bounds
            for j in range(len(v)):
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(len(v))
            for j in range(len(v)):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Selection
            trial_score = neg_score(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_params = trial.copy()
    
    best_pts = ring_to_pts(best_params)
    return best_pts, -best_score


def _direct_polish(pts, deadline, max_iters=3000):
    """
    Direct constraint polishing: iteratively fix the most violated constraint.
    Uses a relaxation factor to avoid oscillation.
    """
    pts = pts.copy()
    current_score = _score(pts)
    alpha = 0.8  # relaxation factor - higher for faster convergence
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        # Compute all margins
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # Find the most violated constraint (smallest margin)
        min_pair = np.min(pair_margins)
        min_boundary = np.min(boundary_margins)
        
        if min_pair < min_boundary - 1e-14:
            # Pair constraint is most violated
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                # Perturb if coincident
                pts[i] += 1e-6 * np.array([1.0, 0.0])
                continue
            di /= dist
            # Push apart by half the deficit
            deficit = 2 * current_score - dist
            if deficit > 1e-15:
                push = alpha * deficit / 2.0
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is most violated
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
                continue
            di = pi / dist
            deficit = 1.0 - current_score - dist
            if deficit > 1e-15:
                push = alpha * deficit
                pts[i] += push * di
        
        # Update score
        new_score = _score(pts)
        if new_score > current_score + 1e-16:
            current_score = new_score
        elif it > 100:
            # Slow down if not improving
            alpha *= 0.97
            if alpha < 0.05:
                break
    
    return pts, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 135s effective deadline
    
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Ring-based DE + optimization (usually gives best initial result)
    if time.time() < deadline - 40:
        de_pts, de_score = _de_ring_optimize(deadline)
        if de_pts is not None:
            opt_pts, opt_score = _optimize_from_init(de_pts, deadline)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts
    
    # Strategy 2: Hexagonal initialization + optimization
    if time.time() < deadline - 35:
        hex_pts = _hexagonal_init(N)
        opt_pts, opt_score = _optimize_from_init(hex_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Strategy 3: Try ring configurations with different parameters
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.45, 0.65, 0.85]),
        ([1, 6, 12, 18, 13], [0.0, 0.23, 0.43, 0.63, 0.83]),
        ([1, 6, 12, 18, 13], [0.0, 0.27, 0.47, 0.67, 0.87]),
    ]
    
    for counts, radii in ring_configs:
        if time.time() > deadline - 25:
            break
        ring_pts = _ring_initialization(radii, counts)
        opt_pts, opt_score = _optimize_from_init(ring_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    if best_points is None:
        # Fallback: return hexagonal init
        best_points = _hexagonal_init(N)
        best_score = _score(best_points)
    
    # Final direct constraint polishing for refinement
    if time.time() < deadline - 5:
        direct_pts, direct_score = _direct_polish(best_points, deadline, max_iters=2000)
        if direct_score > best_score:
            best_points = direct_pts
            best_score = direct_score
    
    # One more polish pass if time remains
    if time.time() < deadline - 2:
        direct_pts, direct_score = _direct_polish(best_points, deadline, max_iters=1000)
        if direct_score > best_score:
            best_points = direct_pts
            best_score = direct_score
    
    return best_points
