import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(10):
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * (0.90 + 0.05 * attempt)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        if len(rows) < n_points:
            continue
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.03, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        k = int(1.0 / spacing) + 4
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
    return best_pts


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    # Use efficient computation: sum of squared differences
    sq_norms = np.sum(pts * pts, axis=1)
    cross = pts @ pts.T
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
    np.maximum(sq_dists, 0.0, out=sq_dists)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair = np.sqrt(sq_dists.min()) / 2.0
    return min(min_pair, min_bound)


def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum


def _objective_and_grad(x, T):
    """
    Objective: minimize -softmin(constraints) where constraints are:
      - pairwise half-distances d_ij/2 for i<j
      - boundary distances 1 - |c_i|
    
    Uses log-sum-exp soft-min for differentiability.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary constraints: 1 - |c_i|
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (n,)
    
    # Pairwise constraints: d_ij / 2 for i < j
    # Compute all pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    # Take upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0
    
    # All constraint values
    all_vals = np.concatenate([pair_half, boundary])
    
    # Soft minimum
    obj = -_soft_min_logsumexp(all_vals, T)
    
    # Gradient computation
    # d(softmin)/d(vals) = -exp(-vals/T) / sum(exp(-vals/T))
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        # Degenerate case: gradient is one-hot at min
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)  # sum to 1
        grad_all = -weights  # d(softmin)/d(val_k)
    
    # Now chain rule: d(obj)/d(x) = -sum_k grad_all[k] * d(constraint_k)/d(x)
    grad_obj = np.zeros_like(x)
    
    n_pairs = len(pair_half)
    
    # Unit vectors for pairwise gradients
    # Reuse d, but need to handle diagonal (0 distance)
    safe_dists = np.where(d > 1e-10, d, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]  # (n, n, 2)
    
    # Build weight matrix W where W[i,j] is the weight of pair (i,j)
    # Note: grad_all[:n_pairs] corresponds to pairs in iu order
    W = np.zeros((n, n))
    W[iu] = grad_all[:n_pairs]
    W[iu[1], iu[0]] = grad_all[:n_pairs]  # symmetric
    
    # Gradient for each point c_i from pairs:
    # grad_pairs[i] = 0.5 * sum_j W[i,j] * unit_vecs[i,j]
    # Using einsum for efficiency
    grad_pairs = 0.5 * np.einsum('ij,ijc->ic', W, unit_vecs)
    
    # Boundary part: constraint = 1 - ||c_i||
    # d(constraint)/d(c_i) = -c_i / ||c_i||
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]
    
    # Total gradient
    grad_pts = grad_pairs + grad_boundary
    grad_obj = grad_pts.ravel()
    
    return obj, grad_obj


def _tighten_min_constraint(pts, n_iter=50, step_size=0.001):
    """
    Directly push the most constrained points to increase the minimum constraint.
    For each iteration, find the minimum constraint and nudge points to increase it.
    """
    pts = pts.copy()
    for _ in range(n_iter):
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        # Efficient distance computation
        sq_norms = np.sum(pts * pts, axis=1)
        cross = pts @ pts.T
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
        np.maximum(sq_dists, 0.0, out=sq_dists)
        np.fill_diagonal(sq_dists, np.inf)
        pair_half = np.sqrt(sq_dists) / 2.0
        
        # Find the minimum constraint
        min_pair_idx = np.unravel_index(np.argmin(pair_half), pair_half.shape)
        min_boundary_idx = np.argmin(boundary)
        
        min_pair_val = pair_half[min_pair_idx]
        min_boundary_val = boundary[min_boundary_idx]
        
        if min_pair_val < min_boundary_val:
            # The binding constraint is a pair
            i, j = min_pair_idx
            if i == j:
                continue
            vec = pts[i] - pts[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            unit = vec / dist
            # Move i away from j and j away from i
            pts[i] += step_size * unit
            pts[j] -= step_size * unit
        else:
            # The binding constraint is boundary
            i = min_boundary_idx
            norm_i = norms[i]
            if norm_i < 1e-10:
                continue
            unit = pts[i] / norm_i
            # Move i inward
            pts[i] -= step_size * unit
        
        # Ensure points stay inside the container
        new_norms = np.linalg.norm(pts, axis=1)
        mask = new_norms >= 1.0
        if np.any(mask):
            pts[mask] *= 0.999 / new_norms[mask, np.newaxis]
    
    return pts


def _local_refine(pts, n_iter=200, step_size=0.001, n_critical=20):
    """
    Local refinement: identify the most constrained points and optimize them
    while keeping others fixed. This allows more aggressive optimization of
    the critical region.
    """
    pts = pts.copy()
    
    for it in range(n_iter):
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        sq_norms = np.sum(pts * pts, axis=1)
        cross = pts @ pts.T
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
        np.maximum(sq_dists, 0.0, out=sq_dists)
        np.fill_diagonal(sq_dists, np.inf)
        pair_half = np.sqrt(sq_dists) / 2.0
        
        # Find the most constrained points
        min_val = min(pair_half.min(), boundary.min())
        
        # Identify critical points: those involved in constraints near the minimum
        critical_mask = np.zeros(len(pts), dtype=bool)
        critical_mask[boundary < min_val + 0.002] = True
        for i in range(len(pts)):
            if np.any(pair_half[i] < min_val + 0.002):
                critical_mask[i] = True
        
        n_crit = np.sum(critical_mask)
        if n_crit == 0 or n_crit > n_critical:
            critical_mask = np.zeros(len(pts), dtype=bool)
            # Pick the n_critical most constrained points
            all_constraints = np.concatenate([pair_half.ravel(), boundary])
            # Map back to points
            point_constraints = np.full(len(pts), np.inf)
            for i in range(len(pts)):
                point_constraints[i] = min(boundary[i], pair_half[i].min())
            order = np.argsort(point_constraints)
            critical_mask[order[:n_critical]] = True
            n_crit = n_critical
        
        crit_idx = np.where(critical_mask)[0]
        if n_crit < 2:
            break
        
        # Optimize only critical points using gradient descent
        T = 0.001
        for _ in range(5):
            crit_pts = pts[crit_idx]
            x = crit_pts.ravel()
            
            # Compute gradient for critical points only
            # Simplified: use the full gradient but only update critical points
            obj, grad = _objective_and_grad(pts.ravel(), T)
            grad_pts = grad.reshape(-1, 2)
            
            # Update only critical points
            for ci in range(n_crit):
                idx = crit_idx[ci]
                pts[idx] -= step_size * grad_pts[idx]
            
            # Keep non-critical points fixed
            non_crit_idx = np.where(~critical_mask)[0]
            if len(non_crit_idx) > 0:
                pass  # They're already unchanged
            
            # Ensure validity
            new_norms = np.linalg.norm(pts, axis=1)
            mask = new_norms >= 1.0
            if np.any(mask):
                pts[mask] *= 0.999 / new_norms[mask, np.newaxis]
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple diverse initializations and keep the best
    best_pts = None
    best_r = -1.0
    
    seeds = [42, 0, 1, 2, 3]
    for seed in seeds:
        if time.time() > deadline - 60.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init(N, seed=42)
        best_r = _compute_radius(best_pts)
    
    x0 = best_pts.ravel()
    
    # Temperature annealing schedule - more steps at lower temperatures
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    
    for T in temperatures:
        if time.time() > deadline - 25.0:
            break
        try:
            # Use L-BFGS-B with gradient
            maxiter = 500 if T <= 0.001 else 300
            maxfun = 1500 if T <= 0.001 else 800
            result = minimize(
                lambda x: _objective_and_grad(x, T)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxls': 100,
                    'maxfun': maxfun,
                }
            )
            if time.time() > deadline:
                break
            new_pts = result.x.reshape(-1, 2)
            # Check validity
            norms = np.linalg.norm(new_pts, axis=1)
            if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                continue
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                x0 = new_pts.ravel()
        except Exception:
            continue
    
    # Tighten pass: directly push binding constraints
    if time.time() < deadline - 15.0:
        try:
            tightened = _tighten_min_constraint(best_pts, n_iter=150, step_size=0.0005)
            norms = np.linalg.norm(tightened, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(tightened)):
                tight_r = _compute_radius(tightened)
                if tight_r > best_r:
                    best_r = tight_r
                    best_pts = tightened.copy()
        except Exception:
            pass
    
    # Local refinement on critical points
    if time.time() < deadline - 12.0:
        try:
            refined = _local_refine(best_pts, n_iter=100, step_size=0.0008, n_critical=30)
            norms = np.linalg.norm(refined, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(refined)):
                ref_r = _compute_radius(refined)
                if ref_r > best_r:
                    best_r = ref_r
                    best_pts = refined.copy()
        except Exception:
            pass
    
    # Final high-precision polish with very low temperature
    if time.time() < deadline - 8.0:
        try:
            T_final = 0.00001
            result = minimize(
                lambda x: _objective_and_grad(x, T_final)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 3000,
                    'ftol': 1e-14,
                    'gtol': 1e-13,
                    'maxls': 200,
                    'maxfun': 8000,
                }
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Final tighten after polish
    if time.time() < deadline - 3.0:
        try:
            tightened = _tighten_min_constraint(best_pts, n_iter=100, step_size=0.0002)
            norms = np.linalg.norm(tightened, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(tightened)):
                tight_r = _compute_radius(tightened)
                if tight_r > best_r:
                    best_r = tight_r
                    best_pts = tightened.copy()
        except Exception:
            pass
    
    # Second local refinement pass
    if time.time() < deadline - 1.0:
        try:
            refined = _local_refine(best_pts, n_iter=50, step_size=0.0005, n_critical=20)
            norms = np.linalg.norm(refined, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(refined)):
                ref_r = _compute_radius(refined)
                if ref_r > best_r:
                    best_r = ref_r
                    best_pts = refined.copy()
        except Exception:
            pass
    
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask):
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
    
    return best_pts
