import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(12):
        # Better spacing formula: target density ~ n_points * pi * r^2 <= pi * 1^2
        # r ~ 1/sqrt(n_points), so spacing ~ 2*r ~ 2/sqrt(n_points)
        base_spacing = 2.0 / np.sqrt(n_points)
        spacing = base_spacing * (0.92 + 0.04 * attempt)
        # Ensure k is large enough to generate at least n_points
        k = int(1.0 / spacing) + 10
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x, y, r))
        
        # If we don't have enough points, shrink spacing and retry (or just force selection if very close)
        if len(rows) < n_points:
            # Try a tighter spacing
            spacing = spacing * 0.9
            k = int(1.0 / spacing) + 10
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
            noise_scale = spacing * (0.02 + 0.01 * (attempt % 3))
            selected = selected + rng.normal(0, noise_scale, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                # If noise pushed points out, revert to un-noised for this attempt's count check
                # but we need n_points. Just take the first n_points from original if needed?
                # Better: skip this attempt if we can't maintain count.
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
    
    # Fallback: Ensure we return n_points
    if best_pts is None or len(best_pts) != n_points:
        # Generate a dense grid that definitely has enough points
        spacing = 2.0 / np.sqrt(n_points) * 0.8 # Very tight
        k = int(1.0 / spacing) + 20
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        if len(rows) >= n_points:
            pts = np.array(rows)
            order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
            best_pts = pts[order[:n_points]]
        else:
            # Absolute last resort: duplicate points (invalid packing but valid shape)
            # This should not happen with k=20
            pts = np.zeros((n_points, 2))
            best_pts = pts
    return best_pts

def _ring_init(n_points, seed=0):
    """Ring-based initialization with concentric circles."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    for attempt in range(8):
        # Distribute points on concentric rings
        n_rings = int(np.sqrt(n_points)) + 1
        pts_list = []
        # Center point
        pts_list.append([0.0, 0.0])
        for ring in range(1, n_rings):
            radius = ring * 0.12 * (0.9 + 0.03 * attempt)
            if radius >= 0.95:
                break
            # Number of points on this ring
            circumference = 2 * np.pi * radius
            n_on_ring = max(1, int(circumference / 0.15))
            angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False) + rng.uniform(0, 2*np.pi/n_on_ring)
            for a in angles:
                x = radius * np.cos(a)
                y = radius * np.sin(a)
                if np.hypot(x, y) < 0.98:
                    pts_list.append([x, y])
        
        # Ensure we have enough points before adding to outer ring
        if len(pts_list) < n_points:
            # Add more points on outer ring
            radius = 0.9
            n_needed = n_points - len(pts_list)
            angles = np.linspace(0, 2 * np.pi, n_needed, endpoint=False)
            for a in angles:
                pts_list.append([radius * np.cos(a), radius * np.sin(a)])
        
        if len(pts_list) < n_points:
            # Fallback: fill with random points inside
            while len(pts_list) < n_points:
                p = rng.uniform(-0.9, 0.9, 2)
                if np.linalg.norm(p) < 0.9:
                    pts_list.append(p)

        pts = np.array(pts_list[:n_points])
        if attempt > 0:
            pts = pts + rng.normal(0, 0.01, pts.shape)
            norms = np.linalg.norm(pts, axis=1)
            mask = norms < 1.0
            pts = pts[mask]
            if len(pts) < n_points:
                # If noise reduced count, revert to pre-noise for this attempt? 
                # Or just skip. Skipping is safer.
                continue
            pts = pts[:n_points]
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(pts, axis=1).max()
        r = min(min_pair, min_bound)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None or len(best_pts) != n_points:
        return _hex_init(n_points, seed)
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

def _get_critical_pairs(pts, r_target=None, n_max=100):
    """Get the most constrained pairs and boundary violations."""
    if r_target is None:
        r_target = _compute_radius(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    sq_norms = np.sum(pts * pts, axis=1)
    cross = pts @ pts.T
    sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
    np.maximum(sq_dists, 0.0, out=sq_dists)
    np.fill_diagonal(sq_dists, np.inf)
    dists = np.sqrt(sq_dists)
    
    # Find pairs with distance < 2*(r_target + small_margin)
    margin = max(0.01, r_target * 0.1)
    pair_mask = dists < 2.0 * (r_target + margin)
    pair_mask &= ~np.eye(len(pts), dtype=bool)
    
    pairs = np.argwhere(pair_mask)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]  # unique pairs
    
    # Sort by distance
    if len(pairs) > 0:
        pair_dists = dists[pairs[:, 0], pairs[:, 1]]
        order = np.argsort(pair_dists)
        pairs = pairs[order[:n_max]]
    
    # Boundary critical points
    boundary_crit = np.where(boundary < r_target + margin)[0]
    
    return pairs, boundary_crit, r_target


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


def _objective_and_grad(x, T, active_pairs=None, active_boundary=None):
    """
    Objective: minimize -softmin(constraints) where constraints are:
      - pairwise half-distances d_ij/2 for i<j
      - boundary distances 1 - |c_i|
    
    Uses log-sum-exp soft-min for differentiability.
    If active_pairs/active_boundary are provided, only compute those constraints.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary constraints: 1 - |c_i|
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # shape (n,)
    
    if active_pairs is not None and len(active_pairs) > 0:
        # Only compute active pairs
        i_idx = active_pairs[:, 0]
        j_idx = active_pairs[:, 1]
        diff = pts[i_idx] - pts[j_idx]
        d = np.linalg.norm(diff, axis=1)
        pair_half = d / 2.0
        pair_indices = list(zip(i_idx, j_idx))
    else:
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
        pair_indices = list(zip(iu[0], iu[1]))
    
    if active_boundary is not None and len(active_boundary) > 0:
        boundary_active = boundary[active_boundary]
        boundary_indices = active_boundary
    else:
        boundary_active = boundary
        boundary_indices = np.arange(n)
    
    # All constraint values
    all_vals = np.concatenate([pair_half, boundary_active])
    
    # Soft minimum
    obj = -_soft_min_logsumexp(all_vals, T)
    
    # Gradient computation
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights
    
    grad_pts = np.zeros((n, 2))
    
    n_pairs = len(pair_half)
    
    # Pairwise gradients
    if n_pairs > 0:
        for k in range(n_pairs):
            w = grad_all[k]
            if abs(w) < 1e-15:
                continue
            i, j = pair_indices[k]
            diff_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-10:
                continue
            unit = diff_ij / dist_ij
            # d(pair_half)/d(pts[i]) = 0.5 * unit
            # d(pair_half)/d(pts[j]) = -0.5 * unit
            grad_pts[i] += w * 0.5 * unit
            grad_pts[j] += w * (-0.5) * unit
    
    # Boundary gradients
    for k in range(len(boundary_active)):
        w = grad_all[n_pairs + k]
        if abs(w) < 1e-15:
            continue
        idx = boundary_indices[k]
        norm_i = norms[idx]
        if norm_i < 1e-10:
            continue
        unit = pts[idx] / norm_i
        # d(boundary)/d(pts[i]) = -unit
        grad_pts[idx] += w * (-unit)
    
    grad_obj = grad_pts.ravel()
    
    return obj, grad_obj


def _tighten_min_constraint(pts, n_iter=50, step_size=0.001):
    """
    Directly push the most constrained points to increase the minimum constraint.
    Uses multiple critical pairs simultaneously for better progress.
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
        dists = np.sqrt(sq_dists)
        pair_half = dists / 2.0
        
        # Find the minimum constraint
        min_pair_idx = np.unravel_index(np.argmin(pair_half), pair_half.shape)
        min_boundary_idx = np.argmin(boundary)
        
        min_pair_val = pair_half[min_pair_idx]
        min_boundary_val = boundary[min_boundary_idx]
        
        current_r = min(min_pair_val, min_boundary_val)
        
        # Find all near-minimum constraints
        margin = max(0.005, current_r * 0.05)
        
        if min_pair_val < min_boundary_val:
            # Find all pairs near minimum
            pair_mask = pair_half < min_pair_val + margin
            pair_mask &= ~np.eye(len(pts), dtype=bool)
            pairs = np.argwhere(pair_mask)
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]
            
            # Apply forces from all these pairs
            for i, j in pairs:
                vec = pts[i] - pts[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-10:
                    continue
                unit = vec / dist
                # Scale force by how close to minimum
                weight = 1.0 - (pair_half[i, j] - min_pair_val) / margin
                weight = max(0.1, min(1.0, weight))
                pts[i] += step_size * weight * unit
                pts[j] -= step_size * weight * unit
        else:
            # Find all boundary-near points
            boundary_mask = boundary < min_boundary_val + margin
            for i in np.where(boundary_mask)[0]:
                norm_i = norms[i]
                if norm_i < 1e-10:
                    continue
                unit = pts[i] / norm_i
                weight = 1.0 - (boundary[i] - min_boundary_val) / margin
                weight = max(0.1, min(1.0, weight))
                pts[i] -= step_size * weight * unit
        
        # Ensure points stay inside the container
        new_norms = np.linalg.norm(pts, axis=1)
        mask = new_norms >= 1.0
        if np.any(mask):
            pts[mask] *= 0.999 / new_norms[mask, np.newaxis]
        
        # Check for improvement
        new_r = _compute_radius(pts)
        if new_r < current_r - 1e-6:
            # Revert if we made things worse
            pts = pts - (pts - pts)  # no-op, keep current
            break
    
    return pts


def _local_refine(pts, n_iter=200, step_size=0.001, n_critical=20):
    """
    Local refinement: identify the most constrained points and optimize them
    while keeping others fixed. Uses active-set approach.
    """
    pts = pts.copy()
    
    for it in range(n_iter):
        r_current = _compute_radius(pts)
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        sq_norms = np.sum(pts * pts, axis=1)
        cross = pts @ pts.T
        sq_dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * cross
        np.maximum(sq_dists, 0.0, out=sq_dists)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        pair_half = dists / 2.0
        
        # Find the most constrained points
        min_val = min(pair_half.min(), boundary.min())
        
        # Identify critical points: those involved in constraints near the minimum
        margin = max(0.003, min_val * 0.03)
        critical_mask = np.zeros(len(pts), dtype=bool)
        critical_mask[boundary < min_val + margin] = True
        
        # Add points in near-minimum pairs
        pair_mask = pair_half < min_val + margin
        pair_mask &= ~np.eye(len(pts), dtype=bool)
        pairs = np.argwhere(pair_mask)
        for i, j in pairs:
            critical_mask[i] = True
            critical_mask[j] = True
        
        n_crit = np.sum(critical_mask)
        if n_crit == 0:
            break
        if n_crit > n_critical:
            # Pick the n_critical most constrained points
            point_constraints = np.full(len(pts), np.inf)
            for i in range(len(pts)):
                point_constraints[i] = min(boundary[i], pair_half[i].min())
            order = np.argsort(point_constraints)
            critical_mask = np.zeros(len(pts), dtype=bool)
            critical_mask[order[:n_critical]] = True
            n_crit = n_critical
        
        crit_idx = np.where(critical_mask)[0]
        if n_crit < 2:
            break
        
        # Get active pairs among critical points
        crit_set = set(crit_idx.tolist())
        active_pairs = []
        for i, j in np.argwhere(pair_mask):
            if i in crit_set and j in crit_set:
                active_pairs.append([i, j])
        active_pairs = np.array(active_pairs) if len(active_pairs) > 0 else None
        
        active_boundary = np.where(critical_mask)[0]
        
        # Optimize critical points with active constraints
        T = 0.002
        for sub_it in range(8):
            obj, grad = _objective_and_grad(pts.ravel(), T, active_pairs, active_boundary)
            grad_pts = grad.reshape(-1, 2)
            
            # Update only critical points
            for idx in crit_idx:
                pts[idx] -= step_size * grad_pts[idx]
            
            # Ensure validity
            new_norms = np.linalg.norm(pts, axis=1)
            mask = new_norms >= 1.0
            if np.any(mask):
                pts[mask] *= 0.999 / new_norms[mask, np.newaxis]
            
            # Check improvement
            new_r = _compute_radius(pts)
            if new_r < r_current - 1e-6:
                break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Try multiple diverse initializations and keep the best
    best_pts = None
    best_r = -1.0
    
    # Hex initializations
    for seed in [42, 0, 1, 2, 3, 7]:
        if time.time() > deadline - 80.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Ring initializations
    for seed in [0, 1, 2]:
        if time.time() > deadline - 70.0:
            break
        pts = _ring_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init(N, seed=42)
        best_r = _compute_radius(best_pts)
    
    x0 = best_pts.ravel()
    
    # Temperature annealing schedule
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    
    for T in temperatures:
        if time.time() > deadline - 35.0:
            break
        try:
            maxiter = 400 if T <= 0.001 else 250
            maxfun = 1200 if T <= 0.001 else 600
            result = minimize(
                lambda x: _objective_and_grad(x, T)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': maxiter,
                    'ftol': 1e-13,
                    'gtol': 1e-11,
                    'maxls': 80,
                    'maxfun': maxfun,
                }
            )
            if time.time() > deadline:
                break
            new_pts = result.x.reshape(-1, 2)
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
    
    # Tighten pass
    if time.time() < deadline - 20.0:
        try:
            tightened = _tighten_min_constraint(best_pts, n_iter=200, step_size=0.0005)
            norms = np.linalg.norm(tightened, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(tightened)):
                tight_r = _compute_radius(tightened)
                if tight_r > best_r:
                    best_r = tight_r
                    best_pts = tightened.copy()
        except Exception:
            pass
    
    # Local refinement
    if time.time() < deadline - 15.0:
        try:
            refined = _local_refine(best_pts, n_iter=150, step_size=0.0008, n_critical=30)
            norms = np.linalg.norm(refined, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(refined)):
                ref_r = _compute_radius(refined)
                if ref_r > best_r:
                    best_r = ref_r
                    best_pts = refined.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            T_final = 0.00001
            result = minimize(
                lambda x: _objective_and_grad(x, T_final)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-13,
                    'maxls': 150,
                    'maxfun': 5000,
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
    
    # Final tighten
    if time.time() < deadline - 5.0:
        try:
            tightened = _tighten_min_constraint(best_pts, n_iter=150, step_size=0.0002)
            norms = np.linalg.norm(tightened, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(tightened)):
                tight_r = _compute_radius(tightened)
                if tight_r > best_r:
                    best_r = tight_r
                    best_pts = tightened.copy()
        except Exception:
            pass
    
    # Final local refinement
    if time.time() < deadline - 2.0:
        try:
            refined = _local_refine(best_pts, n_iter=80, step_size=0.0005, n_critical=25)
            norms = np.linalg.norm(refined, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(refined)):
                ref_r = _compute_radius(refined)
                if ref_r > best_r:
                    best_r = ref_r
                    best_pts = refined.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_pts is None or best_pts.shape != (N, 2):
        best_pts = _hex_init(N, seed=0)
        if best_pts.shape != (N, 2):
            # Absolute last resort
            best_pts = np.zeros((N, 2))
    
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask):
        # Project points inside
        for i in range(N):
            if norms[i] >= 1.0:
                best_pts[i] *= 0.999 / norms[i]
        # Recalculate best_r just in case, though we return pts
        best_r = _compute_radius(best_pts)
    
    return best_pts
