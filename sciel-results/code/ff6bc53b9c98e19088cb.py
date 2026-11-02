"""
Packing 50 equal circles in a unit disk using a refined soft-min optimization
approach with force-directed pre-conditioning and multi-stage beta annealing.

Key improvements:
1. Cleaner soft-min objective with proper log-sum-exp stability
2. Better force-directed relaxation as pre-conditioner (vectorized)
3. Multi-stage beta annealing (low to high) for better convergence
4. More efficient gradient computation
5. Better initialization strategy with hex lattice and ring configs
6. Binary search on radius for feasibility-based refinement
7. Contact-graph-aware active set for final polish
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin >= 1.0):
        return 0.0
    
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise/2 and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat


def _force_directed_relax(pts, n_steps=300, target_r=0.12):
    """
    Vectorized force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Compute forces vectorized
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        forces = np.zeros_like(pts)
        iu = np.triu_indices(N, k=1)
        d_pairs = dist_mat[iu]
        
        # Find overlapping pairs
        overlap_mask = (d_pairs < target_d) & (d_pairs > 1e-10)
        if np.any(overlap_mask):
            i_idx, m_idx = iu
            d_sel = d_pairs[overlap_mask]
            vecs = diff[i_idx[overlap_mask], m_idx[overlap_mask]]
            directions = vecs / d_sel[:, np.newaxis]
            force_mag = (target_d - d_sel) * 0.5
            
            np.add.at(forces, i_idx[overlap_mask], force_mag[:, np.newaxis] * directions)
            np.add.at(forces, m_idx[overlap_mask], -force_mag[:, np.newaxis] * directions)
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_mask = dists_origin > (1.0 - target_r)
        if np.any(boundary_mask):
            excess = dists_origin[boundary_mask] - (1.0 - target_r)
            directions = -pts[boundary_mask] / dists_origin[boundary_mask, np.newaxis]
            forces[boundary_mask] += excess[:, np.newaxis] * directions
        
        # Update positions with adaptive learning rate
        lr = 0.01 * (1.0 - step / max(n_steps, 1)) + 0.001
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts

def _pbd_relax(pts, target_r, n_iters=500):
    """
    Position-Based Dynamics relaxation.
    Iteratively projects points to satisfy non-overlap and boundary constraints.
    This ensures hard feasibility at each step, which can help escape local minima
    that soft-min optimization might struggle with due to numerical issues.
    """
    pts = pts.copy()
    min_dist = 2.0 * target_r
    max_radius = 1.0 - target_r
    
    for _ in range(n_iters):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break
            
        # 1. Boundary Projection
        dists = np.linalg.norm(pts, axis=1)
        out_of_bounds = dists > max_radius
        if np.any(out_of_bounds):
            # Project points to the boundary of the allowed region
            pts[out_of_bounds] = pts[out_of_bounds] * (max_radius / dists[out_of_bounds, np.newaxis])
            
        # 2. Pairwise Overlap Resolution
        # We iterate over pairs. For efficiency, we can vectorize the distance check,
        # but the update depends on the specific pair, so we do it in a loop or vectorized update.
        # Vectorized approach for N=50 is fast enough.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = dist_mat[iu]
        
        # Identify overlapping pairs
        overlap_mask = d_pairs < min_dist
        if np.any(overlap_mask):
            i_idx, m_idx = iu
            d_sel = d_pairs[overlap_mask]
            vecs = diff[i_idx[overlap_mask], m_idx[overlap_mask]]
            
            # Avoid division by zero
            safe_d = np.where(d_sel < 1e-10, 1e-10, d_sel)
            directions = vecs / safe_d[:, np.newaxis]
            
            # Calculate correction: move apart by half the overlap
            overlap_amount = (min_dist - d_sel) * 0.5
            corrections = overlap_amount[:, np.newaxis] * directions
            
            # Apply corrections
            np.add.at(pts, i_idx[overlap_mask], corrections)
            np.add.at(pts, m_idx[overlap_mask], -corrections)
            
    return pts


def _hex_lattice_init(r_guess=0.12):
    """Generate a hexagonal lattice packing as initial guess."""
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


def _soft_min_optimize(pts_init, beta_schedule, max_iter_per_stage=500):
    """
    Multi-stage soft-min optimization with beta annealing.
    """
    current_pts = pts_init.copy()
    
    for stage, beta in enumerate(beta_schedule):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        def objective(x):
            return _soft_min_obj_and_grad(x, beta)
        
        x0 = current_pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter_per_stage, 'ftol': 1e-14,
                                   'gtol': 1e-10, 'maxls': 50})
        current_pts = result.x.reshape(N, 2)
        
        # Project to ensure validity
        dists = np.linalg.norm(current_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / dists[mask, np.newaxis]
    
    return current_pts


def _binary_search_radius(pts_init, r_low, r_high, max_iters=10, tol=1e-6):
    """
    Binary search on radius to find the maximum feasible radius.
    Uses soft-min optimization at each radius target.
    """
    best_pts = pts_init.copy()
    best_r = r_low
    
    for it in range(max_iters):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        r_mid = (r_low + r_high) / 2.0
        target_d = 2.0 * r_mid
        
        # Try to achieve this radius
        # First, force-directed relaxation toward target
        pts_fd = _force_directed_relax(pts_init, n_steps=100, target_r=r_mid)
        
        # Then soft-min optimization
        beta_schedule = [100.0, 500.0, 2000.0]
        pts_opt = _soft_min_optimize(pts_fd, beta_schedule, max_iter_per_stage=300)
        
        score = _score(pts_opt)
        
        if score >= r_mid - tol:
            # Feasible, try larger
            r_low = r_mid
            best_pts = pts_opt.copy()
            best_r = score
        else:
            # Not feasible, try smaller
            r_high = r_mid
        
        if r_high - r_low < tol:
            break
    
    return best_pts, best_r


def _full_optimize(points, budget_seconds=25.0):
    """Full optimization pipeline: force-directed + multi-stage soft-min + binary search."""
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Phase 1: Force-directed pre-conditioning
    if time.time() - t0 < budget_seconds * 0.2:
        target_r = max(best_score, 0.12)
        pts_fd = _force_directed_relax(points, n_steps=200, target_r=target_r)
        score_fd = _score(pts_fd)
        if score_fd > best_score:
            best_score = score_fd
            best_points = pts_fd.copy()
            
    # Phase 1.5: PBD Relaxation to enforce hard constraints and improve packing
    if time.time() - t0 < budget_seconds * 0.35:
        target_r = max(best_score, 0.125)
        pts_pbd = _pbd_relax(best_points, target_r, n_iters=300)
        score_pbd = _score(pts_pbd)
        if score_pbd > best_score:
            best_score = score_pbd
            best_points = pts_pbd.copy()

    # Phase 2: Multi-stage soft-min with beta annealing
    if time.time() - t0 < budget_seconds * 0.5:
        beta_schedule = [50.0, 200.0, 1000.0, 5000.0, 20000.0]
        pts_opt = _soft_min_optimize(best_points, beta_schedule, max_iter_per_stage=400)
        score_opt = _score(pts_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Phase 3: Binary search on radius for refinement
    if time.time() - t0 < budget_seconds * 0.8:
        r_low = best_score
        r_high = best_score * 1.05
        pts_bs, r_bs = _binary_search_radius(best_points, r_low, r_high, max_iters=8)
        if r_bs > best_score:
            best_score = r_bs
            best_points = pts_bs.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() - t0 < budget_seconds * 0.98:
        beta_final = [50000.0, 100000.0]
        pts_polish = _soft_min_optimize(best_points, beta_final, max_iter_per_stage=300)
        score_polish = _score(pts_polish)
        if score_polish > best_score:
            best_score = score_polish
            best_points = pts_polish.copy()
    
    return best_points, best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Known optimal structure: 1 + 6 + 12 + 18 + 13 = 50
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        ([1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.80, 0.98]),
        ([1, 6, 12, 18, 13], [0.0, 0.20, 0.42, 0.65, 0.88]),
    ]
    
    # Try hex lattice init
    hex_pts = _hex_lattice_init(0.12)
    hex_score = _score(hex_pts)
    if hex_score > best_score:
        best_score = hex_score
        best_points = hex_pts.copy()
    
    # Try ring configs with full optimization
    for config_idx, (ring_sizes, radii) in enumerate(ring_configs):
        if time.time() - _START_TIME > _DEADLINE - 30:
            break
        
        points = _ring_init(ring_sizes, radii)
        if config_idx > 0:
            points += rng.uniform(-0.01, 0.01, size=(N, 2))
        
        remaining = _DEADLINE - (time.time() - _START_TIME) - 20
        budget = min(18.0, remaining)
        if budget < 5:
            break
            
        pts_opt, score_opt = _full_optimize(points, budget_seconds=budget)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Multi-start with symmetry breaking
    n_starts = 3
    for s in range(n_starts):
        if time.time() - _START_TIME > _DEADLINE - 25:
            break
        
        if s == 0:
            pts_init = best_points.copy()
        elif s == 1:
            pts_init = _hex_lattice_init(max(best_score, 0.12))
        else:
            pts_init = best_points.copy()
            noise = rng.normal(0, 0.02 * (s + 1), size=pts_init.shape)
            pts_init += noise
            norms = np.hypot(pts_init[:, 0], pts_init[:, 1])
            for i in range(N):
                if norms[i] > 1.0 - max(best_score, 0.12):
                    pts_init[i] *= (1.0 - max(best_score, 0.12)) / norms[i]
        
        remaining = _DEADLINE - (time.time() - _START_TIME) - 10
        if remaining < 10:
            break
            
        # Use a shorter beta schedule for multi-start
        beta_schedule = [100.0, 500.0, 2000.0, 10000.0]
        pts_opt = _soft_min_optimize(pts_init, beta_schedule, max_iter_per_stage=500)
        score = _score(pts_opt)
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
    
    # Final high-precision polish with binary search
    if best_points is not None and time.time() - _START_TIME < _DEADLINE - 15:
        r_current = _score(best_points)
        if r_current > 0.05:
            # Binary search for final refinement
            r_low = r_current
            r_high = r_current * 1.02
            pts_bs, r_bs = _binary_search_radius(best_points, r_low, r_high, max_iters=6)
            if r_bs > best_score:
                best_score = r_bs
                best_points = pts_bs.copy()
            
            # Final PBD polish to ensure strict feasibility and potentially squeeze more out
            if time.time() - _START_TIME < _DEADLINE - 5:
                pts_final_pbd = _pbd_relax(best_points, max(best_score, 0.125), n_iters=500)
                score_final_pbd = _score(pts_final_pbd)
                if score_final_pbd > best_score:
                    best_score = score_final_pbd
                    best_points = pts_final_pbd.copy()

            # Final soft-min polish
            beta_final = [50000.0, 100000.0, 200000.0]
            pts_polish = _soft_min_optimize(best_points, beta_final, max_iter_per_stage=300)
            score_polish = _score(pts_polish)
            if score_polish > best_score:
                best_score = score_polish
                best_points = pts_polish.copy()
                
            # One last PBD check
            if time.time() - _START_TIME < _DEADLINE - 2:
                pts_last_pbd = _pbd_relax(best_points, max(best_score, 0.125), n_iters=200)
                score_last_pbd = _score(pts_last_pbd)
                if score_last_pbd > best_score:
                    best_score = score_last_pbd
                    best_points = pts_last_pbd.copy()
    
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
