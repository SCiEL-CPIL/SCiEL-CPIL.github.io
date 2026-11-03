import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    for attempt in range(15):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.92 + 0.02 * attempt)
        
        k = int(1.0 / spacing) + 6
        lattice_pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                lattice_pts.append((x, y))
        lattice_pts = np.array(lattice_pts)
        
        angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotated = np.column_stack([
            lattice_pts[:, 0] * cos_a - lattice_pts[:, 1] * sin_a,
            lattice_pts[:, 0] * sin_a + lattice_pts[:, 1] * cos_a
        ])
        
        norms = np.linalg.norm(rotated, axis=1)
        margin = spacing * 0.5
        mask = norms <= 1.0 - margin
        if mask.sum() < n_points:
            continue
        
        candidates = rotated[mask]
        cand_norms = norms[mask]
        
        if attempt > 0:
            candidates = candidates + rng.normal(0, spacing * 0.03, candidates.shape)
            norms2 = np.linalg.norm(candidates, axis=1)
            valid = norms2 < 1.0
            if valid.sum() < n_points:
                continue
            candidates = candidates[valid]
            cand_norms = norms2[valid]
        
        if len(candidates) > n_points:
            order = np.argsort(cand_norms, kind="stable")
            selected = candidates[order[:n_points]]
        else:
            selected = candidates
        
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
            
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        k = int(1.0 / spacing) + 6
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
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
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

def _objective_and_grad(x, T, edge_indices=None):
    """
    Objective: minimize -softmin(constraints)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    if edge_indices is None:
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
    else:
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])
        
    all_vals = np.concatenate([pair_half, boundary])
    
    obj = -_soft_min_logsumexp(all_vals, T)
    
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
        
    grad_pts = np.zeros_like(pts)
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary
    
    if edge_indices is None:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        W = np.zeros((n, n))
        n_pairs = len(pair_half)
        iu = np.triu_indices(n, k=1)
        W[iu] = grad_all[:n_pairs]
        W[iu[1], iu[0]] = grad_all[:n_pairs]
        
        weighted_unit = W[:, :, np.newaxis] * unit_vecs
        grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
        grad_pts += grad_pairs
    else:
        for k, (i, j) in enumerate(edge_indices):
            w = grad_all[k]
            if w == 0: continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10: continue
            unit_vec = diff / d
            grad_pts[i] += 0.5 * w * unit_vec
            grad_pts[j] -= 0.5 * w * unit_vec
            
    return obj, grad_pts.ravel()

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.008
    cooldown = 0.9995
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        target = 2.0 * r_est
        too_close = dists < target * 1.5
        safe_dists = np.where(too_close, dists, 1.0)
        
        excess = np.where(too_close, target - dists, 0.0)
        force_mag = np.where(too_close, excess / safe_dists, 0.0)
        
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.08
        in_wall = norms > (1.0 - wall_margin)
        if np.any(in_wall):
            excess_b = np.where(in_wall, norms - (1.0 - wall_margin), 0.0)
            f_mag = 15.0 * excess_b
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            boundary_force = -pos / safe_norms[:, np.newaxis] * f_mag[:, np.newaxis]
            forces += boundary_force
        
        pos = pos + lr * forces
        
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.998
            
        if step % 50 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = min(r, 0.06)
                
        lr *= cooldown
        
    r_final = _compute_radius(pos)
    if r_final > best_r:
        best_r = r_final
        best_pos = pos.copy()
        
    return best_pos, best_r

def _get_active_edges(pts, k=8):
    """Get the k nearest neighbors for each point to form sparse edges."""
    n = len(pts)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    edges = set()
    for i in range(n):
        nn = np.argsort(d[i])[:k]
        for j in nn:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
                
    return list(edges)

def _local_polish(pts, deadline):
    """High-precision local polish using sparse gradient and tight tolerances."""
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    for polish_iter in range(5):
        if time.time() > deadline:
            break
            
        try:
            tri = Delaunay(best_pts)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            edge_list = list(edges)
        except Exception:
            edge_list = _get_active_edges(best_pts, k=6)
            
        if not edge_list:
            break
            
        temperatures = [0.00005, 0.00002, 0.00001, 0.000005]
        x0 = best_pts.ravel()
        
        for T in temperatures:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T, edge_list)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T, edge_list)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 300,
                        'ftol': 1e-15,
                        'gtol': 1e-14,
                        'maxls': 100,
                        'maxfun': 800,
                    }
                )
                new_pts = result.x.reshape(-1, 2)
                norms = np.linalg.norm(new_pts, axis=1)
                if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                    new_r = _compute_radius(new_pts)
                    if new_r > best_r:
                        best_r = new_r
                        best_pts = new_pts.copy()
                        x0 = new_pts.ravel()
            except Exception:
                pass
                
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Initialization and Force Relaxation
    seeds = [42, 123, 7, 999, 555, 314, 271]
    for seed in seeds:
        if time.time() > deadline - 55.0:
            break
            
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=1500, deadline=deadline - 50.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r
            
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
        
    # Phase 2: Global Optimization with Soft-Min
    if time.time() < deadline - 25.0:
        temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        x0 = best_pts.ravel()
        
        for T in temperatures:
            if time.time() > deadline - 20.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-13,
                        'gtol': 1e-12,
                        'maxls': 60,
                        'maxfun': 500,
                    }
                )
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
                
    # Phase 3: Local Polish with Sparse Gradient using Delaunay
    if time.time() < deadline - 8.0 and best_pts is not None:
        polished_pts, polished_r = _local_polish(best_pts, deadline - 3.0)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
            
    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        
    return best_pts
