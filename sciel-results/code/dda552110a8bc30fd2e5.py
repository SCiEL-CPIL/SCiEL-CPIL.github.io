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
        base_spacing = 2.0 / np.sqrt(n_points) * 0.95
        spacing = base_spacing * (0.92 + 0.015 * attempt)
        
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                angle = rng.uniform(0, 2*np.pi)
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))
        
        if len(rows) < n_points:
            continue
            
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        
        if attempt > 0:
            noise_scale = spacing * 0.03
            selected = selected + rng.normal(0, noise_scale, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
            
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
            
    if best_pts is None:
        spacing = 2.0 / np.sqrt(n_points) * 0.95
        k = int(1.0 / spacing) + 5
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

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation with adaptive parameters."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.02
    cooldown = 0.998
    wall_margin = 0.08
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        target = 2.0 * r_est
        mask = dists < target
        
        safe_dists = np.where(mask, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        excess = np.where(mask, target - dists, 0.0)
        force_mag = 3.0 * excess / safe_dists
        
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        norms = np.linalg.norm(pos, axis=1)
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 15.0 * excess_b
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction
        
        pos = pos + lr * forces
        
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999
            
        if step % 20 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r
                lr = min(0.05, lr * 1.1)
                
        lr *= cooldown
        
    return best_pos, best_r

def _epigraph_obj_and_grad(z, n):
    """
    Epigraph formulation: maximize t subject to pairwise and boundary constraints.
    z = [t, x_1, y_1, ..., x_n, y_n]
    Objective: -t + 0.5 * (sum of squared violations)
    """
    t = z[0]
    p = z[1:].reshape(n, 2)
    
    # Project to container
    r = np.hypot(p[:, 0], p[:, 1])
    mask = r > 1.0
    if np.any(mask):
        p[mask] = p[mask] / r[mask, np.newaxis]
        r[mask] = 1.0
        
    # Pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1e10)
    d = np.sqrt(d2)
    
    # Boundary distances
    b = 1.0 - r
    
    # Violations
    viol_pair = np.maximum(0.0, 2.0 * t - d)
    viol_bound = np.maximum(0.0, t - b)
    
    obj = -t + 0.5 * (np.sum(viol_pair * viol_pair) + np.sum(viol_bound * viol_bound))
    
    # Gradient
    grad = np.zeros_like(z)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vec = diff / d[:, :, np.newaxis]
    unit_vec = np.where(np.isfinite(unit_vec), unit_vec, 0.0)
    
    w_pair = 2.0 * viol_pair
    grad_p_pair = np.sum(w_pair[:, :, np.newaxis] * unit_vec, axis=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inward_vec = p / r[:, np.newaxis]
    inward_vec = np.where(np.isfinite(inward_vec), inward_vec, 0.0)
    
    grad_p_bound = viol_bound[:, np.newaxis] * inward_vec
    
    grad_p = -grad_p_pair + grad_p_bound
    
    grad[0] = -1.0 + 4.0 * np.sum(viol_pair) + np.sum(viol_bound)
    grad[1:] = grad_p.reshape(-1)
    
    return obj, grad

def _epigraph_optimize(pts, t_init, deadline):
    """Optimize using epigraph formulation with L-BFGS-B."""
    n = len(pts)
    pts = pts.copy()
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, np.newaxis]
        
    best_pts = pts.copy()
    best_score = _compute_radius(best_pts)
    
    z0 = np.concatenate(([t_init], pts.flatten()))
    bounds = [(t_init * 0.5, 1.0)] + [(-1.0, 1.0)] * (2 * n)
    
    current_z = z0.copy()
    max_iter_per_call = 200
    
    while True:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                lambda z: _epigraph_obj_and_grad(z, n)[0],
                current_z,
                jac=lambda z: _epigraph_obj_and_grad(z, n)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': max_iter_per_call,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'maxfun': 800
                }
            )
        except Exception:
            break
            
        if not res.success and res.status != 0:
            break
            
        current_z = res.x
        new_pts = current_z[1:].reshape(n, 2)
        r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        out_mask = r > 1.0
        if np.any(out_mask):
            new_pts[out_mask] = new_pts[out_mask] / r[out_mask, np.newaxis]
            
        new_score = _compute_radius(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            # Update t_init to current best
            t_init = best_score * 1.01
            bounds[0] = (t_init * 0.5, 1.0)
        else:
            # No improvement, try one more time with tighter tolerance
            if time.time() > deadline - 5:
                break
            max_iter_per_call = max(50, max_iter_per_call // 2)
            
    return best_pts, best_score

def _sparse_epigraph_obj_and_grad(z, n, edge_list):
    """
    Sparse epigraph formulation using only Delaunay edges for pairwise constraints.
    """
    t = z[0]
    p = z[1:].reshape(n, 2)
    
    r = np.hypot(p[:, 0], p[:, 1])
    mask = r > 1.0
    if np.any(mask):
        p[mask] = p[mask] / r[mask, np.newaxis]
        r[mask] = 1.0
        
    # Sparse pairwise distances
    pair_half = np.zeros(len(edge_list))
    for k, (i, j) in enumerate(edge_list):
        diff = p[i] - p[j]
        pair_half[k] = np.linalg.norm(diff) / 2.0
        
    b = 1.0 - r
    
    viol_pair = np.maximum(0.0, 2.0 * t - 2.0 * pair_half)
    viol_bound = np.maximum(0.0, t - b)
    
    obj = -t + 0.5 * (np.sum(viol_pair * viol_pair) + np.sum(viol_bound * viol_bound))
    
    grad = np.zeros_like(z)
    grad[0] = -1.0 + 4.0 * np.sum(viol_pair) + np.sum(viol_bound)
    
    grad_p = np.zeros_like(p)
    
    # Boundary gradient
    with np.errstate(divide='ignore', invalid='ignore'):
        inward_vec = p / r[:, np.newaxis]
    inward_vec = np.where(np.isfinite(inward_vec), inward_vec, 0.0)
    grad_p += viol_bound[:, np.newaxis] * inward_vec
    
    # Pairwise gradient (sparse)
    for k, (i, j) in enumerate(edge_list):
        w = 2.0 * viol_pair[k]
        if w == 0:
            continue
        diff = p[i] - p[j]
        d = np.linalg.norm(diff)
        if d < 1e-10:
            continue
        unit_vec = diff / d
        grad_p[i] -= 0.5 * w * unit_vec
        grad_p[j] += 0.5 * w * unit_vec
        
    grad[1:] = grad_p.reshape(-1)
    
    return obj, grad

def _sparse_epigraph_optimize(pts, t_init, edge_list, deadline):
    """Optimize using sparse epigraph formulation."""
    n = len(pts)
    pts = pts.copy()
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, np.newaxis]
        
    best_pts = pts.copy()
    best_score = _compute_radius(best_pts)
    
    z0 = np.concatenate(([t_init], pts.flatten()))
    bounds = [(t_init * 0.5, 1.0)] + [(-1.0, 1.0)] * (2 * n)
    
    current_z = z0.copy()
    max_iter_per_call = 300
    
    while True:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                lambda z: _sparse_epigraph_obj_and_grad(z, n, edge_list)[0],
                current_z,
                jac=lambda z: _sparse_epigraph_obj_and_grad(z, n, edge_list)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': max_iter_per_call,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxls': 150,
                    'maxfun': 1200
                }
            )
        except Exception:
            break
            
        if not res.success and res.status != 0:
            break
            
        current_z = res.x
        new_pts = current_z[1:].reshape(n, 2)
        r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        out_mask = r > 1.0
        if np.any(out_mask):
            new_pts[out_mask] = new_pts[out_mask] / r[out_mask, np.newaxis]
            
        new_score = _compute_radius(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            t_init = best_score * 1.01
            bounds[0] = (t_init * 0.5, 1.0)
        else:
            if time.time() > deadline - 3:
                break
            max_iter_per_call = max(50, max_iter_per_call // 2)
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Multiple initializations with force relaxation
    inits = [
        _hex_init(N, seed=42),
        _hex_init(N, seed=123),
        _hex_init(N, seed=999),
        _hex_init(N, seed=555),
        _hex_init(N, seed=777)
    ]
    
    for pts in inits:
        if time.time() > deadline - 50.0:
            break
            
        r = _compute_radius(pts)
        
        # Force relaxation
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=1000, deadline=deadline - 40.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r
            
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
        
    # Phase 2: Epigraph optimization (full pairwise)
    if time.time() < deadline - 25.0:
        t_init = best_r * 0.99
        ep_pts, ep_r = _epigraph_optimize(best_pts, t_init, deadline - 20.0)
        
        if ep_r > best_r:
            best_r = ep_r
            best_pts = ep_pts.copy()
            
    # Phase 3: Sparse epigraph polish using Delaunay edges
    if time.time() < deadline - 10.0 and best_pts is not None:
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
            edge_list = []
            
        if edge_list:
            t_init = best_r * 0.995
            sparse_pts, sparse_r = _sparse_epigraph_optimize(
                best_pts, t_init, edge_list, deadline - 5.0
            )
            
            if sparse_r > best_r:
                best_r = sparse_r
                best_pts = sparse_pts.copy()
                
    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        
    return best_pts
