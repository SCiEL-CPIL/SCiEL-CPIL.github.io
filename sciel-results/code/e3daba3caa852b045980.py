import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300

def _get_active_edges(pts, r_target, max_edges=150):
    """Get the most constrained edges for sparse optimization."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    # Find pairs that are close to the target distance
    target = 2.0 * r_target
    # Get all pairs sorted by distance
    iu = np.triu_indices(len(pts), k=1)
    pair_dists = d[iu]
    # Sort by distance (closest first)
    order = np.argsort(pair_dists, kind='stable')
    # Take the closest pairs up to max_edges
    n_edges = min(max_edges, len(order))
    edges = [(iu[0][order[k]], iu[1][order[k]]) for k in range(n_edges)]
    return edges

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Try a few different spacings and rotations
    for attempt in range(15):
        # Spacing estimation - tune more carefully
        # For N circles in unit disk, optimal r ~ 0.95/sqrt(N) for hex packing
        # But we need to account for boundary effects
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.92
        # Vary spacing slightly
        spacing = base_spacing * (0.93 + 0.015 * attempt)
        
        k = int(1.0 / spacing) + 6
        rows = []
        
        # Generate hex lattice points
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                # Only keep points that fit with some margin
                if r <= 1.0 - spacing / 2.0 - 0.01:
                    rows.append((x, y, r))
        
        if len(rows) < n_points:
            continue
            
        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        
        # Select points closest to center first, but with some diversity
        order = np.argsort(radii, kind="stable")
        
        # Try different selection strategies
        if attempt % 3 == 0:
            selected = pts[order[:n_points]]
        elif attempt % 3 == 1:
            # Random subset from top 1.5x
            top_k = min(len(pts), int(n_points * 1.5))
            top_indices = order[:top_k]
            selected_indices = rng.choice(top_indices, size=n_points, replace=False)
            selected = pts[selected_indices]
        else:
            # Mix: 80% from top, 20% random from rest
            top_k = min(len(pts), int(n_points * 0.8))
            top_indices = order[:top_k]
            rest_indices = order[top_k:]
            if len(rest_indices) > 0:
                rest_selected = rng.choice(rest_indices, size=n_points - top_k, replace=False)
                selected = np.vstack([pts[top_indices], pts[rest_selected]])
            else:
                selected = pts[order[:n_points]]
        
        # Add small noise to break symmetry
        if attempt > 2:
            noise_scale = spacing * 0.03
            selected = selected + rng.normal(0, noise_scale, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0 - 0.01
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
            
        # Compute radius
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
            
    if best_pts is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.92
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

def _random_repulsive_init(n_points, seed=0):
    """Initialize with random points and push them apart."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.75, 0.75, size=(n_points, 2))
    
    # More aggressive repulsion
    for _ in range(80):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_idx = np.unravel_index(np.argmin(d), d.shape)
        if d[min_idx] < 0.04:
            i, j = min_idx
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-6:
                diff = diff / norm
                push = 0.015 * (0.04 - d[min_idx])
                pts[i] += push * diff
                pts[j] -= push * diff
                # Keep inside
                pts[i] = np.clip(pts[i], -0.93, 0.93)
                pts[j] = np.clip(pts[j], -0.93, 0.93)
                
    # Project inside
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 0.93
    pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.93
    
    return pts

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
    If edge_indices is provided, only consider those pairs for the pairwise term.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise distances
    if edge_indices is None:
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
        n_pairs = len(pair_half)
    else:
        # Sparse case
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])
        n_pairs = len(pair_half)
        
    all_vals = np.concatenate([pair_half, boundary])
    
    obj = -_soft_min_logsumexp(all_vals, T)
    
    # Gradient
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
    
    # Boundary gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary
    
    # Pairwise gradient
    if edge_indices is None:
        # Full pairwise
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        W = np.zeros((n, n))
        iu = np.triu_indices(n, k=1)
        W[iu] = grad_all[:n_pairs]
        W[iu[1], iu[0]] = grad_all[:n_pairs]
        
        weighted_unit = W[:, :, np.newaxis] * unit_vecs
        grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
        grad_pts += grad_pairs
    else:
        # Sparse pairwise
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
    """Fast force-directed relaxation using spatial hashing for O(N) neighbor search."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    # Estimate initial radius
    r_est = best_r
    
    lr = 0.008
    cooldown = 0.998
    
    # Spatial hash grid size based on expected spacing
    grid_size = max(2 * r_est, 0.05)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        # Build spatial hash
        cell_map = {}
        for i in range(n):
            cx = int(pos[i, 0] / grid_size)
            cy = int(pos[i, 1] / grid_size)
            cell = (cx, cy)
            if cell not in cell_map:
                cell_map[cell] = []
            cell_map[cell].append(i)
        
        # Pairwise repulsion using spatial hash
        target = 2.0 * r_est * 1.02
        for i in range(n):
            cx = int(pos[i, 0] / grid_size)
            cy = int(pos[i, 1] / grid_size)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (cx + dx, cy + dy)
                    if cell not in cell_map:
                        continue
                    for j in cell_map[cell]:
                        if j <= i:
                            continue
                        diff = pos[i] - pos[j]
                        dist = np.linalg.norm(diff)
                        if dist < target and dist > 1e-10:
                            unit_vec = diff / dist
                            excess = target - dist
                            force_mag = 3.0 * excess / dist
                            forces[i] += force_mag * unit_vec
                            forces[j] -= force_mag * unit_vec
        
        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.08
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
        
        # Project back into unit disk with small margin
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 0.995
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.994
            
        # Update best
        if step % 30 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r
                grid_size = max(2 * r_est, 0.05)
                
        lr *= cooldown
        
    return best_pos, best_r

def _anneal_restart(pts, best_r, rng, deadline, max_iter=20):
    """Anneal with random restarts to escape local minima."""
    pos = pts.copy()
    current_r = best_r
    temperature = 0.005
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Random perturbation
        perturbation_scale = temperature * np.sqrt(2.0 * current_r)
        pos_perturbed = pos + rng.normal(0, perturbation_scale, pos.shape)
        
        # Keep inside unit disk
        norms = np.linalg.norm(pos_perturbed, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            pos_perturbed[mask] = pos_perturbed[mask] / norms[mask, np.newaxis] * 0.985
            
        r_perturbed = _compute_radius(pos_perturbed)
        
        # Accept if better or with Metropolis probability
        if r_perturbed >= current_r or rng.random() < np.exp((r_perturbed - current_r) / (temperature * 0.001)):
            pos = pos_perturbed
            current_r = r_perturbed
            
        # Cool down
        temperature *= 0.95
        
    return pos, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin
    
    best_pts = None
    best_r = 0.0
    rng = np.random.default_rng(12345)
    
    # Phase 1: Initialization and Force Relaxation
    # Use multiple initialization strategies with more variety
    inits = [
        _hex_init(N, seed=42),
        _hex_init(N, seed=123),
        _hex_init(N, seed=777),
        _random_repulsive_init(N, seed=7),
        _random_repulsive_init(N, seed=99),
        _hex_init(N, seed=314)
    ]
    
    for pts in inits:
        if time.time() > deadline - 60.0:
            break
            
        r = _compute_radius(pts)
        
        # Force relaxation - use spatial hashing for speed
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=800, deadline=deadline - 55.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r
            
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
        
    # Phase 2: Anneal restart to explore configuration space
    if time.time() < deadline - 40.0:
        annealed_pts, annealed_r = _anneal_restart(best_pts, best_r, rng, deadline - 35.0, max_iter=15)
        if annealed_r > best_r:
            best_r = annealed_r
            best_pts = annealed_pts.copy()
            
        # Quick force relax after anneal
        if time.time() < deadline - 30.0:
            relaxed_pts, relaxed_r = _force_relax(best_pts, n_steps=500, deadline=deadline - 25.0)
            if relaxed_r > best_r:
                best_r = relaxed_r
                best_pts = relaxed_pts.copy()
                
    # Phase 3: Global Optimization with Soft-Min (full pairwise)
    if time.time() < deadline - 15.0:
        temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005]
        x0 = best_pts.ravel()
        
        for T in temperatures:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 50,
                        'maxfun': 300,
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
                
    # Phase 4: Local Polish with Dynamic Active Edges
    if time.time() < deadline - 3.0 and best_pts is not None:
        # Use dynamic active edge detection
        edge_list = _get_active_edges(best_pts, best_r, max_edges=120)
        
        if edge_list:
            temperatures = [0.0001, 0.00005, 0.00002, 0.00001, 0.000005, 0.000002]
            x0 = best_pts.ravel()
            
            for T in temperatures:
                if time.time() > deadline - 1.5:
                    break
                try:
                    # Update active edges periodically
                    if T < 0.00003:
                        edge_list = _get_active_edges(best_pts, best_r, max_edges=100)
                        
                    result = minimize(
                        lambda x: _objective_and_grad(x, T, edge_list)[0],
                        x0,
                        jac=lambda x: _objective_and_grad(x, T, edge_list)[1],
                        method='L-BFGS-B',
                        options={
                            'maxiter': 300,
                            'ftol': 1e-14,
                            'gtol': 1e-12,
                            'maxls': 80,
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
                    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 1.0 and best_pts is not None:
        edge_list = _get_active_edges(best_pts, best_r, max_edges=80)
        if edge_list:
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, 0.000001, edge_list)[0],
                    best_pts.ravel(),
                    jac=lambda x: _objective_and_grad(x, 0.000001, edge_list)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 500,
                        'ftol': 1e-16,
                        'gtol': 1e-14,
                        'maxls': 100,
                        'maxfun': 1000,
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
                    
    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
        
    return best_pts
