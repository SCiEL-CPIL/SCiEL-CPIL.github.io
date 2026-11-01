import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 300

def _compute_score(pts):
    """Compute the packing radius."""
    min_pair = np.min(pdist(pts)) / 2.0
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_boundary)

def _soft_min_value_and_grad(pts, beta):
    """
    Soft-min surrogate: -log(sum(exp(-beta * d_i))) / beta
    where d_i are the distances (pair distances and boundary distances).
    Returns (value, gradient).
    
    For numerical stability, we use log-sum-exp.
    """
    # Pair distances
    pair_dists = pdist(pts)
    # Boundary distances
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min = -log(sum(exp(-beta * d))) / beta
    # For stability: min_d = min(all_dists)
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = -(log_sum + (-beta * min_d)) / beta
    # soft_min_val = min_d - log(sum(exp(-beta*(d - min_d)))) / beta
    # Let me redo: soft_min = -log(sum(exp(-beta*d)))/beta
    # = -log(exp(-beta*min_d) * sum(exp(-beta*(d-min_d))))/beta
    # = min_d - log(sum(exp(-beta*(d-min_d))))/beta
    
    soft_min_val = min_d - log_sum / beta
    
    # Gradient: d(soft_min)/dx_i = sum_j (-beta * exp(-beta*d_j) * dd_j/dx_i) / (beta * sum(exp(-beta*d_j)))
    # = -sum(exp(-beta*d_j) * dd_j/dx_i) / sum(exp(-beta*d_j))
    # Note: weights w_j = exp(-beta*d_j) / sum(exp(-beta*d_j))
    # soft_min = sum_j w_j * d_j (weighted average with weights concentrated on min)
    # Actually: soft_min = -log(sum(exp(-beta*d_j)))/beta
    # d(soft_min)/dx_i = (1/beta) * sum_j (exp(-beta*d_j) * d(d_j)/dx_i) / sum_j exp(-beta*d_j)
    # = sum_j w_j * d(d_j)/dx_i where w_j = exp(-beta*d_j)/sum(exp(-beta*d_j))
    
    # Compute weights
    exp_terms_normalized = exp_terms / np.sum(exp_terms)
    
    # We need to split back: first len(pair_dists) are pair distances, rest are boundary
    n_pairs = len(pair_dists)
    w_pairs = exp_terms_normalized[:n_pairs]
    w_boundary = exp_terms_normalized[n_pairs:]
    
    # Gradient from pair distances
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # d(d_ij)/dx_j = (x_j - x_i) / d_ij
    grad = np.zeros_like(pts)
    
    # Precompute pair indices
    # pair_dists from pdist is in order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = pair_dists[idx]
            if d > 1e-12:
                diff = pts[i] - pts[j]
                grad[i] += w_pairs[idx] * diff / d
                grad[j] -= w_pairs[idx] * diff / d
            idx += 1
    
    # Gradient from boundary distances
    # d(1 - |x_i|)/dx_i = -x_i / |x_i|
    for i in range(N):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            grad[i] -= w_boundary[i] * pts[i] / r
    
    return soft_min_val, grad

def _hex_lattice_init(n, max_trials=20):
    """Generate hexagonal lattice initial points."""
    best_points = None
    best_score = -1
    
    for trial in range(max_trials):
        spacing = 2.0 / (np.sqrt(n) + 1.0) * (1.0 - trial * 0.012)
        if spacing <= 0.001:
            continue
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        if len(points) < n:
            continue
        
        points = np.array(points)
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)[:n]
        pts_sel = points[idx].copy()
        
        score = _compute_score(pts_sel)
        if score > best_score:
            best_score = score
            best_points = pts_sel.copy()
    
    return best_points

def _ring_init(n):
    """Initialize points in concentric rings."""
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    remaining = n - 1
    ring = 1
    count = 0
    while remaining > 0:
        # Approximate ring radius
        # Number of points on ring k is roughly 6k
        # But we need to fit within the circle
        n_on_ring = min(remaining, max(6, 6 * ring))
        radius = 0.28 * np.sqrt(ring + 1.0)
        if radius > 0.95:
            radius = 0.95
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False) + np.pi / 6
        for k in range(n_on_ring):
            if count >= n:
                break
            pts[count] = [radius * np.cos(angles[k]), radius * np.sin(angles[k])]
            count += 1
        remaining -= n_on_ring
        ring += 1
    
    return pts[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Try multiple initializations
    best_radius = -1
    best_pts = None
    
    init_strategies = [
        ("hex", lambda: _hex_lattice_init(N)),
        ("ring", lambda: _ring_init(N)),
    ]
    
    for name, init_fn in init_strategies:
        if time.time() > deadline - 20.0:
            break
        try:
            pts = init_fn()
            if pts is None or len(pts) < N:
                continue
            pts = pts[:N].copy()
            # Small random perturbation to break symmetry
            pts += np.random.normal(0, 0.001, size=pts.shape)
            # Project inside disk
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            mask = r_pts > 0.99
            if np.any(mask):
                pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * 0.99
            
            radius = _compute_score(pts)
            if radius > best_radius:
                best_radius = radius
                best_pts = pts.copy()
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback: uniform random
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        best_radius = _compute_score(best_pts)
    
    # Step 2: Multi-phase soft-min optimization with increasing beta
    # Phase 1: low beta (smooth), Phase 2: medium, Phase 3: high beta (sharp)
    
    beta_schedule = [
        (20.0, 150),   # (beta, max L-BFGS iterations)
        (50.0, 150),
        (100.0, 150),
        (200.0, 200),
        (500.0, 200),
        (1000.0, 200),
        (2000.0, 300),
        (5000.0, 300),
    ]
    
    for beta, max_iter in beta_schedule:
        if time.time() > deadline - 5.0:
            break
        
        x_flat = best_pts.flatten().copy()
        
        try:
            def obj_and_grad(x, b):
                pts = x.reshape(-1, 2)
                # Project inside disk
                r_pts = np.hypot(pts[:, 0], pts[:, 1])
                mask = r_pts > 0.999
                if np.any(mask):
                    pts = pts.copy()
                    pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * 0.999
                val, grad = _soft_min_value_and_grad(pts, b)
                # We want to MAXIMIZE soft_min, so minimize -soft_min
                return -val, -grad
            
            def obj_only(x):
                return obj_and_grad(x, beta)[0]
            
            def grad_only(x):
                return obj_and_grad(x, beta)[1]
            
            res = minimize(
                obj_only,
                x_flat,
                jac=grad_only,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            x_flat = res.x
            pts = x_flat.reshape(-1, 2)
            
            # Project back to disk
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            mask = r_pts > 0.999
            if np.any(mask):
                pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * 0.999
            
            current_radius = _compute_score(pts)
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    # Step 3: Final polish with very high beta
    if time.time() < deadline - 3.0:
        try:
            x_flat = best_pts.flatten().copy()
            
            def obj_and_grad_final(x, b):
                pts = x.reshape(-1, 2)
                r_pts = np.hypot(pts[:, 0], pts[:, 1])
                mask = r_pts > 0.9999
                if np.any(mask):
                    pts = pts.copy()
                    pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * 0.9999
                val, grad = _soft_min_value_and_grad(pts, b)
                return -val, -grad
            
            def obj_only_final(x):
                return obj_and_grad_final(x, 10000.0)[0]
            
            def grad_only_final(x):
                return obj_and_grad_final(x, 10000.0)[1]
            
            res = minimize(
                obj_only_final,
                x_flat,
                jac=grad_only_final,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            
            x_flat = res.x
            pts = x_flat.reshape(-1, 2)
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            mask = r_pts > 0.9999
            if np.any(mask):
                pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * 0.9999
            
            current_radius = _compute_score(pts)
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
                
        except Exception:
            pass
    
    # Step 4: Local repulsion-based refinement (fast, no gradient)
    if time.time() < deadline - 2.0:
        pts = best_pts.copy()
        current_spacing = best_radius * 2.0
        
        for it in range(3000):
            if time.time() > deadline - 1.0:
                break
            
            # Compute forces from nearby points only
            # Use simple O(N^2) but with early cutoff
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    d = np.linalg.norm(diff)
                    if d < 1e-12:
                        continue
                    if d < current_spacing * 4.0:
                        force_mag = 1.0 / (d * d + 1e-10)
                        forces[i] += diff / d * force_mag
                        forces[j] -= diff / d * force_mag
            
            # Boundary force
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            boundary_margin = 1.0 - r_pts
            for i in range(N):
                if r_pts[i] > 1e-12:
                    excess = max(0.0, 0.01 - boundary_margin[i])
                    if excess > 0:
                        forces[i] -= pts[i] / r_pts[i] * excess * excess * 200.0
            
            # Adaptive step
            base_step = 0.001 * np.exp(-it / 1000.0)
            force_mag = np.linalg.norm(forces, axis=1)
            max_step = 0.005
            scale = np.minimum(base_step, max_step / (force_mag + 1e-10))
            pts_new = pts + forces * scale[:, np.newaxis]
            
            # Project inside disk
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask = r_new > 0.9999
            if np.any(mask):
                pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
            
            r_new_val = _compute_score(pts_new)
            if r_new_val > best_radius:
                best_radius = r_new_val
                best_pts = pts_new.copy()
                pts = pts_new.copy()
                current_spacing = best_radius * 2.0
            else:
                pts = pts_new.copy()
    
    return best_pts
