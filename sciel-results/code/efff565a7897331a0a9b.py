"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using hexagonal lattice.
2. Refine with combined gradient ascent on soft-min radius and Voronoi cell area.
3. Final polish pass with tighter parameters.
4. Deadline guard to ensure termination within 120s.
"""
import numpy as np
import time
from scipy.spatial import Voronoi

N = 300

# EVOLVE-BLOCK-START
def _hex_init(n, spacing=None):
    """Generate initial hexagonal lattice positions inside unit disk."""
    if spacing is None:
        spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= n:
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.97


def _compute_voronoi_areas(pts):
    """
    Compute Voronoi cell areas for points inside unit disk.
    Returns array of cell areas. Points with unbounded cells get area based on boundary distance.
    """
    n = len(pts)
    areas = np.zeros(n)
    
    # Add boundary points to make Voronoi bounded
    n_boundary = 200
    angles = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    boundary_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_pts = np.vstack([pts, boundary_pts])
    
    try:
        vor = Voronoi(all_pts)
        
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Unbounded cell: approximate area using distance to boundary
                dist_to_boundary = 1.0 - np.linalg.norm(pts[i])
                areas[i] = np.pi * dist_to_boundary ** 2
            else:
                vertices = vor.vertices[region]
                # Compute polygon area
                if len(vertices) < 3:
                    areas[i] = 0.0
                else:
                    # Shoelace formula
                    x = vertices[:, 0]
                    y = vertices[:, 1]
                    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
                    areas[i] = area
                    
    except Exception:
        # Fallback: use simple distance-based estimate
        norms = np.linalg.norm(pts, axis=1)
        areas = np.pi * (1.0 - norms) ** 2
    
    return areas


def _voronoi_gradient(pts, k, threshold=0.15):
    """
    Compute gradient of soft-min of Voronoi cell areas.
    Returns (soft_value, gradient_array).
    """
    n = len(pts)
    areas = _compute_voronoi_areas(pts)
    
    # Soft min of areas
    m_a = float(np.min(areas))
    shifted_a = areas - m_a
    exp_a = np.exp(-k * shifted_a)
    sum_exp_a = float(np.sum(exp_a))
    soft_area = m_a - np.log(sum_exp_a) / k
    weights_a = exp_a / sum_exp_a
    
    # Gradient: for each point, the gradient of its cell area w.r.t. its position
    # This is complex, so we use a finite difference approximation
    grad = np.zeros_like(pts)
    eps = 1e-6
    
    for i in range(n):
        if weights_a[i] < 1e-12:
            continue
        
        # Compute area at perturbed positions
        for d in range(2):
            pts_plus = pts.copy()
            pts_minus = pts.copy()
            pts_plus[i, d] += eps
            pts_minus[i, d] -= eps
            
            # Keep inside disk
            for p in [pts_plus, pts_minus]:
                norms = np.linalg.norm(p, axis=1)
                too_far = norms > 0.999
                if np.any(too_far):
                    p[too_far] = p[too_far] * (0.999 / norms[too_far])
            
            area_plus = _compute_voronoi_areas(pts_plus)[i]
            area_minus = _compute_voronoi_areas(pts_minus)[i]
            
            # Gradient of soft min w.r.t. x_{i,d}
            # d(soft_min)/dx = w_i * d(area_i)/dx
            grad[i, d] = weights_a[i] * (area_plus - area_minus) / (2 * eps)
    
    return soft_area, grad


def _compute_radius(pts):
    """Compute the minimum radius for a given set of points."""
    n = len(pts)
    if n < 2:
        return float(np.min(1.0 - np.linalg.norm(pts, axis=1)))
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = float(np.min(dists))
    min_boundary = float(np.min(1.0 - np.linalg.norm(pts, axis=1)))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_gradient(pts, k, threshold=0.15):
    """
    Compute soft-min radius and its gradient using log-sum-exp.
    Uses sparse pairwise for efficiency.
    Returns (soft_value, gradient_array).
    """
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Soft min of boundary constraints
    m_b = float(np.min(boundary))
    shifted_b = boundary - m_b
    exp_b = np.exp(-k * shifted_b)
    sum_exp_b = float(np.sum(exp_b))
    soft_boundary = m_b - np.log(sum_exp_b) / k
    weights_b = exp_b / sum_exp_b
    
    # Gradient of soft_boundary w.r.t. pts
    # d(soft_min)/d(c_i) = -w_i * c_i / |c_i|  (for boundary term 1-|c_i|)
    grad_boundary = np.zeros_like(pts)
    for i in range(n):
        if norms[i] > 1e-10:
            grad_boundary[i] = -weights_b[i] * pts[i] / norms[i]
        else:
            grad_boundary[i] = np.array([0.0, 0.0])
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_radii = dists / 2.0
    
    # Sparse: only consider pairs within threshold
    close_mask = (dists < threshold) & (dists > 1e-10)
    close_pairs = np.argwhere(close_mask)
    close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]
    
    if len(close_pairs) == 0:
        soft_pair = np.inf
        grad_pair = np.zeros_like(pts)
    else:
        pair_vals = pair_radii[close_pairs[:, 0], close_pairs[:, 1]]
        m_p = float(np.min(pair_vals))
        shifted_p = pair_vals - m_p
        exp_p = np.exp(-k * shifted_p)
        sum_exp_p = float(np.sum(exp_p))
        soft_pair = m_p - np.log(sum_exp_p) / k
        weights_p = exp_p / sum_exp_p
        
        grad_pair = np.zeros_like(pts)
        for idx in range(len(close_pairs)):
            w = weights_p[idx]
            if w < 1e-12:
                continue
            i, j = close_pairs[idx]
            diff_ij = pts[i] - pts[j]
            dist_ij = dists[i, j]
            if dist_ij < 1e-10:
                continue
            # d(pair_radius)/d(c_i) = (c_i - c_j) / (2 * dist_ij)
            # d(soft_min)/d(c_i) = w * d(pair_radius)/d(c_i)
            unit = diff_ij / dist_ij
            grad_pair[i] += w * unit / 2.0
            grad_pair[j] -= w * unit / 2.0
    
    # Combine: take the tighter constraint's gradient
    if soft_boundary < soft_pair:
        return soft_boundary, grad_boundary
    else:
        return soft_pair, grad_pair


def _combined_gradient(pts, k, alpha=0.5):
    """
    Combine soft-min radius gradient with Voronoi area gradient.
    Returns (combined_value, combined_gradient).
    """
    radius_val, radius_grad = _soft_min_gradient(pts, k)
    area_val, area_grad = _voronoi_gradient(pts, k)
    
    # Normalize gradients to similar scale
    radius_norm = np.linalg.norm(radius_grad)
    area_norm = np.linalg.norm(area_grad)
    
    if radius_norm > 1e-12:
        radius_grad = radius_grad / radius_norm
    if area_norm > 1e-12:
        area_grad = area_grad / area_norm
    
    # Combine with weighting
    combined_grad = (1 - alpha) * radius_grad + alpha * area_grad
    combined_val = (1 - alpha) * radius_val + alpha * area_val
    
    return combined_val, combined_grad


def _optimize(pts, time_budget, initial_lr=0.005, initial_k=30.0):
    """Optimize packing using gradient ascent on combined soft-min radius and Voronoi area."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    k = initial_k
    lr = initial_lr
    patience = 0
    max_patience = 500
    alpha = 0.3  # Weight for Voronoi area term
    
    t0 = time.time()
    step = 0
    while time.time() - t0 < time_budget:
        combined_val, grad = _combined_gradient(pts, k, alpha=alpha)
        
        # Gradient ascent
        pts = pts + lr * grad
        
        # Keep points inside the unit disk
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
        
        # Compute actual score
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate and sharpness
        step += 1
        if step % 100 == 0:
            lr = max(0.0005, lr * 0.95)
            k = min(200.0, k * 1.05)  # Increase sharpness over time
            # Reduce alpha over time to focus more on radius
            alpha = max(0.1, alpha * 0.98)
        
        if patience > max_patience:
            break
    
    return best_pts, best_score


def _polish(pts, time_budget):
    """Final high-precision polish with tighter parameters, focusing on radius."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    k = 100.0
    lr = 0.001
    patience = 0
    max_patience = 300
    
    t0 = time.time()
    step = 0
    while time.time() - t0 < time_budget:
        # Use only radius gradient for polish
        soft_val, grad = _soft_min_gradient(pts, k, threshold=0.10)
        
        pts = pts + lr * grad
        
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] = pts[too_far] * (0.999 / norms[too_far])
        
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        step += 1
        if step % 50 == 0:
            lr = max(0.0001, lr * 0.9)
            k = min(500.0, k * 1.03)
        
        if patience > max_patience:
            break
    
    return best_pts, best_score


def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    norms = np.linalg.norm(pos, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
    
    # Check for coincident points
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = float(np.min(dist))
    if min_dist < 1e-4:
        rng = np.random.default_rng(42)
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < 1e-4:
                    pos[i] += rng.standard_normal(2) * 1e-3
                    pos[j] -= rng.standard_normal(2) * 1e-3
                    n_i = np.linalg.norm(pos[i])
                    if n_i > 0.999:
                        pos[i] = pos[i] * (0.999 / n_i)
                    n_j = np.linalg.norm(pos[j])
                    if n_j > 0.999:
                        pos[j] = pos[j] * (0.999 / n_j)
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s budget, leaving 20s margin
    
    # Step 1: Generate initial positions
    positions = _hex_init(N)
    
    # Step 2: Main optimization
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        positions, _ = _optimize(positions, time_budget=time_remaining - 15.0)
    
    # Step 3: Polish pass
    time_remaining = deadline - time.time()
    if time_remaining > 3.0:
        positions, _ = _polish(positions, time_budget=time_remaining - 5.0)
    
    # Final validation
    positions = _validate_and_fix(positions)
    
    return positions
# EVOLVE-BLOCK-END
