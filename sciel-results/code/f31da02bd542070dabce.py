"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _optimize(pts, time_budget):
    """
    Two-phase optimization:
    Phase 1: Gradient ascent with soft-min, moderate sharpness, larger steps.
    Phase 2: Polish with very sharp soft-min, small steps, for final refinement.
    """
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _compute_radius(pts)
    
    t0 = time.time()
    
    # Phase 1: Broad optimization
    k = 50.0
    lr = 0.01
    phase1_time = time_budget * 0.6
    
    while time.time() - t0 < phase1_time:
        # Compute soft-min radius and gradient
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_radii = dists / 2.0
        
        # Soft min of boundary
        m_b = np.min(boundary)
        shifted_b = boundary - m_b
        log_sum_b = np.log(np.sum(np.exp(-k * shifted_b)))
        soft_boundary = m_b - log_sum_b / k
        
        weights_b = np.exp(-k * shifted_b)
        weights_b = weights_b / np.sum(weights_b)
        
        grad = np.zeros_like(pts)
        for i in range(n):
            if norms[i] > 1e-10:
                grad[i] += -weights_b[i] * pts[i] / norms[i]
            else:
                grad[i] += -weights_b[i] * np.array([1.0, 0.0])
        
        # Soft min of pair radii (only close pairs for efficiency)
        threshold = 0.15
        close_mask = (dists < threshold) & (dists > 1e-10)
        close_pairs = np.argwhere(close_mask)
        close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]
        
        if len(close_pairs) > 0:
            pair_vals = pair_radii[close_pairs[:, 0], close_pairs[:, 1]]
            m_p = np.min(pair_vals)
            shifted_p = pair_vals - m_p
            log_sum_p = np.log(np.sum(np.exp(-k * shifted_p)))
            soft_pair = m_p - log_sum_p / k
            
            weights_p = np.exp(-k * shifted_p)
            weights_p = weights_p / np.sum(weights_p)
            
            for idx, (i, j) in enumerate(close_pairs):
                w = weights_p[idx]
                if w < 1e-10:
                    continue
                diff_ij = pts[i] - pts[j]
                dist_ij = np.linalg.norm(diff_ij)
                if dist_ij < 1e-10:
                    continue
                grad[i] += w * diff_ij / (2.0 * dist_ij)
                grad[j] -= w * diff_ij / (2.0 * dist_ij)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Gradient ascent step
        pts = pts + lr * grad
        
        # Keep inside unit disk
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] *= 0.999 / norms[too_far]
        
        # Check improvement
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        # Annealing
        lr *= 0.995
        k *= 1.005
        
        if lr < 1e-6:
            break
    
    # Phase 2: Polish with very sharp soft-min
    pts = best_pts.copy()
    k = 500.0
    lr = 0.001
    
    while time.time() - t0 < time_budget:
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_radii = dists / 2.0
        
        # Soft min of boundary
        m_b = np.min(boundary)
        shifted_b = boundary - m_b
        log_sum_b = np.log(np.sum(np.exp(-k * shifted_b)))
        soft_boundary = m_b - log_sum_b / k
        
        weights_b = np.exp(-k * shifted_b)
        weights_b = weights_b / np.sum(weights_b)
        
        grad = np.zeros_like(pts)
        for i in range(n):
            if norms[i] > 1e-10:
                grad[i] += -weights_b[i] * pts[i] / norms[i]
            else:
                grad[i] += -weights_b[i] * np.array([1.0, 0.0])
        
        # Soft min of pair radii
        threshold = 0.12
        close_mask = (dists < threshold) & (dists > 1e-10)
        close_pairs = np.argwhere(close_mask)
        close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]
        
        if len(close_pairs) > 0:
            pair_vals = pair_radii[close_pairs[:, 0], close_pairs[:, 1]]
            m_p = np.min(pair_vals)
            shifted_p = pair_vals - m_p
            log_sum_p = np.log(np.sum(np.exp(-k * shifted_p)))
            soft_pair = m_p - log_sum_p / k
            
            weights_p = np.exp(-k * shifted_p)
            weights_p = weights_p / np.sum(weights_p)
            
            for idx, (i, j) in enumerate(close_pairs):
                w = weights_p[idx]
                if w < 1e-10:
                    continue
                diff_ij = pts[i] - pts[j]
                dist_ij = np.linalg.norm(diff_ij)
                if dist_ij < 1e-10:
                    continue
                grad[i] += w * diff_ij / (2.0 * dist_ij)
                grad[j] -= w * diff_ij / (2.0 * dist_ij)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Small step
        pts = pts + lr * grad
        
        # Keep inside unit disk
        norms = np.linalg.norm(pts, axis=1)
        too_far = norms > 0.999
        if np.any(too_far):
            pts[too_far] *= 0.999 / norms[too_far]
        
        # Check improvement
        current_score = _compute_radius(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        # Annealing
        lr *= 0.998
        k *= 1.002
        
        if lr < 1e-8:
            break
    
    return best_pts


def _validate_and_fix(pts):
    """Validate and fix any issues with the packing."""
    pts = np.array(pts, dtype=np.float64)
    
    # Check for non-finite values
    if not np.all(np.isfinite(pts)):
        pts[~np.isfinite(pts)] = 0.0
    
    # Check shape
    if pts.shape != (N, 2):
        if pts.size == N * 2:
            pts = pts.reshape(N, 2)
        else:
            # Fallback to hex init
            pts = _hex_init(N)
    
    # Ensure inside unit disk
    norms = np.linalg.norm(pts, axis=1)
    too_far = norms > 1.0
    if np.any(too_far):
        pts[too_far] *= 1.0 / norms[too_far]
    
    return pts


def construct():
    """Construct the packing. Returns array of shape (N, 2)."""
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Initial hexagonal lattice
    pts = _hex_init(N)
    
    # Validate initial
    pts = _validate_and_fix(pts)
    
    # Optimize
    remaining = deadline - (time.time() - t0)
    if remaining > 5.0:
        pts = _optimize(pts, remaining)
    
    # Final validation
    pts = _validate_and_fix(pts)
    
    return pts
# EVOLVE-BLOCK-END
