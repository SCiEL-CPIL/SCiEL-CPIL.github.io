"""Pack 200 equal circles in a unit square using discrete lattice search + continuous polish."""
import numpy as np
import time
from scipy.spatial import KDTree

N = 200
_TIME_LIMIT = 135  # seconds, leave margin before 150s kill

def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance using KDTree
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
        if min_pair_dist == np.inf:
            from scipy.spatial.distance import cdist
            D = cdist(pts, pts)
            np.fill_diagonal(D, np.inf)
            min_pair_dist = np.min(D)

    return min(wall_dist, min_pair_dist / 2.0)


def _generate_hex_lattice(spacing, margin=0.0):
    """
    Generate a hexagonal lattice of candidate points inside [-0.5-margin, 0.5+margin]^2.
    Returns array of shape (M, 2).
    """
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing
    
    pts = []
    y = -0.5 - margin
    row_idx = 0
    while y <= 0.5 + margin:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 - margin + shift
        while x <= 0.5 + margin:
            pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1
    
    if not pts:
        return np.empty((0, 2))
    return np.array(pts)


def _greedy_select(lattice_pts, N, target_r=None):
    """
    Greedy selection of N points from lattice_pts to maximize minimum distance.
    Uses a KDTree for efficient neighbor queries.
    
    Args:
        lattice_pts: (M, 2) array of candidate points
        N: number of points to select
        target_r: if provided, only select points that are at least 2*target_r from
                  already selected points and from walls
    
    Returns:
        (N, 2) array of selected points
    """
    M = len(lattice_pts)
    if M <= N:
        return lattice_pts.copy()
    
    selected = []
    selected_pts = np.empty((0, 2))
    available = set(range(M))
    
    # Precompute wall distances for each lattice point
    wall_dists = np.minimum(
        np.minimum(lattice_pts[:, 0] + 0.5, 0.5 - lattice_pts[:, 0]),
        np.minimum(lattice_pts[:, 1] + 0.5, 0.5 - lattice_pts[:, 1])
    )
    
    # Start with the point that has the best wall distance (most central)
    best_idx = np.argmax(wall_dists)
    selected.append(best_idx)
    selected_pts = lattice_pts[best_idx:best_idx+1]
    available.discard(best_idx)
    
    for count in range(1, N):
        if time.time() > _TIME_LIMIT - 15:
            break
        
        # Find the best available point
        best_idx = None
        best_score = -1
        
        # Build KDTree from selected points for efficient distance queries
        if len(selected_pts) > 0:
            tree = KDTree(selected_pts)
        else:
            tree = None
        
        for idx in available:
            # Check wall constraint
            wd = wall_dists[idx]
            if target_r is not None and wd < target_r:
                continue
            
            # Check distance to already selected points
            if tree is not None and len(selected_pts) > 0:
                dist, _ = tree.query(lattice_pts[idx])
                if target_r is not None and dist < 2 * target_r:
                    continue
                score = min(wd, dist / 2.0)
            else:
                score = wd
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is None:
            # Fallback: pick the point with best wall distance
            remaining = list(available)
            if remaining:
                best_idx = max(remaining, key=lambda i: wall_dists[i])
            else:
                break
        
        selected.append(best_idx)
        selected_pts = np.vstack([selected_pts, lattice_pts[best_idx:best_idx+1]])
        available.discard(best_idx)
    
    return lattice_pts[selected]


def _local_polish(pts, r_target, max_iters=2000, lr=0.0005, cooling=0.9995):
    """
    Force-directed local optimization to push all constraints above r_target.
    """
    pts = pts.copy()
    n = len(pts)
    current_lr = lr
    
    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 10:
            break
        
        forces = np.zeros_like(pts)
        tree = KDTree(pts)
        
        for i in range(n):
            # Wall forces
            if pts[i, 0] < -0.5 + r_target:
                forces[i, 0] += 5.0 * (r_target - (pts[i, 0] + 0.5))
            if pts[i, 0] > 0.5 - r_target:
                forces[i, 0] -= 5.0 * (pts[i, 0] - (0.5 - r_target))
            if pts[i, 1] < -0.5 + r_target:
                forces[i, 1] += 5.0 * (r_target - (pts[i, 1] + 0.5))
            if pts[i, 1] > 0.5 - r_target:
                forces[i, 1] -= 5.0 * (pts[i, 1] - (0.5 - r_target))
            
            # Repulsive forces from close neighbors
            neighbors = tree.query_ball_point(pts[i], 2 * r_target + 0.02)
            for j in neighbors:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.005
                    continue
                if dist < 2 * r_target:
                    overlap = 2 * r_target - dist
                    force_mag = 10.0 * overlap / dist
                    forces[i] += force_mag * diff
        
        pts += current_lr * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        current_lr *= cooling
        
        if np.max(np.abs(forces)) < 1e-7:
            break
    
    return pts


def _soft_min_objective(pts, beta=50.0):
    """
    Compute soft-min of (pairwise distances/2, wall distances) and its gradient.
    """
    n = len(pts)
    if n == 0:
        return 0.0, np.zeros_like(pts)
    
    # Wall distances
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])
    
    # Pair distances
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    
    if len(pairs) == 0:
        pair_vals = np.array([10.0])
    else:
        pair_vals = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in pairs])
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft min computation
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    soft_min_val = -(max_neg + np.log(sum_exp)) / beta
    
    # Gradient
    grad = np.zeros_like(pts)
    weights = exp_shifted / sum_exp
    
    # Wall gradients
    for i in range(n):
        grad[i, 0] += weights[4*i] * 1.0
        grad[i, 0] += weights[4*i+1] * (-1.0)
        grad[i, 1] += weights[4*i+2] * 1.0
        grad[i, 1] += weights[4*i+3] * (-1.0)
    
    # Pair gradients
    if len(pairs) > 0:
        for k, (i, j) in enumerate(pairs):
            w = weights[4*n + k]
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            grad[i] += w * diff / (2.0 * dist)
            grad[j] -= w * diff / (2.0 * dist)
    
    return soft_min_val, grad


def _smooth_polish(pts, max_iters=3000, lr=0.0003, cooling=0.9997, beta_start=10.0, beta_end=200.0):
    """
    Gradient-based smooth optimization to polish the packing.
    """
    pts = pts.copy()
    
    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 5:
            break
        
        progress = it / max_iters
        beta = beta_start + (beta_end - beta_start) * progress
        
        obj, grad = _soft_min_objective(pts, beta)
        
        pts += lr * grad
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        lr *= cooling
        
        if np.max(np.abs(grad)) < 1e-8:
            break
    
    return pts


def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Strategy:
    # 1. Try multiple hex lattice spacings with greedy selection
    # 2. Polish each result with force-directed optimization
    # 3. Final smooth polish on the best result
    
    # Different lattice spacings to try (spacing between adjacent lattice points)
    # For 200 circles in unit square, typical spacing ~ 1/sqrt(200) ≈ 0.07
    spacings = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    
    for spacing in spacings:
        if time.time() - t0 > _TIME_LIMIT - 20:
            break
        
        # Generate hex lattice
        lattice = _generate_hex_lattice(spacing, margin=0.01)
        
        if len(lattice) < N:
            continue
        
        # Greedy selection
        selected = _greedy_select(lattice, N)
        
        if len(selected) < N:
            continue
        
        r_init = _get_radius(selected)
        
        # Polish with force-directed
        r_target = r_init * 1.02
        polished = _local_polish(selected, r_target, max_iters=1500, lr=0.0004, cooling=0.9996)
        r_polished = _get_radius(polished)
        
        # Keep the better of initial and polished
        if r_polished >= r_init:
            current_r, current_pts = r_polished, polished
        else:
            current_r, current_pts = r_init, selected
        
        if current_r > best_r:
            best_r = current_r
            best_pts = current_pts.copy()
    
    # If we have a good result, do a final smooth polish
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 25:
        smooth_result = _smooth_polish(best_pts, max_iters=2000, lr=0.0002, cooling=0.9998, 
                                         beta_start=15.0, beta_end=150.0)
        r_smooth = _get_radius(smooth_result)
        if r_smooth > best_r:
            best_r = r_smooth
            best_pts = smooth_result.copy()
    
    # One more force-directed pass on the best
    if best_pts is not None and time.time() - t0 < _TIME_LIMIT - 10:
        r_target = best_r * 1.01
        final_polish = _local_polish(best_pts, r_target, max_iters=1000, lr=0.0002, cooling=0.9998)
        r_final = _get_radius(final_polish)
        if r_final > best_r:
            best_r = r_final
            best_pts = final_polish.copy()
    
    # Fallback
    if best_pts is None:
        # Simple grid fallback
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
