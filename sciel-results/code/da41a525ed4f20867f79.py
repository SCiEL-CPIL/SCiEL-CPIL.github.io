# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _all_triangle_indices(n):
    """Precompute all C(n,3) triangle index triples."""
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    # area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _subgradient(pts, tri_idx, smooth_beta=50.0):
    """
    Compute subgradient of the smooth surrogate min-area objective.
    Uses softmax-weighted subgradients of the active triangles.
    Returns (gradient, min_area_estimate).
    """
    areas = _triangle_areas(pts, tri_idx)
    # Soft-min via log-sum-exp trick
    # We want to approximate min(areas) with a smooth function
    # soft_min = -log(sum(exp(-beta * areas))) / beta
    # Gradient of soft_min w.r.t. pts:
    # d(soft_min)/d(area_i) = -exp(-beta*area_i) / sum(exp(-beta*areas)) * (-beta) / beta
    #                         = exp(-beta*area_i) / sum(exp(-beta*areas))
    # But we want to MAXIMIZE min, so we use soft-max of areas
    # soft_max = log(sum(exp(beta * areas))) / beta
    # d(soft_max)/d(area_i) = exp(beta*area_i) / sum(exp(beta*areas))
    # Then d(soft_max)/d(pts) = sum_i w_i * d(area_i)/d(pts)
    
    beta = smooth_beta
    # For numerical stability
    shifted = beta * areas
    max_shifted = shifted.max()
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / exp_shifted.sum()
    
    # soft_max value
    soft_max_val = (max_shifted + np.log(exp_shifted.sum())) / beta
    
    # Now compute gradient: for each triangle, the subgradient of area w.r.t. each point
    # area_i = 0.5 * |cross(b-a, c-a)| for triangle (i,j,k)
    # The gradient of the signed area (without abs) w.r.t. point p is:
    # For point a: 0.5 * (b - c) rotated 90 deg = 0.5 * (-(b_y-c_y), b_x-c_x)
    # For point b: 0.5 * (c - a) rotated 90 deg = 0.5 * (-(c_y-a_y), c_x-a_x)
    # For point c: 0.5 * (a - b) rotated 90 deg = 0.5 * (-(a_y-b_y), a_x-b_x)
    # But we have abs, so we need the sign of the cross product
    
    # Compute signed areas (2x the actual area for efficiency)
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    
    # signed area = 0.5 * (a_x(b_y-c_y) + b_x(c_y-a_y) + c_x(a_y-b_y))
    signed = (a[:, 0] * (b[:, 1] - c[:, 1]) +
              b[:, 0] * (c[:, 1] - a[:, 1]) +
              c[:, 0] * (a[:, 1] - b[:, 1]))
    
    # sign of each triangle
    signs = np.sign(signed)
    signs[signs == 0] = 1.0  # avoid zero
    
    # Gradient contribution from each triangle to each of its 3 points
    # d(area)/d(a) = 0.5 * sign * (-(b_y - c_y), b_x - c_x)
    # d(area)/d(b) = 0.5 * sign * (-(c_y - a_y), c_x - a_x)
    # d(area)/d(c) = 0.5 * sign * (-(a_y - b_y), a_x - b_x)
    
    grad = np.zeros_like(pts)
    
    # For point a (index tri_idx[:,0])
    da = 0.5 * signs[:, None] * np.stack([-(b[:, 1] - c[:, 1]), b[:, 0] - c[:, 0]], axis=1)
    # For point b (index tri_idx[:,1])
    db = 0.5 * signs[:, None] * np.stack([-(c[:, 1] - a[:, 1]), c[:, 0] - a[:, 0]], axis=1)
    # For point c (index tri_idx[:,2])
    dc = 0.5 * signs[:, None] * np.stack([-(a[:, 1] - b[:, 1]), a[:, 0] - b[:, 0]], axis=1)
    
    # Weighted sum
    w = weights[:, None]
    grad_contrib_a = w * da
    grad_contrib_b = w * db
    grad_contrib_c = w * dc
    
    # Accumulate into grad
    np.add.at(grad, tri_idx[:, 0], grad_contrib_a)
    np.add.at(grad, tri_idx[:, 1], grad_contrib_b)
    np.add.at(grad, tri_idx[:, 2], grad_contrib_c)
    
    return grad, soft_max_val

def _score(pts):
    """Compute the score: min_triangle_area / hull_area."""
    try:
        n = len(pts)
        tri_idx = _all_triangle_indices(n)
        areas = _triangle_areas(pts, tri_idx)
        min_area = areas.min()
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area
    except Exception:
        return 0.0

def _regular_polygon(n, r=1.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return r * np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _halton_sequence(n, dims=2):
    """Generate n Halton sequence points in [0,1]^dims."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, dims))
    for i in range(n):
        for j in range(dims):
            points[i, j] = halton(i + 1, j + 2)
    return points

def _optimize_from_start(pts, tri_idx, max_iters=500, time_limit=60.0):
    """
    Subgradient ascent on the smooth surrogate for min triangle area.
    We maximize min_area / hull_area. Since hull_area changes slowly,
    we primarily maximize min_area via subgradient ascent.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Adaptive step size
    step = 0.01
    patience = 0
    max_patience = 50
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        grad, soft_val = _subgradient(pts, tri_idx, smooth_beta=80.0)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad = grad / grad_norm
        
        # Try the step
        new_pts = pts + step * grad
        
        # Check if points are still valid (finite, etc.)
        if not np.all(np.isfinite(new_pts)):
            step *= 0.5
            continue
        
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            patience = 0
            # Increase step slightly
            step = min(step * 1.05, 0.05)
        else:
            patience += 1
            step *= 0.9
            if patience > max_patience:
                break
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    """
    n = 13
    tri_idx = _all_triangle_indices(n)
    
    deadline = time.time() + 580  # Safety margin
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Regular polygon
    starts.append(_regular_polygon(n, r=1.0))
    
    # 2. Halton sequence
    starts.append(_halton_sequence(n))
    
    # 3. Random starts with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        rng = np.random.default_rng(seed)
        starts.append(rng.random((n, 2)))
    
    # 4. Points near circle with some perturbation
    for seed in [10, 11, 12]:
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.uniform(-0.1, 0.1, n)
        radii = 1.0 + rng.uniform(-0.1, 0.1, n)
        pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        starts.append(pts)
    
    # 5. Grid-like arrangements
    # 4x4 grid minus one corner, then slightly perturb
    grid_x = np.array([0, 0.5, 1, 1.5] * 4)
    grid_y = np.array([0]*4 + [0.5]*4 + [1]*4 + [1.5]*4)
    grid_pts = np.stack([grid_x, grid_y], axis=1)
    starts.append(grid_pts[:n])
    
    best_pts = None
    best_score = -1.0
    
    # Optimize from each start
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Scale start to have hull area ~1 for consistency
        try:
            hull = ConvexHull(start_pts)
            hull_area = hull.volume
            if hull_area > 0:
                start_pts = start_pts / np.sqrt(hull_area)
        except Exception:
            pass
        
        # Time per optimization: divide remaining time among remaining starts
        remaining_starts = len(starts) - i
        time_per_opt = min(60.0, (deadline - time.time()) / max(1, remaining_starts))
        if time_per_opt < 5.0:
            break
        
        pts, score = _optimize_from_start(start_pts, tri_idx, max_iters=300, time_limit=time_per_opt)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final polish: more iterations on the best solution
    if best_pts is not None and time.time() < deadline - 30:
        time_left = deadline - time.time() - 10
        best_pts, best_score = _optimize_from_start(best_pts, tri_idx, max_iters=1000, time_limit=time_left)
    
    # Normalize: scale so that the result is well-conditioned
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(n, r=1.0)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _regular_polygon(n, r=1.0)
    
    return best_pts

# EVOLVE-BLOCK-END
