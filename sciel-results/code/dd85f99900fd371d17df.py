import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

# Precompute triangle indices for 13 points
N_POINTS = 13
TRIANGLE_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)))
N_TRIS = len(TRIANGLE_IDX)

def compute_all_areas(points):
    """Compute all 286 triangle areas efficiently."""
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return areas

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0.0 if the points are collinear or invalid.
    """
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-10:
        return 0.0
    areas = compute_all_areas(points)
    A_min = np.min(areas)
    return A_min / A_hull

def soft_min_objective(x, beta):
    """
    Soft-min surrogate objective for BFGS.
    Minimizes -softmin(areas) / hull_area.
    """
    points = x.reshape(N_POINTS, 2)
    if not np.all(np.isfinite(points)):
        return 1e6
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 1e6
    if a_hull <= 1e-10:
        return 1e6
    areas = compute_all_areas(points)
    m = np.min(areas)
    # Use log-sum-exp for numerical stability
    shifted = areas - m
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = m - (1.0 / beta) * log_sum
    val = -softmin_val / a_hull
    if not np.isfinite(val):
        return 1e6
    return val

def soft_min_gradient(x, beta):
    """
    Numerical gradient of the soft-min objective.
    """
    points = x.reshape(N_POINTS, 2)
    if not np.all(np.isfinite(points)):
        return np.zeros_like(x)
    
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return np.ones_like(x) * 1e3
    if a_hull <= 1e-10:
        return np.ones_like(x) * 1e3
    
    areas = compute_all_areas(points)
    m = np.min(areas)
    shifted = areas - m
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    softmin_val = m - (1.0 / beta) * log_sum
    
    # Gradient of softmin w.r.t. areas
    weights = exp_terms / np.sum(exp_terms)  # softmax weights
    d_softmin_d_areas = -weights  # d/dareas of softmin
    
    # Gradient of areas w.r.t. points
    # area = 0.5 * |cross|, cross = (p2-p1) x (p3-p1)
    # For each triangle, compute gradient of cross product
    grad = np.zeros_like(x)
    
    for t in range(N_TRIS):
        i, j, k = TRIANGLE_IDX[t]
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross_val = v1[0] * v2[1] - v1[1] * v2[0]
        area_val = 0.5 * abs(cross_val)
        
        if area_val < 1e-15:
            continue
        
        # Sign of cross product
        sign = 1.0 if cross_val > 0 else -1.0
        
        # Gradient of cross_val w.r.t. p1, p2, p3
        # cross = (p2x - p1x)(p3y - p1y) - (p2y - p1y)(p3x - p1x)
        # d/dp1x = -(p3y - p1y) + (p2y - p1y) = p2y - p3y
        # d/dp1y = -(p2x - p1x) + (p3x - p1x) = p3x - p2x
        # d/dp2x = (p3y - p1y)
        # d/dp2y = -(p3x - p1x)
        # d/dp3x = -(p2y - p1y)
        # d/dp3y = (p2x - p1x)
        
        dp1x = p2[1] - p3[1]
        dp1y = p3[0] - p2[0]
        dp2x = p3[1] - p1[1]
        dp2y = -(p3[0] - p1[0])
        dp3x = -(p2[1] - p1[1])
        dp3y = (p2[0] - p1[0])
        
        # d(area)/d(points) = 0.5 * sign * d(cross)/d(points)
        coeff = 0.5 * sign * d_softmin_d_areas[t]
        
        grad[i, 0] += coeff * dp1x
        grad[i, 1] += coeff * dp1y
        grad[j, 0] += coeff * dp2x
        grad[j, 1] += coeff * dp2y
        grad[k, 0] += coeff * dp3x
        grad[k, 1] += coeff * dp3y
    
    # Chain rule: d(obj)/d(points) = d(obj)/d(softmin) * d(softmin)/d(areas) * d(areas)/d(points)
    # obj = -softmin / a_hull
    # d(obj)/d(softmin) = -1/a_hull
    # So total gradient = -1/a_hull * grad
    grad /= a_hull
    grad *= -1.0  # Wait, let me re-derive
    
    # Actually: obj = -softmin / a_hull
    # d(obj)/d(x) = -1/a_hull * d(softmin)/d(x)
    # d(softmin)/d(x) = sum_t d(softmin)/d(area_t) * d(area_t)/d(x)
    # = sum_t (-weights[t]) * d(area_t)/d(x)
    
    # So the gradient above already has the -weights included in coeff
    # Let me redo:
    grad = np.zeros_like(x)
    for t in range(N_TRIS):
        i, j, k = TRIANGLE_IDX[t]
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross_val = v1[0] * v2[1] - v1[1] * v2[0]
        area_val = 0.5 * abs(cross_val)
        
        if area_val < 1e-15:
            continue
        
        sign = 1.0 if cross_val > 0 else -1.0
        
        dp1x = p2[1] - p3[1]
        dp1y = p3[0] - p2[0]
        dp2x = p3[1] - p1[1]
        dp2y = -(p3[0] - p1[0])
        dp3x = -(p2[1] - p1[1])
        dp3y = (p2[0] - p1[0])
        
        # d(area_t)/d(points) = 0.5 * sign * d(cross)/d(points)
        coeff = 0.5 * sign * (-weights[t])  # -weights because d(softmin)/d(area) = -weights
        coeff /= a_hull  # divide by a_hull from obj = -softmin/a_hull
        
        grad[i, 0] += coeff * dp1x
        grad[i, 1] += coeff * dp1y
        grad[j, 0] += coeff * dp2x
        grad[j, 1] += coeff * dp2y
        grad[k, 0] += coeff * dp3x
        grad[k, 1] += coeff * dp3y
    
    # obj = -softmin / a_hull, so d(obj)/d(x) = -1/a_hull * d(softmin)/d(x)
    # d(softmin)/d(x) = sum_t d(softmin)/d(area_t) * d(area_t)/d(x) = sum_t (-w_t) * d(area_t)/d(x)
    # So d(obj)/d(x) = -1/a_hull * sum_t (-w_t) * d(area_t)/d(x) = 1/a_hull * sum_t w_t * d(area_t)/d(x)
    
    # Let me just use numerical gradient for safety
    eps = 1e-7
    grad_num = np.zeros_like(x)
    for d in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[d] += eps
        x_minus[d] -= eps
        grad_num[d] = (soft_min_objective(x_plus, beta) - soft_min_objective(x_minus, beta)) / (2 * eps)
    
    return grad_num

def bfgs_optimize(points, beta, time_limit=30.0, maxiter=1000):
    """
    Run BFGS optimization on the soft-min surrogate.
    """
    x0 = points.flatten()
    start_time = time.time()
    
    def objective(x):
        if time.time() > start_time + time_limit:
            raise RuntimeError("Time limit exceeded")
        return soft_min_objective(x, beta)
    
    try:
        result = minimize(
            objective,
            x0,
            method='BFGS',
            options={'maxiter': maxiter, 'gtol': 1e-8}
        )
        if result.success or result.fun < 1e5:
            return result.x.reshape(N_POINTS, 2)
    except Exception:
        pass
    
    return points

def hard_min_polish(points, time_limit=20.0, max_iters=500):
    """
    Direct hard-min polish: move vertices of min-area triangles apart.
    """
    points = points.copy()
    start_time = time.time()
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()
    
    for iteration in range(max_iters):
        if time.time() > start_time + time_limit:
            break
        
        areas = compute_all_areas(points)
        min_idx = np.argsort(areas)[:10]
        
        improved = False
        for tri_idx in min_idx[:5]:
            i, j, k = TRIANGLE_IDX[tri_idx]
            vi, vj, vk = points[i], points[j], points[k]
            centroid = (vi + vj + vk) / 3.0
            
            for idx in [i, j, k]:
                v = points[idx].copy()
                dir_vec = v - centroid
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm < 1e-10:
                    continue
                dir_vec /= dir_norm
                
                for step_frac in [0.01, 0.02, 0.05, 0.1]:
                    new_v = v + dir_vec * step_frac * dir_norm
                    new_points = points.copy()
                    new_points[idx] = new_v
                    new_score = compute_score(new_points)
                    if new_score > current_score:
                        points = new_points
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                for idx in [i, j, k]:
                    v = points[idx].copy()
                    dir_vec = centroid - v
                    dir_norm = np.linalg.norm(dir_vec)
                    if dir_norm < 1e-10:
                        continue
                    dir_vec /= dir_norm
                    
                    for step_frac in [0.01, 0.02, 0.05]:
                        new_v = v + dir_vec * step_frac * dir_norm
                        new_points = points.copy()
                        new_points[idx] = new_v
                        new_score = compute_score(new_points)
                        if new_score > current_score:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        if not improved and iteration > 20:
            break
    
    return best_points, best_score

def sa_polish(points, time_limit=30.0):
    """
    Simulated annealing polish.
    """
    points = points.copy()
    start_time = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()
    
    T = 0.05
    T_min = 1e-6
    alpha = 0.995
    step_size = 0.05
    
    while T > T_min and time.time() < start_time + time_limit:
        idx = rng.integers(0, N_POINTS)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        new_points = points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        
        new_score = compute_score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            delta = current_score - new_score
            if delta < 1e-10 or rng.random() < np.exp(-delta / T):
                points = new_points
                current_score = new_score
        
        T *= alpha
        step_size *= 0.9995
    
    return best_points, best_score

def generate_seeds():
    """
    Generate a variety of high-quality initial configurations.
    """
    seeds = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    seeds.append(pts)
    
    # 2. Regular 12-gon + Center
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    pts12_center = np.vstack([np.array([0.0, 0.0]), pts12])
    seeds.append(pts12_center)
    
    # 3. Two Concentric Rings (7 outer, 6 inner)
    ang7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    ang6 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    r_out = 1.0
    r_in = 0.6
    p_out = np.column_stack([r_out * np.cos(ang7), r_out * np.sin(ang7)])
    p_in = np.column_stack([r_in * np.cos(ang6), r_in * np.sin(ang6)])
    seeds.append(np.vstack([p_out, p_in]))
    
    # 4. Two Concentric Rings (6 outer, 7 inner)
    ang6b = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ang7b = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    p_out_b = np.column_stack([1.0 * np.cos(ang6b), 1.0 * np.sin(ang6b)])
    p_in_b = np.column_stack([0.6 * np.cos(ang7b), 0.6 * np.sin(ang7b)])
    seeds.append(np.vstack([p_out_b, p_in_b]))
    
    # 5. SDP-inspired: maximize minimum pairwise distances
    angles_init = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts_sdp = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    from scipy.spatial import cKDTree
    for _ in range(200):
        tree = cKDTree(pts_sdp)
        pairs = tree.query_pairs(0.3)
        if not pairs:
            break
        for i, j in list(pairs)[:5]:
            mid = (pts_sdp[i] + pts_sdp[j]) / 2
            dir_vec = pts_sdp[i] - pts_sdp[j]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-10:
                continue
            dir_vec /= dir_norm
            pts_sdp[i] += dir_vec * 0.02
            pts_sdp[j] -= dir_vec * 0.02
    seeds.append(pts_sdp)
    
    # 6. Distorted Grid (4x3 + 1)
    xs = np.array([0, 1, 2, 3])
    ys = np.array([0, 1, 2])
    grid_pts = []
    for x in xs:
        for y in ys:
            grid_pts.append([x, y])
    grid_pts.append([1.5, 1.0])
    grid_pts = np.array(grid_pts)
    grid_pts = (grid_pts - grid_pts.mean(axis=0))
    rng = np.random.default_rng(42)
    grid_pts += rng.normal(0, 0.2, size=grid_pts.shape)
    seeds.append(grid_pts)
    
    # 7. Alternating Radii (Star-like)
    angles_alt = np.linspace(0, 2*np.pi, 13, endpoint=False)
    radii_alt = np.ones(13)
    radii_alt[::2] = 1.0
    radii_alt[1::2] = 0.5
    pts_alt = np.column_stack([radii_alt * np.cos(angles_alt), radii_alt * np.sin(angles_alt)])
    seeds.append(pts_alt)
    
    # 8. Triangular lattice-inspired
    pts = []
    for i in range(5):
        for j in range(5 - i):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            pts.append([x, y])
    pts = np.array(pts)
    com = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - com, axis=1)
    indices_to_remove = np.argsort(dists)[:2]
    mask = np.ones(15, dtype=bool)
    mask[indices_to_remove] = False
    pts_13 = pts[mask]
    center = np.mean(pts_13, axis=0)
    pts_13 -= center
    max_norm = np.max(np.linalg.norm(pts_13, axis=1))
    pts_13 /= max_norm
    seeds.append(pts_13)
    
    return seeds

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses BFGS with soft-min surrogate for smooth optimization, then hard-min polish.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit
    
    seeds = generate_seeds()
    
    # Add random seeds for diversity
    rng_master = np.random.default_rng(12345)
    for _ in range(5):
        random_seed = rng_master.uniform(-1, 1, size=(13, 2))
        if compute_score(random_seed) > 0.1:
            seeds.append(random_seed)
    
    best_points_overall = None
    best_score_overall = -1.0
    
    n_seeds = len(seeds)
    time_per_seed = (deadline - start_time - 30.0) / n_seeds
    
    for i, seed in enumerate(seeds):
        if time.time() >= deadline - 30.0:
            break
        
        current_deadline = min(time.time() + time_per_seed, deadline - 30.0)
        
        # Phase 1: BFGS with soft-min surrogate (annealed beta)
        points = seed.copy()
        score = compute_score(points)
        
        # Anneal beta from low to high for progressively sharper optimization
        betas = [10, 50, 100, 200, 500]
        for beta in betas:
            remaining = current_deadline - time.time()
            if remaining < 5.0:
                break
            time_for_bfgs = min(remaining * 0.4, 25.0)
            if time_for_bfgs > 2.0:
                points = bfgs_optimize(points, beta, time_limit=time_for_bfgs, maxiter=500)
                new_score = compute_score(points)
                if new_score > score:
                    score = new_score
        
        # Phase 2: Hard-min polish
        remaining = current_deadline - time.time()
        if remaining > 5.0:
            time_for_polish = min(remaining * 0.4, 20.0)
            points, score = hard_min_polish(points, time_limit=time_for_polish)
        
        # Phase 3: SA polish
        remaining = current_deadline - time.time()
        if remaining > 5.0:
            time_for_sa = min(remaining * 0.5, 20.0)
            points, score = sa_polish(points, time_limit=time_for_sa)
        
        if score > best_score_overall:
            best_score_overall = score
            best_points_overall = points.copy()
    
    # Final polish on best found
    if best_points_overall is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            # BFGS with high beta
            time_for_bfgs = min(remaining * 0.4, 25.0)
            best_points_overall = bfgs_optimize(best_points_overall, 500, time_limit=time_for_bfgs, maxiter=500)
            
            # Hard polish
            remaining = deadline - time.time() - 5.0
            if remaining > 5.0:
                time_for_polish = min(remaining * 0.5, 20.0)
                best_points_overall, best_score_overall = hard_min_polish(best_points_overall, time_limit=time_for_polish)
            
            # SA
            remaining = deadline - time.time() - 5.0
            if remaining > 5.0:
                time_for_sa = min(remaining * 0.5, 15.0)
                best_points_overall, best_score_overall = sa_polish(best_points_overall, time_limit=time_for_sa)
    
    # Fallback
    if best_points_overall is None:
        best_points_overall = seeds[0]
    
    # Ensure validity
    if not np.all(np.isfinite(best_points_overall)):
        best_points_overall = seeds[0]
    
    try:
        hull = ConvexHull(best_points_overall)
        if hull.volume <= 1e-10:
            return seeds[0]
    except:
        return seeds[0]
    
    return best_points_overall
