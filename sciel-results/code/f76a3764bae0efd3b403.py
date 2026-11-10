# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
N_TRI = len(TRI_IDX)

def compute_all_areas(points):
    """Compute areas of all 286 triangles. Returns (areas, A_hull)."""
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        A_hull = 1.0
    return areas, A_hull

def compute_score(points):
    """Hard Heilbronn score: A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    areas, A_hull = compute_all_areas(points)
    if A_hull <= 1e-12:
        return 0.0
    return np.min(areas) / A_hull

def soft_min_score(points, beta=50.0):
    """
    Smooth approximation of A_min / A_hull using soft-min.
    Higher beta = closer to true min.
    """
    areas, A_hull = compute_all_areas(points)
    if A_hull <= 1e-12:
        return 0.0
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # For numerical stability, subtract min first
    a_min = np.min(areas)
    exp_terms = np.exp(-beta * (areas - a_min))
    soft_min_val = a_min - np.log(np.sum(exp_terms)) / beta
    return soft_min_val / A_hull

def soft_min_gradient(points, beta=50.0):
    """
    Gradient of soft_min_score w.r.t. point coordinates.
    Returns gradient array of shape (N, 2).
    """
    areas, A_hull = compute_all_areas(points)
    if A_hull <= 1e-12:
        return np.zeros((N, 2))
    
    # Soft min computation
    a_min = np.min(areas)
    exp_terms = np.exp(-beta * (areas - a_min))
    sum_exp = np.sum(exp_terms)
    
    # d(soft_min)/d(area_i) = -exp(-beta*(area_i - a_min)) / sum_exp
    # = -exp_terms[i] / sum_exp
    d_softmin_d_area = -exp_terms / sum_exp
    
    # d(score)/d(area_i) = d_softmin_d_area[i] / A_hull
    d_score_d_area = d_softmin_d_area / A_hull
    
    # Now compute d(area_i)/d(points) for each triangle i
    # area_i = 0.5 * |cross(v1, v2)| where v1 = p2-p1, v2 = p3-p1
    # For the signed cross: cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # d(cross)/d(x1) = -(y3-y1) + (y2-y1)*0 ... let me be careful
    # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # d(cross)/dx1 = -(y3-y1)
    # d(cross)/dy1 = (x2-x1)
    # d(cross)/dx2 = (y3-y1)
    # d(cross)/dy2 = -(x3-x1)
    # d(cross)/dx3 = -(y2-y1)
    # d(cross)/dy3 = (x2-x1)
    
    # But area = 0.5 * |cross|, so d(area)/d(cross) = 0.5 * sign(cross)
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1x = p2[:, 0] - p1[:, 0]
    v1y = p2[:, 1] - p1[:, 1]
    v2x = p3[:, 0] - p1[:, 0]
    v2y = p3[:, 1] - p1[:, 1]
    
    cross_signed = v1x * v2y - v1y * v2x
    
    # d(area)/d(cross) = 0.5 * sign(cross)
    sign_cross = np.sign(cross_signed)
    sign_cross[np.abs(cross_signed) < 1e-12] = 1.0  # avoid zero
    d_area_d_cross = 0.5 * sign_cross
    
    # d(score)/d(point_j) for each triangle
    # For triangle i with vertices (a, b, c):
    # d(area_i)/dx_a = -0.5 * sign * (y_c - y_a)
    # d(area_i)/dy_a = 0.5 * sign * (x_b - x_a)
    # d(area_i)/dx_b = 0.5 * sign * (y_c - y_a)
    # d(area_i)/dy_b = -0.5 * sign * (x_c - x_a)
    # d(area_i)/dx_c = -0.5 * sign * (y_b - y_a)
    # d(area_i)/dy_c = 0.5 * sign * (x_b - x_a)
    
    # Accumulate gradient
    grad = np.zeros((N, 2))
    
    # For point a (index TRI_IDX[:,0]):
    da_dx = d_score_d_area * d_area_d_cross * (-v2y)
    da_dy = d_score_d_area * d_area_d_cross * (v1x)
    np.add.at(grad[:, 0], TRI_IDX[:, 0], da_dx)
    np.add.at(grad[:, 1], TRI_IDX[:, 0], da_dy)
    
    # For point b (index TRI_IDX[:,1]):
    db_dx = d_score_d_area * d_area_d_cross * (v2y)
    db_dy = d_score_d_area * d_area_d_cross * (-v2x)
    np.add.at(grad[:, 0], TRI_IDX[:, 1], db_dx)
    np.add.at(grad[:, 1], TRI_IDX[:, 1], db_dy)
    
    # For point c (index TRI_IDX[:,2]):
    dc_dx = d_score_d_area * d_area_d_cross * (-v1y)
    dc_dy = d_score_d_area * d_area_d_cross * (v1x)
    np.add.at(grad[:, 0], TRI_IDX[:, 2], dc_dx)
    np.add.at(grad[:, 1], TRI_IDX[:, 2], dc_dy)
    
    return grad

def gradient_ascent(points, time_limit, rng, beta_start=20.0, beta_end=100.0,
                     lr_start=0.05, lr_end=0.001, cooling=0.99995):
    """
    Gradient ascent on soft_min_score with annealing beta and learning rate.
    """
    current_pts = points.copy()
    current_score = soft_min_score(current_pts, beta_start)
    best_pts = current_pts.copy()
    best_score = compute_score(best_pts)
    
    beta = beta_start
    lr = lr_start
    n_iter = 0
    stall_count = 0
    stall_threshold = 500
    
    while time.time() < time_limit:
        # Compute gradient
        grad = soft_min_gradient(current_pts, beta)
        
        # Update
        current_pts = current_pts + lr * grad
        
        # Keep points in reasonable bounds
        current_pts = np.clip(current_pts, -5, 5)
        
        # Compute actual score
        new_score = compute_score(current_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = current_pts.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Anneal
        beta *= cooling
        lr *= cooling
        
        # If stalled, try a random perturbation
        if stall_count > stall_threshold:
            idx = rng.integers(N)
            delta = rng.normal(0, 0.1, size=2)
            current_pts[idx] += delta
            current_pts = np.clip(current_pts, -5, 5)
            stall_count = 0
        
        n_iter += 1
        if n_iter % 1000 == 0:
            # Periodically check if we're making progress
            pass
        
        if beta > beta_end:
            beta = beta_end
        if lr < lr_end:
            lr = lr_end
    
    return best_pts, best_score

def fast_local_search(points, time_limit, rng, step_size=0.01):
    """
    Fast local search that only recomputes affected triangles.
    """
    current_pts = points.copy()
    current_score = compute_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Precompute which triangles involve each point
    point_triangles = [[] for _ in range(N)]
    for i, tri in enumerate(TRI_IDX):
        for j in tri:
            point_triangles[j].append(i)
    
    n_iter = 0
    stall_count = 0
    
    while time.time() < time_limit:
        # Pick a random point
        idx = rng.integers(N)
        
        # Generate perturbation
        delta = rng.normal(0, step_size, size=2)
        old_val = current_pts[idx].copy()
        current_pts[idx] = old_val + delta
        current_pts[idx] = np.clip(current_pts[idx], -5, 5)
        
        # Recompute only affected triangles
        affected_tris = point_triangles[idx]
        p1 = current_pts[TRI_IDX[affected_tris, 0]]
        p2 = current_pts[TRI_IDX[affected_tris, 1]]
        p3 = current_pts[TRI_IDX[affected_tris, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        new_areas = np.abs(cross) * 0.5
        
        # Get current areas for comparison (approximate: use min of affected)
        # For efficiency, just compute full score occasionally
        if n_iter % 50 == 0:
            new_score = compute_score(current_pts)
        else:
            # Quick estimate: min of affected areas / current hull area
            try:
                hull = ConvexHull(current_pts)
                A_hull = hull.volume
            except Exception:
                current_pts[idx] = old_val
                continue
            if A_hull <= 1e-12:
                current_pts[idx] = old_val
                continue
            new_score = np.min(new_areas) / A_hull
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
                stall_count = 0
        else:
            # Accept with some probability (soft)
            if rng.random() < 0.1:
                current_score = new_score
            else:
                current_pts[idx] = old_val
            stall_count += 1
        
        # Adaptive step size
        if stall_count > 200:
            step_size *= 0.8
            stall_count = 0
        
        n_iter += 1
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0
    rng = np.random.default_rng(42)

    best_points = None
    best_score = -1.0

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 7 outer, 6 inner
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p7 = np.column_stack([np.cos(ang7), np.sin(ang7)])
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    p6 = np.column_stack([0.55 * np.cos(ang6), 0.55 * np.sin(ang6)])
    candidates.append(np.vstack([p7, p6]))
    
    # 3. 7 outer, 6 inner (offset)
    ang6b = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 12
    p6b = np.column_stack([0.6 * np.cos(ang6b), 0.6 * np.sin(ang6b)])
    candidates.append(np.vstack([p7, p6b]))
    
    # 4. 5/5/3
    ang5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5o = np.column_stack([np.cos(ang5), np.sin(ang5)])
    ang5m = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5m = np.column_stack([0.6 * np.cos(ang5m), 0.6 * np.sin(ang5m)])
    ang3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
    p3i = np.column_stack([0.3 * np.cos(ang3), 0.3 * np.sin(ang3)])
    candidates.append(np.vstack([p5o, p5m, p3i]))
    
    # 5. 4/5/4
    ang4o = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4o = np.column_stack([np.cos(ang4o), np.sin(ang4o)])
    ang5m2 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5m2 = np.column_stack([0.65 * np.cos(ang5m2), 0.65 * np.sin(ang5m2)])
    ang4i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i = np.column_stack([0.35 * np.cos(ang4i), 0.35 * np.sin(ang4i)])
    candidates.append(np.vstack([p4o, p5m2, p4i]))
    
    # 6. 6 outer, 7 inner
    ang6o = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p6o = np.column_stack([np.cos(ang6o), np.sin(ang6o)])
    ang7i = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    p7i = np.column_stack([0.5 * np.cos(ang7i), 0.5 * np.sin(ang7i)])
    candidates.append(np.vstack([p6o, p7i]))
    
    # 7. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(N) / (golden ** 2)
    fib_radii = np.sqrt(np.arange(N) / N)
    candidates.append(np.column_stack([fib_radii * np.cos(fib_angles),
                                       fib_radii * np.sin(fib_angles)]))
    
    # 8. 4/5/4 variant
    ang4a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4a = np.column_stack([np.cos(ang4a), np.sin(ang4a)])
    ang5b = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5b = np.column_stack([0.65 * np.cos(ang5b), 0.65 * np.sin(ang5b)])
    ang4c = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4c = np.column_stack([0.3 * np.cos(ang4c), 0.3 * np.sin(ang4c)])
    candidates.append(np.vstack([p4a, p5b, p4c]))
    
    # 9. 5/4/4
    ang5o2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 10
    p5o2 = np.column_stack([np.cos(ang5o2), np.sin(ang5o2)])
    ang4m2 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4m2 = np.column_stack([0.6 * np.cos(ang4m2), 0.6 * np.sin(ang4m2)])
    ang4i2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i2 = np.column_stack([0.3 * np.cos(ang4i2), 0.3 * np.sin(ang4i2)])
    candidates.append(np.vstack([p5o2, p4m2, p4i2]))
    
    # 10. Random initializations
    for _ in range(8):
        candidates.append(rng.random((N, 2)) * 2 - 1)
    
    # Perturbed structured
    for base_idx in [0, 1, 3, 4, 6]:
        if base_idx < len(candidates):
            candidates.append(candidates[base_idx] + rng.normal(0, 0.03, size=(N, 2)))

    n_cand = len(candidates)
    
    # Phase 1: Quick evaluation of all candidates
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline - 5:
            break
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()

    # Phase 2: Gradient ascent on top candidates
    top_candidates = []
    for i, init_pts in enumerate(candidates):
        s = compute_score(init_pts)
        top_candidates.append((s, i))
    top_candidates.sort(reverse=True)
    
    # Run gradient ascent on top 5 candidates
    for rank, (score, idx) in enumerate(top_candidates[:5]):
        if time.time() >= deadline - 10:
            break
        remaining = deadline - 10 - time.time()
        time_per_run = min(remaining / max(5 - rank, 1), 120.0)
        run_deadline = time.time() + time_per_run
        
        ga_pts, ga_score = gradient_ascent(candidates[idx], run_deadline, rng)
        if ga_score > best_score:
            best_score = ga_score
            best_points = ga_pts.copy()

    # Phase 3: Fast local search polishing
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        ls_deadline = time.time() + remaining
        ls_pts, ls_score = fast_local_search(best_points, ls_deadline, rng)
        if ls_score > best_score:
            best_score = ls_score
            best_points = ls_pts.copy()

    # Phase 4: Final gradient polish
    if best_points is not None and time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        final_deadline = time.time() + remaining
        final_pts, final_score = gradient_ascent(best_points, final_deadline, rng,
                                                   beta_start=50.0, beta_end=150.0,
                                                   lr_start=0.01, lr_end=0.0005)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
# EVOLVE-BLOCK-END
