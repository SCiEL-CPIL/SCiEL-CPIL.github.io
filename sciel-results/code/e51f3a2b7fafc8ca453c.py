# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import differential_evolution, minimize

def _compute_areas(points):
    """Compute all triangle areas and hull area for a set of points."""
    if len(points) < 3:
        return None, None
    if not np.all(np.isfinite(points)):
        return None, None
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return None, None
    if a_hull <= 1e-12:
        return None, None
    
    n = len(points)
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_flat = i_flat[mask]
    j_flat = j_flat[mask]
    k_flat = k_flat[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    return areas, a_hull

def _score_fast(points):
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    areas, a_hull = _compute_areas(points)
    if areas is None:
        return 0.0
    min_area = np.min(areas)
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _get_min_triangles(points, k=3):
    """Get indices of the k smallest triangles."""
    areas, a_hull = _compute_areas(points)
    if areas is None:
        return None, None
    
    n = len(points)
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_flat = i_flat[mask]
    j_flat = j_flat[mask]
    k_flat = k_flat[mask]
    
    # Get k smallest areas
    k = min(k, len(areas))
    min_indices = np.argsort(areas)[:k]
    
    return list(zip(i_flat[min_indices], j_flat[min_indices], k_flat[min_indices])), areas

def _delaunay_repair(points, max_iterations=500, tol=1e-8):
    """
    Delaunay triangulation repair strategy:
    1. Compute Delaunay triangulation
    2. Find minimum-area triangles
    3. Move vertices of min triangles toward centroid of adjacent triangles
    4. Iterate until convergence or improvement stalls
    """
    n = len(points)
    current_points = points.copy()
    current_score = _score_fast(current_points)
    
    for iteration in range(max_iterations):
        # Find minimum-area triangles
        min_tris, all_areas = _get_min_triangles(current_points, k=5)
        if min_tris is None:
            break
        
        # Compute Delaunay triangulation
        try:
            delaunay = Delaunay(current_points)
        except Exception:
            break
        
        # Build adjacency: for each triangle, find which other triangles share edges
        simplices = delaunay.simplices  # shape (n_simplices, 3)
        n_simplices = len(simplices)
        
        # For each point, collect the triangles it belongs to
        point_to_tris = {i: [] for i in range(n)}
        for idx, tri in enumerate(simplices):
            for v in tri:
                point_to_tris[v].append(idx)
        
        # Get the minimum-area triangle
        i_min, j_min, k_min = min_tris[0]
        
        # Find triangles sharing edges with the min triangle
        min_tri_set = set([i_min, j_min, k_min])
        adjacent_tris = []
        for idx, tri in enumerate(simplices):
            tri_set = set(tri)
            # Shares at least one edge (2 vertices) with min triangle
            if len(min_tri_set & tri_set) == 2 and idx not in [min_tris[0]]:
                adjacent_tris.append(idx)
        
        # Compute centroid of adjacent triangles
        if adjacent_tris:
            adjacent_centroids = np.array([
                (current_points[simplices[idx][0]] + 
                 current_points[simplices[idx][1]] + 
                 current_points[simplices[idx][2]]) / 3.0
                for idx in adjacent_tris
            ])
            target = np.mean(adjacent_centroids, axis=0)
        else:
            # No adjacent triangles, move toward hull center
            target = np.mean(current_points, axis=0)
        
        # Move the vertex of the min triangle that is farthest from target
        # (this should increase the area)
        dists = np.linalg.norm(
            np.array([current_points[i_min], current_points[j_min], current_points[k_min]]) - target,
            axis=1
        )
        farthest_local = np.argmax(dists)
        vertex_to_move = [i_min, j_min, k_min][farthest_local]
        
        # Compute step: move toward target by a fraction
        direction = target - current_points[vertex_to_move]
        step_norm = np.linalg.norm(direction)
        if step_norm < 1e-12:
            break
        direction /= step_norm
        
        # Try multiple step sizes
        improved = False
        for step_frac in [0.1, 0.05, 0.02, 0.01, 0.2, 0.3]:
            step_size = step_frac * step_norm
            trial_points = current_points.copy()
            trial_points[vertex_to_move] = current_points[vertex_to_move] + step_size * direction
            
            trial_score = _score_fast(trial_points)
            if trial_score > current_score + tol:
                current_points = trial_points
                current_score = trial_score
                improved = True
                break
        
        if not improved:
            # Try moving other vertices of the min triangle
            for vertex in [i_min, j_min, k_min]:
                if vertex == vertex_to_move:
                    continue
                direction = target - current_points[vertex]
                step_norm = np.linalg.norm(direction)
                if step_norm < 1e-12:
                    continue
                direction /= step_norm
                
                for step_frac in [0.1, 0.05, 0.02, 0.01]:
                    step_size = step_frac * step_norm
                    trial_points = current_points.copy()
                    trial_points[vertex] = current_points[vertex] + step_size * direction
                    
                    trial_score = _score_fast(trial_points)
                    if trial_score > current_score + tol:
                        current_points = trial_points
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Stagnation - try a small random perturbation
            rng = np.random.default_rng(seed=iteration + 12345)
            perturbation = rng.normal(size=(n, 2)) * 0.01
            trial_points = current_points + perturbation
            trial_score = _score_fast(trial_points)
            if trial_score > current_score:
                current_points = trial_points
                current_score = trial_score
            else:
                break
    
    return current_points, current_score

def _soft_min_score(points, beta=20.0):
    """
    Compute a smooth surrogate for A_min/A_hull using log-sum-exp.
    Maximizing this encourages ALL triangles to be large, not just the minimum.
    """
    areas, a_hull = _compute_areas(points)
    if areas is None:
        return -1e10
    if np.any(areas <= 1e-12):
        return -1e10
    
    # Normalize areas by hull area to make scale-invariant
    norm_areas = areas / a_hull
    
    # Soft minimum via log-sum-exp: 
    # soft_min(x) ≈ -log(sum(exp(-beta * x))) / beta
    # This is a smooth lower bound on min(x) that approaches min(x) as beta→∞
    neg_beta_x = -beta * norm_areas
    max_val = np.max(neg_beta_x)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val

def _negative_score_flat(flat_points):
    """Objective function for differential evolution (minimize negative score)."""
    points = flat_points.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score_fast(points)
    if s <= 0:
        return 1e10
    return -s

def _negative_soft_min_flat(flat_points, beta=20.0):
    """Objective for soft-min optimization (minimize negative soft-min score)."""
    points = flat_points.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _soft_min_score(points, beta=beta)
    if s <= -1e9:
        return 1e10
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using Delaunay triangulation repair strategy.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 13
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_pts)
    
    # 2. Hexagonal lattice pattern
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 3. Grid-based
    grid_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Multiple random initializations
    rng = np.random.default_rng(seed=12345)
    for seed_offset in range(8):
        rng_seed = np.random.default_rng(seed=1000 + seed_offset * 100)
        initializations.append(rng_seed.random((n, 2)) * 2 - 1)
    
    # 5. Fibonacci sphere-like distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = np.zeros((n, 2))
    for i in range(n):
        angle = i * golden_angle
        radius = 0.5 + 0.5 * (i / (n - 1))
        fib_pts[i] = [radius * np.cos(angle), radius * np.sin(angle)]
    initializations.append(fib_pts)
    
    # Evaluate all initializations and apply Delaunay repair
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        s = _score_fast(init_pts)
        update_best(init_pts, s)
        
        # Apply Delaunay repair to this initialization
        if time.time() < deadline - 30:
            repaired_pts, repaired_score = _delaunay_repair(init_pts, max_iterations=200)
            update_best(repaired_pts, repaired_score)
    
    # Run differential evolution from the best point found
    if best_points is not None and time.time() < deadline - 120:
        bounds = [(-2, 2)] * (2 * n)
        best_init = best_points.flatten()
        
        try:
            result = differential_evolution(
                _negative_score_flat,
                bounds,
                maxiter=50,
                popsize=25,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=True,
                init='sobol',
                x0=best_init,
                updating='immediate',
                callback=None
            )
            
            if result.success or result.fun < 0:
                de_points = result.x.reshape(n, 2)
                s = _score_fast(de_points)
                if s > 0:
                    update_best(de_points, s)
                    
                    # Apply Delaunay repair to DE result
                    if time.time() < deadline - 30:
                        repaired_pts, repaired_score = _delaunay_repair(de_points, max_iterations=200)
                        update_best(repaired_pts, repaired_score)
        except Exception:
            pass
    
    # Soft-minimum optimization for global pressure
    if best_points is not None and time.time() < deadline - 90:
        current = best_points.flatten().copy()
        
        for beta in [15.0, 30.0]:
            if time.time() > deadline - 90:
                break
            try:
                res = minimize(
                    _negative_soft_min_flat,
                    current,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 1e9:
                    pts = res.x.reshape(n, 2)
                    s = _score_fast(pts)
                    if s > best_score:
                        update_best(pts, s)
                        current = res.x.copy()
                        
                        # Apply Delaunay repair
                        if time.time() < deadline - 30:
                            repaired_pts, repaired_score = _delaunay_repair(pts, max_iterations=150)
                            update_best(repaired_pts, repaired_score)
            except Exception:
                pass
    
    # Final Delaunay repair polish with more iterations
    if best_points is not None and time.time() < deadline - 20:
        final_pts, final_score = _delaunay_repair(best_points, max_iterations=500, tol=1e-10)
        update_best(final_pts, final_score)
    
    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    final_score = _score_fast(best_points)
    if final_score <= 0:
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
