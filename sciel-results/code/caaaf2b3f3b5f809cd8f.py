# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import itertools

N = 13
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull (vectorized)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def delaunay_repair(points, time_limit, max_iter=5000):
    """
    Delaunay-based local repair: iteratively move vertices incident to
    minimum-area triangles toward the centroid of their adjacent triangles
    to increase A_min while preserving hull structure.
    """
    points = points.copy()
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()
    
    no_improve = 0
    step_scale = 0.3
    
    for iteration in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Compute all triangle areas
        p1 = points[TRI_IDX[:, 0]]
        p2 = points[TRI_IDX[:, 1]]
        p3 = points[TRI_IDX[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) * 0.5
        
        # Find minimum area triangle(s)
        min_area = np.min(areas)
        if min_area <= 1e-10:
            break
        
        # Get indices of triangles with area below threshold (bottom 10%)
        threshold = min_area * 1.1
        low_tri_indices = np.where(areas < threshold)[0]
        
        if len(low_tri_indices) == 0:
            break
        
        # Collect vertices that are in low-area triangles
        low_vertices = set()
        for tri_idx in low_tri_indices:
            tri = TRI_IDX[tri_idx]
            low_vertices.add(tri[0])
            low_vertices.add(tri[1])
            low_vertices.add(tri[2])
        
        # Compute Delaunay triangulation to find adjacent triangles
        try:
            tri = Delaunay(points)
        except Exception:
            # Fallback: just move toward overall centroid
            centroid = points.mean(axis=0)
            for v in low_vertices:
                points[v] += step_scale * (centroid - points[v])
            new_score = compute_score(points)
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                step_scale *= 0.95
            continue
        
        # For each low-area vertex, find its adjacent triangles and move toward their centroid
        for v in low_vertices:
            # Find all Delaunay triangles containing vertex v
            adj_tris = []
            for t_idx in range(len(tri.simplices)):
                simplex = tri.simplices[t_idx]
                if v in simplex:
                    adj_tris.append(simplex)
            
            if len(adj_tris) == 0:
                continue
            
            # Compute centroid of adjacent triangles
            adj_centroid = np.mean([points[t].mean(axis=0) for t in adj_tris], axis=0)
            
            # Move vertex toward adjacent centroid
            direction = adj_centroid - points[v]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                points[v] += step_scale * direction / norm * norm
        
        new_score = compute_score(points)
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            no_improve = 0
        else:
            no_improve += 1
            step_scale *= 0.97
            # Revert to best if no improvement for a while
            if no_improve > 50:
                points = best_points.copy()
                step_scale = 0.3
                no_improve = 0
        
        if step_scale < 1e-6:
            break
    
    return best_points, best_score

def nelder_mead_optimize(points, time_limit, temperature=0.01, max_iter=5000):
    """
    Nelder-Mead optimization on the soft-min surrogate, then refine with hard min.
    """
    points = points.copy()
    n = len(points)
    flat = points.flatten()
    
    # Create initial simplex
    simplex = [flat.copy()]
    step = 0.1
    for i in range(2 * n):
        p = flat.copy()
        p[i] += step
        simplex.append(p)
    
    def objective(flat_arr):
        pts = flat_arr.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        try:
            hull = ConvexHull(pts)
            A_hull = hull.volume
        except Exception:
            return 1e10
        if A_hull <= 0:
            return 1e10
        
        p1 = pts[TRI_IDX[:, 0]]
        p2 = pts[TRI_IDX[:, 1]]
        p3 = pts[TRI_IDX[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = np.abs(cross) * 0.5
        
        scaled = -areas / temperature
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_area = -temperature * log_sum
        
        return -soft_min_area / A_hull
    
    best_flat = flat.copy()
    best_val = objective(flat)
    
    # Nelder-Mead iterations
    for iteration in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Sort simplex
        vals = [objective(s) for s in simplex]
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        
        best_val = vals[0]
        if best_val < objective(best_flat):
            best_flat = simplex[0].copy()
        
        # Centroid of all but worst
        worst = simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        reflected = centroid + alpha * (centroid - worst)
        ref_val = objective(reflected)
        
        if ref_val < vals[0]:
            # Expansion
            gamma = 2.0
            expanded = centroid + gamma * (centroid - worst)
            exp_val = objective(expanded)
            if exp_val < ref_val:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        elif ref_val < vals[-2]:
            simplex[-1] = reflected
        else:
            # Contraction
            rho = 0.5
            contracted = centroid + rho * (worst - centroid)
            con_val = objective(contracted)
            if con_val < vals[-1]:
                simplex[-1] = contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    
    # Convert back to points
    best_points = best_flat.reshape(n, 2)
    
    # Polish with hard min via local search
    hard_score = compute_score(best_points)
    
    return best_points, hard_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    
    Uses Delaunay-based constrained repair as the primary strategy, with
    random perturbation and Nelder-Mead as fallbacks.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Regular 13-gon (primary for Delaunay repair)
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts1)
    
    # Candidate 2: Regular 12-gon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts2 = np.column_stack([np.cos(angles), np.sin(angles)])
    pts2 = np.vstack([np.array([0.0, 0.0]), pts2])
    candidates.append(pts2)
    
    # Candidate 3: Two concentric hexagons + center
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts_h1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    angles2 = angles1 + np.pi/6
    pts_h2 = np.column_stack([1.0 * np.cos(angles2), 1.0 * np.sin(angles2)])
    pts3 = np.vstack([np.array([0.0, 0.0]), pts_h1, pts_h2])
    candidates.append(pts3)
    
    # Candidate 4: Two concentric polygons (7 and 6)
    def make_concentric_polygons(n1=7, n2=6, r1=1.0, r2=0.5):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([p1, p2])
    
    candidates.append(make_concentric_polygons(7, 6, 1.0, 0.5))
    candidates.append(make_concentric_polygons(6, 7, 1.0, 0.6))
    
    # Candidate 5: Fibonacci-like distribution
    def make_fibonacci_2d():
        n = 13
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(n) / golden_ratio
        radii = np.sqrt(np.arange(n) / n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    candidates.append(make_fibonacci_2d())
    
    # Candidate 6: Grid-like arrangement
    def make_grid_arrangement():
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0, 1, 2, 3])
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) < 13:
                    points.append([i, j])
        points = np.array(points)
        points = (points - points.mean(axis=0)) / points.std(axis=0)
        return points
    
    candidates.append(make_grid_arrangement())
    
    # Add random perturbations to structured configs
    rng = np.random.default_rng(12345)
    for base in candidates[:3]:
        perturb = rng.normal(0, 0.05, size=base.shape)
        candidates.append(base + perturb)
    
    # Add some fully random starts
    for _ in range(5):
        pts_rand = rng.random((13, 2)) * 2 - 1
        candidates.append(pts_rand)
    
    # Candidate: Ring with jittered inner points
    def make_ring_inner():
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        inner = np.column_stack([0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)])
        return np.vstack([outer, inner])
    
    candidates.append(make_ring_inner())
    candidates.append(make_ring_inner() + rng.normal(0, 0.03, size=(13,2)))
    
    # Candidate: Alternating radii pattern
    def make_alternating():
        angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
        radii = np.ones(13)
        radii[::2] = 1.0
        radii[1::2] = 0.6
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts
    
    candidates.append(make_alternating())
    candidates.append(make_alternating() + rng.normal(0, 0.03, size=(13,2)))
    
    # Local search function combining Delaunay repair and random perturbation
    def local_search(points, time_limit):
        points = points.copy()
        current_score = compute_score(points)
        best_score_local = current_score
        best_points_local = points.copy()
        
        # Phase 1: Delaunay-based repair (primary)
        if time.time() < time_limit * 0.6:
            dl_deadline = time_limit * 0.6
            dl_pts, dl_score = delaunay_repair(points, dl_deadline, max_iter=3000)
            if dl_score > best_score_local:
                best_score_local = dl_score
                best_points_local = dl_pts
        
        # Phase 2: Random perturbation local search
        step_size = 0.1
        no_improve_count = 0
        rng_local = np.random.default_rng(seed=int(time.time() * 1000) % 100000)
        
        if time.time() < time_limit * 0.8:
            while time.time() < time_limit * 0.8:
                improved = False
                for idx in range(13):
                    dx = rng_local.uniform(-step_size, step_size)
                    dy = rng_local.uniform(-step_size, step_size)
                    
                    new_points = best_points_local.copy()
                    new_points[idx, 0] += dx
                    new_points[idx, 1] += dy
                    
                    new_score = compute_score(new_points)
                    
                    if new_score > best_score_local:
                        best_points_local = new_points
                        best_score_local = new_score
                        improved = True
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                
                if not improved:
                    step_size *= 0.95
                    if step_size < 1e-6:
                        break
        
        # Phase 3: Nelder-Mead polish
        if time.time() < time_limit - 5:
            nm_deadline = time_limit - 5
            nm_pts, nm_score = nelder_mead_optimize(best_points_local, nm_deadline, temperature=0.005)
            if nm_score > best_score_local:
                best_score_local = nm_score
                best_points_local = nm_pts
        
        # Phase 4: Final fine-tuning with smaller steps
        if time.time() < time_limit - 2:
            step_size = 0.01
            points = best_points_local.copy()
            current_score = best_score_local
            for _ in range(100):
                if time.time() > time_limit - 2:
                    break
                improved = False
                for idx in range(13):
                    dx = rng_local.uniform(-step_size, step_size)
                    dy = rng_local.uniform(-step_size, step_size)
                    
                    new_points = points.copy()
                    new_points[idx, 0] += dx
                    new_points[idx, 1] += dy
                    
                    new_score = compute_score(new_points)
                    
                    if new_score > current_score:
                        points = new_points
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score_local:
                            best_score_local = current_score
                            best_points_local = points.copy()
                    else:
                        pass
                
                if not improved:
                    step_size *= 0.8
                    if step_size < 1e-7:
                        break
        
        return best_points_local, best_score_local
    
    # Run local search on each candidate
    n_candidates = len(candidates)
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline:
            break
            
        remaining_time = deadline - time.time()
        remaining_candidates = n_candidates - i
        candidate_time = remaining_time / remaining_candidates
        candidate_deadline = min(deadline, time.time() + candidate_time)
        
        res_pts, res_score = local_search(init_pts, candidate_deadline)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
    
    # If we have time left, do a final polish on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        res_pts, res_score = local_search(best_points, deadline - 10.0)
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
    
    # Final fallback
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
