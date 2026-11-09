import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import itertools
import time

# Precompute triangle indices for 13 points
N_POINTS = 13
TRIANGLE_IDX = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0.0 if the points are collinear or invalid.
    """
    if len(points) < 3:
        return 0.0
    
    # Check for valid hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-10:
        return 0.0
        
    # Calculate all triangle areas efficiently using precomputed indices
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def compute_all_areas(points):
    """Compute all 286 triangle areas and return them along with indices."""
    p1 = points[TRIANGLE_IDX[:, 0]]
    p2 = points[TRIANGLE_IDX[:, 1]]
    p3 = points[TRIANGLE_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    return areas

def get_min_area_triangles(points, k=5):
    """Get the k smallest area triangles."""
    areas = compute_all_areas(points)
    min_idx = np.argsort(areas)[:k]
    return min_idx, areas[min_idx]

def delaunay_repair(points, max_iters=500, time_limit=30.0):
    """
    Delaunay-based repair strategy:
    1. Compute Delaunay triangulation
    2. Identify minimum-area triangles
    3. Move vertices of min-area triangles away from each other
    4. Repeat until convergence or time limit
    """
    points = points.copy()
    start_time = time.time()
    current_score = compute_score(points)
    best_score = current_score
    best_points = points.copy()
    
    for iteration in range(max_iters):
        if time.time() > time_limit + start_time:
            break
        
        # Get the smallest area triangles
        min_tri_idx, min_areas = get_min_area_triangles(points, k=10)
        
        if len(min_tri_idx) == 0:
            break
        
        # For each of the worst triangles, try to increase its area
        # by moving its vertices apart
        improved = False
        for tri_idx in min_tri_idx[:5]:  # Focus on worst 5
            i, j, k = TRIANGLE_IDX[tri_idx]
            
            # Get the three vertices
            vi, vj, vk = points[i], points[j], points[k]
            
            # Compute centroid of this triangle
            centroid = (vi + vj + vk) / 3.0
            
            # Try moving each vertex slightly away from the centroid
            # to increase the triangle area
            for idx in [i, j, k]:
                v = points[idx].copy()
                
                # Direction away from centroid
                dir_vec = v - centroid
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm < 1e-10:
                    continue
                dir_vec /= dir_norm
                
                # Try a small step in that direction
                for step_frac in [0.01, 0.02, 0.05, 0.1]:
                    new_v = v + dir_vec * step_frac * np.linalg.norm(v - centroid)
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
                # Try moving vertices in opposite direction (toward each other)
                # This can sometimes help by reshaping
                for idx in [i, j, k]:
                    v = points[idx].copy()
                    centroid = (vi + vj + vk) / 3.0
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
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Early termination if no improvement for many iterations
        if not improved and iteration > 10:
            break
        
        # If we've been stuck, try a random perturbation
        if not improved and iteration % 50 == 0:
            rng = np.random.default_rng(iteration)
            for _ in range(3):
                idx = rng.integers(0, N_POINTS)
                dx = rng.normal(0, 0.05)
                dy = rng.normal(0, 0.05)
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
                    break
    
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
    # Start with regular polygon and then iteratively push apart close pairs
    angles_init = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts_sdp = np.column_stack([np.cos(angles_init), np.sin(angles_init)])
    for _ in range(200):
        tree = cKDTree(pts_sdp)
        pairs = tree.query_pairs(0.3)  # Find pairs closer than 0.3
        if not pairs:
            break
        for i, j in pairs[:5]:  # Fix worst 5 close pairs
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
    
    return seeds

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    Uses Delaunay triangulation repair strategy from multiple diverse seeds.
    """
    start_time = time.time()
    # Safety margin: return 10s before the 600s limit
    deadline = start_time + 590.0
    
    seeds = generate_seeds()
    
    best_points_overall = None
    best_score_overall = -1.0
    
    # Add random seeds to increase diversity
    rng_master = np.random.default_rng(12345)
    for _ in range(5):
        random_seed = rng_master.uniform(-1, 1, size=(13, 2))
        if compute_score(random_seed) > 0.1:
            seeds.append(random_seed)

    n_seeds = len(seeds)
    # Allocate time per seed
    time_per_seed = (deadline - start_time - 20.0) / n_seeds 
    
    for i, seed in enumerate(seeds):
        if time.time() >= deadline - 20.0:
            break
            
        current_deadline = time.time() + time_per_seed
        current_deadline = min(current_deadline, deadline - 20.0)
        
        # Run Delaunay repair
        time_for_repair = min(time_per_seed * 0.7, 30.0)
        optimized_pts, score = delaunay_repair(seed, max_iters=200, time_limit=time_for_repair)
        
        # Then do a final SA polish with remaining time
        remaining = current_deadline - time.time()
        if remaining > 5.0:
            sa_rng = np.random.default_rng(i * 1000 + 42)
            T = 0.05
            T_min = 1e-6
            alpha = 0.99
            step_size = 0.05
            
            points = optimized_pts.copy()
            current_score = score
            best_local = score
            best_local_pts = points.copy()
            
            while T > T_min and time.time() < current_deadline:
                idx = sa_rng.integers(0, N_POINTS)
                dx = sa_rng.normal(0, step_size)
                dy = sa_rng.normal(0, step_size)
                
                new_points = points.copy()
                new_points[idx, 0] += dx
                new_points[idx, 1] += dy
                
                new_score = compute_score(new_points)
                
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
                    if current_score > best_local:
                        best_local = current_score
                        best_local_pts = points.copy()
                else:
                    delta = current_score - new_score
                    if sa_rng.random() < np.exp(-delta / T):
                        points = new_points
                        current_score = new_score
                
                T *= alpha
                step_size *= 0.9995
            
            if best_local > score:
                optimized_pts = best_local_pts
                score = best_local
        
        if score > best_score_overall:
            best_score_overall = score
            best_points_overall = optimized_pts
    
    # Final polish on best found
    if best_points_overall is not None and time.time() < deadline - 5.0:
        final_deadline = deadline - 5.0
        time_for_final = min(final_deadline - time.time(), 30.0)
        final_pts, final_score = delaunay_repair(best_points_overall, max_iters=100, time_limit=time_for_final)
        if final_score > best_score_overall:
            best_score_overall = final_score
            best_points_overall = final_pts
    
    # Fallback
    if best_points_overall is None:
        best_points_overall = seeds[0]
        
    # Ensure validity
    if not np.all(np.isfinite(best_points_overall)):
        best_points_overall = seeds[0]
        
    # One last check
    try:
        hull = ConvexHull(best_points_overall)
        if hull.volume <= 1e-10:
            return seeds[0]
    except:
        return seeds[0]

    return best_points_overall
