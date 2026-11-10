import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area.
    
    We use a Delaunay-based physics relaxation:
    1. Generate diverse initial configurations.
    2. For each, run a force-directed relaxation that:
       - Computes the Delaunay triangulation
       - Applies repulsive forces based on small triangles (inverse area weighting)
       - Applies attractive forces to maintain Delaunay structure (Laplacian-like)
       - Adapts step size based on score improvement
    3. Polish the best results using Nelder-Mead on the hard objective.
    4. Return the best configuration found within the time limit.
    """
    n = 13
    start_time = time.time()
    time_limit = 585  # Safety margin before 630s limit
    
    # Precompute indices for triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    num_tris = len(tri_i)  # 286

    def get_areas(points):
        """Compute areas of all 286 triangles."""
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return areas

    def get_hull_area(points):
        """Compute area of the convex hull."""
        try:
            hull = scipy.spatial.ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def compute_score(points):
        """Compute the ratio min_triangle_area / hull_area."""
        if not np.all(np.isfinite(points)):
            return 0.0
        hull_area = get_hull_area(points)
        if hull_area < 1e-12:
            return 0.0
        areas = get_areas(points)
        min_area = np.min(areas)
        if min_area < 1e-12:
            return 0.0
        return min_area / hull_area

    def delaunay_forces(points, temperature=1.0):
        """
        Compute forces based on Delaunay triangulation.
        Points in small triangles get repelled; points are attracted toward
        the centroid of their Delaunay neighbors (Laplacian smoothing).
        """
        try:
            tri = scipy.spatial.Delaunay(points)
        except Exception:
            return np.zeros_like(points)

        forces = np.zeros_like(points)
        
        # Compute areas of Delaunay triangles
        tri_points = points[tri.simplices]  # shape (num_tris, 3, 2)
        v1 = tri_points[:, 1] - tri_points[:, 0]
        v2 = tri_points[:, 2] - tri_points[:, 0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        tri_areas = 0.5 * np.abs(cross)
        
        # Weight: inverse area (smaller triangles get larger weight)
        # Use temperature to control the sharpness
        weights = 1.0 / (tri_areas + 1e-8)
        weights = weights / (np.sum(weights) + 1e-12)
        
        # For each triangle, compute the circumcenter and apply forces
        # Points in small triangles should move away from each other
        for t in range(len(tri.simplices)):
            i, j, k = tri.simplices[t]
            w = weights[t]
            
            # Circumcenter of the triangle
            ax, ay = points[i]
            bx, by = points[j]
            cx, cy = points[k]
            
            d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            if abs(d) < 1e-12:
                continue
                
            ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
            uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
            
            circumcenter = np.array([ux, uy])
            
            # Force: move points away from circumcenter if triangle is small
            # This helps separate points in small triangles
            for idx_pt in [i, j, k]:
                to_center = circumcenter - points[idx_pt]
                dist = np.linalg.norm(to_center)
                if dist > 1e-12:
                    # Repel from circumcenter, scaled by weight
                    forces[idx_pt] += w * to_center / dist * temperature
        
        # Laplacian smoothing: attract each point toward the centroid of its Delaunay neighbors
        # Build neighbor list
        neighbor_sums = np.zeros_like(points)
        neighbor_counts = np.zeros(n)
        
        for simplex in tri.simplices:
            i, j, k = simplex
            neighbor_sums[i] += points[j] + points[k]
            neighbor_sums[j] += points[i] + points[k]
            neighbor_sums[k] += points[i] + points[j]
            neighbor_counts[i] += 2
            neighbor_counts[j] += 2
            neighbor_counts[k] += 2
        
        # Add Laplacian force
        for i in range(n):
            if neighbor_counts[i] > 0:
                centroid = neighbor_sums[i] / neighbor_counts[i]
                forces[i] += (centroid - points[i]) * 0.1 * temperature
        
        return forces

    def run_physics_relaxation(initial_points, max_iters=500, base_lr=0.05):
        """
        Run physics-based relaxation with adaptive learning rate.
        """
        points = initial_points.copy()
        best_points = points.copy()
        best_score = compute_score(points)
        
        # Normalize initial configuration
        center = np.mean(points, axis=0)
        points = points - center
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 1e-12:
            points = points / scale
        best_points = points.copy()
        best_score = compute_score(points)
        
        lr = base_lr
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 30:
                break
            
            # Compute forces
            forces = delaunay_forces(points, temperature=1.0)
            
            # Normalize forces
            force_norm = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norm)
            if max_force > 1e-12:
                forces = forces / max_force
            
            # Update positions
            new_points = points + lr * forces
            
            # Re-center and scale
            center = np.mean(new_points, axis=0)
            new_points = new_points - center
            scale = np.max(np.linalg.norm(new_points, axis=1))
            if scale > 1e-12:
                new_points = new_points / scale
            
            # Evaluate
            new_score = compute_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                points = new_points.copy()
                no_improve_count = 0
                # Slightly increase learning rate on improvement
                lr = min(lr * 1.05, 0.2)
            else:
                no_improve_count += 1
                # Decrease learning rate when stuck
                lr = max(lr * 0.95, 0.005)
                
                if no_improve_count > 50:
                    # Try a small random perturbation
                    points = points + np.random.normal(0, 0.01, (n, 2))
                    no_improve_count = 0
        
        return best_points, best_score

    def hard_objective(params):
        """Objective function to minimize: - (min_area / hull_area)"""
        points = params.reshape((n, 2))
        if not np.all(np.isfinite(points)):
            return 1e10
        hull_area = get_hull_area(points)
        if hull_area < 1e-9:
            return 1e10
        areas = get_areas(points)
        min_area = np.min(areas)
        if min_area < 1e-12:
            return 1e10
        return -(min_area / hull_area)

    # Initial configurations
    rng = np.random.default_rng(seed=42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init1 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Square-ish configuration
    init2 = np.array([
        [-1, -1], [1, -1], [-1, 1], [1, 1],
        [0, -1], [0, 1], [-1, 0], [1, 0],
        [0, 0], [0.5, 0.5], [-0.5, -0.5],
        [0.5, -0.5], [-0.5, 0.5]
    ])
    
    # 3. Random in square
    init3 = rng.uniform(-1, 1, (n, 2))
    
    # 4. Random in disk
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    init4 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    
    # 5. Jittered regular polygon
    init5 = init1 + rng.normal(0, 0.15, (n, 2))
    
    # 6. Grid-like
    grid = np.array([[x, y] for x in [-1.2, -0.4, 0.4, 1.2] for y in [-1.2, -0.4, 0.4, 1.2]])
    init6 = grid[:13]
    
    # 7. Another random
    init7 = rng.uniform(-1, 1, (n, 2))
    
    # 8. Jittered grid
    init8 = grid[:13] + rng.normal(0, 0.2, (13, 2))
    
    # 9. Hexagonal-ish
    init9 = np.array([
        [0, 0], [1, 0], [-1, 0], [0.5, 0.866], [-0.5, 0.866], [0.5, -0.866],
        [-0.5, -0.866], [2, 0], [-2, 0], [1.5, 0.866], [-1.5, 0.866],
        [1.5, -0.866], [-1.5, -0.866]
    ])
    
    initializations = [init1, init2, init3, init4, init5, init6, init7, init8, init9]
    
    best_global_points = None
    best_global_score = -1e10
    
    # Run physics relaxation on each initialization
    for init in initializations:
        if time.time() - start_time > time_limit - 60:
            break
        
        points, score = run_physics_relaxation(init, max_iters=400, base_lr=0.04)
        
        if score > best_global_score:
            best_global_score = score
            best_global_points = points.copy()
    
    # If we still have time, do a final polish with Nelder-Mead on the hard objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30 and best_global_points is not None:
        try:
            res_final = scipy.optimize.minimize(
                hard_objective, 
                best_global_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            final_points = res_final.x.reshape((n, 2))
            final_score = compute_score(final_points)
            
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_points = final_points.copy()
                
        except Exception:
            pass
    
    # Final check and return
    if best_global_points is None:
        # Fallback to regular polygon
        best_global_points = init1
    
    final_points = best_global_points
    
    # Ensure validity
    if not np.all(np.isfinite(final_points)):
        final_points = init1
    
    return final_points

# EVOLVE-BLOCK-END
