import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Initialize with diverse seeds (Icosahedron + noise, random sphere).
    2. Use Simulated Annealing to explore the configuration space.
       - Objective: Maximize a soft-min of distances (surrogate for d_min) 
         while penalizing large d_max.
       - Specifically, maximize: (sum(exp(-d_i^2 / T))^(-1/T)) / d_max
         or simply use the true score (d_min/d_max)^2 with a high acceptance 
         rate for small improvements and occasional large jumps.
    3. The true score is non-smooth, but SA handles this well.
    4. Time-budgeted to finish well before the 630s limit.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # 30s margin

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        return d_min_sq / d_max_sq

    def generate_initial_points():
        """Generate a set of diverse initial configurations."""
        inits = []
        
        # 1. Icosahedron-based
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        
        # Add 2 more points to make 14
        # Place them near the "gaps" or just random perturbations of existing
        # For a simple start, duplicate a vertex with small offset or add poles
        extra1 = np.array([0, 0, 1.0])
        extra2 = np.array([0, 0, -1.0])
        # Scale ico to unit sphere, add extras
        pts_ico = np.vstack([ico_verts, extra1.reshape(1,3), extra2.reshape(1,3)])
        # Normalize to have max distance ~ 2 (diameter of sphere)
        # Actually, let's just keep them on unit sphere, score is scale invariant
        inits.append(pts_ico)
        
        # 2. Random points on sphere
        for _ in range(5):
            pts = np.random.randn(n, d)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            inits.append(pts)
            
        # 3. Icosahedron with noise
        for _ in range(5):
            noisy_ico = ico_verts + np.random.randn(12, 3) * 0.1
            noisy_ico /= np.linalg.norm(noisy_ico, axis=1, keepdims=True)
            extra1 = np.random.randn(3)
            extra1 /= np.linalg.norm(extra1)
            extra2 = np.random.randn(3)
            extra2 /= np.linalg.norm(extra2)
            pts_noisy = np.vstack([noisy_ico, extra1.reshape(1,3), extra2.reshape(1,3)])
            inits.append(pts_noisy)
            
        return inits

    def simulated_annealing(points, max_iters, start_temp, end_temp):
        """
        Perform Simulated Annealing on the point configuration.
        """
        current_points = points.copy()
        current_score = true_score(current_points)
        
        best_points = current_points.copy()
        best_score = current_score
        
        # Precompute indices for faster distance updates? 
        # For n=14, 91 pairs, full recalculation is fast enough in numpy.
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Temperature schedule: Geometric or Linear
            # T = start_temp * (end_temp/start_temp)^(i/max_iters)
            T = start_temp * (end_temp / start_temp) ** (i / max_iters)
            
            # Generate a neighbor
            # Strategy: Pick a random point and perturb it slightly
            # Or pick a random pair and adjust distance?
            # Simpler: Perturb a random point by a small Gaussian noise.
            # The scale of noise should depend on T.
            # Since we are on a sphere-ish manifold, let's project back to sphere?
            # No, the problem doesn't require points to be on a sphere.
            # However, optimal configurations for packing/distance ratios are often on a sphere.
            # Let's allow free movement but normalize scale occasionally?
            # Scale invariance: (d_min/d_max) is invariant to scaling.
            # So we can fix the scale, e.g., d_max = 1 or sum of squares = 1.
            # Let's fix the scale by normalizing the point cloud to have max norm 1?
            # Or just let it float and rely on the ratio.
            
            # Perturbation:
            idx = np.random.randint(n)
            step_size = 0.1 * (T / start_temp + 0.1) # Scale step with temp
            noise = np.random.randn(d) * step_size
            new_points = current_points.copy()
            new_points[idx] += noise
            
            # Optional: Project to unit sphere? 
            # If we project, we constrain the search to the sphere. 
            # Is the optimal config on a sphere? 
            # For maximizing min/max distance ratio, yes, usually points lie on a sphere 
            # (Tammes problem variant). If they didn't, you could push them out to increase 
            # min distance without changing max (if max is determined by diameter) or 
            # adjust. Actually, if points are inside, you can expand them to increase min.
            # So assuming they lie on a sphere is a strong prior.
            # Let's project new_points[idx] back to the sphere defined by the current centroid?
            # Or just normalize the entire cloud?
            # Let's try: Normalize the entire cloud so that the max distance is 1?
            # No, that changes the relative positions.
            # Let's just normalize the norms of individual points to 1?
            # This forces points onto the unit sphere centered at origin.
            # This reduces the dimensionality and might help.
            # But the center might not be the origin.
            # Let's center the points first.
            
            # Center the new points
            center = np.mean(new_points, axis=0)
            new_points_centered = new_points - center
            # Normalize norms to 1?
            norms = np.linalg.norm(new_points_centered, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            new_points_centered /= norms
            
            new_score = true_score(new_points_centered)
            
            # Acceptance criterion
            delta_score = new_score - current_score
            
            if delta_score > 0:
                # Always accept improvement
                current_points = new_points_centered
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Accept with probability exp(delta_score / T)
                # Note: delta_score is negative
                prob = np.exp(delta_score / T)
                if np.random.rand() < prob:
                    current_points = new_points_centered
                    current_score = new_score

        return best_points, best_score

    best_points_overall = None
    best_score_overall = 0.0

    inits = generate_initial_points()
    
    # Run SA for each initial point
    # Budget allocation:
    # Total time ~ 600s.
    # We have ~11 inits.
    # Let's run a fixed number of iterations per init, or time-based.
    # Let's do time-based: give each init 40s.
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Normalize initial points to unit sphere centered at origin
        init_centered = init_pts - np.mean(init_pts, axis=0)
        init_norms = np.linalg.norm(init_centered, axis=1, keepdims=True)
        init_norms[init_norms < 1e-6] = 1.0
        init_centered /= init_norms
        
        # SA parameters
        max_iters = 50000 # Adjust based on speed
        start_temp = 0.5
        end_temp = 0.001
        
        # Estimate time per iteration
        # pdist for 14 points is very fast.
        # Let's just run a fixed number of iterations and check time.
        
        res_pts, res_score = simulated_annealing(init_centered, max_iters, start_temp, end_temp)
        
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_points_overall = res_pts.copy()

    # If SA didn't find anything good, fallback to a known good config
    if best_points_overall is None or best_score_overall < 0.01:
        # Fallback: Icosahedron + 2 points
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p13 = np.array([0, 0, 1.0])
        p14 = np.array([0, 0, -1.0])
        best_points_overall = np.vstack([ico_verts, p13.reshape(1,3), p14.reshape(1,3)])
        
        # Center and normalize
        best_points_overall = best_points_overall - np.mean(best_points_overall, axis=0)
        norms = np.linalg.norm(best_points_overall, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        best_points_overall /= norms

    # Final check
    if best_points_overall.shape != (n, d) or not np.all(np.isfinite(best_points_overall)):
        # Absolute fallback
        best_points_overall = np.zeros((n, d))
        best_points_overall[:, 0] = np.linspace(-1, 1, n)
        
    return best_points_overall
