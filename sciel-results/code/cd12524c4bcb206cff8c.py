import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = np.inf

    def compute_covering_radius(pts):
        """
        Computes the exact covering radius by checking all Delaunay triangles.
        Returns (radius, hull, worst_face_indices)
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_face = None
            
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                # Compute circumcenter of the spherical triangle
                # The circumcenter is the normalized cross product of the vertices
                # if the triangle is acute (circumcenter inside).
                # For obtuse triangles, the "worst point" is the midpoint of the longest edge.
                
                # Vectors
                # Normal to the plane
                n = np.cross(v2 - v1, v3 - v1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                
                # Distance from circumcenter (n) to vertices
                d1 = np.arccos(np.clip(np.dot(n, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(n, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(n, v3), -1.0, 1.0))
                
                # Check if circumcenter is inside the triangle
                # It is inside if all angles are < 90 degrees (acute triangle)
                # Equivalently, the circumcenter n must have positive dot product with the 
                # outward normals of the edges? 
                # Simpler check: If max(d1,d2,d3) > pi/2, the circumcenter is outside the "cap" defined by the triangle 
                # in the sense that the farthest point is on the boundary (edge).
                
                if max(d1, d2, d3) <= np.pi / 2.0:
                    # Circumcenter is the worst point
                    local_rad = max(d1, d2, d3)
                else:
                    # Worst point is midpoint of the longest edge
                    # Edge lengths (angular)
                    d12 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                    d23 = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
                    d31 = np.arccos(np.clip(np.dot(v3, v1), -1.0, 1.0))
                    
                    # Longest edge determines the worst midpoint
                    # The covering radius for this face is half the length of the longest edge
                    local_rad = max(d12, d23, d31) / 2.0
                
                if local_rad > max_rad:
                    max_rad = local_rad
                    worst_face = simplex
                    
            return max_rad, hull, worst_face
        except Exception:
            return np.inf, None, None

    def relax_face(pts, face_idx, target_rad, hull):
        """
        Attempts to move the vertices of a specific face to reduce its circumradius
        towards target_rad, while not increasing the radii of adjacent faces too much.
        This is a local heuristic.
        """
        v1_idx, v2_idx, v3_idx = face_idx
        v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]
        
        # Compute current circumcenter
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            return pts
        
        n = n / n_norm
        d1 = np.arccos(np.clip(np.dot(n, v1), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(n, v2), -1.0, 1.0))
        d3 = np.arccos(np.clip(np.dot(n, v3), -1.0, 1.0))
        
        # If the face is obtuse, the "center" to move towards is the midpoint of the longest edge
        if max(d1, d2, d3) > np.pi / 2.0:
             # Find longest edge
             d12 = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
             d23 = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
             d31 = np.arccos(np.clip(np.dot(v3, v1), -1.0, 1.0))
             
             if d12 >= d23 and d12 >= d31:
                 center = (v1 + v2) / np.linalg.norm(v1 + v2)
             elif d23 >= d12 and d23 >= d31:
                 center = (v2 + v3) / np.linalg.norm(v2 + v3)
             else:
                 center = (v3 + v1) / np.linalg.norm(v3 + v1)
        else:
             center = n

        # Move vertices slightly towards the center to shrink the face
        # This reduces the circumradius of this face.
        # We use a small step size to avoid breaking the hull structure.
        step = 0.05
        
        new_v1 = v1 + step * (center - v1)
        new_v2 = v2 + step * (center - v2)
        new_v3 = v3 + step * (center - v3)
        
        # Normalize
        new_v1 /= np.linalg.norm(new_v1)
        new_v2 /= np.linalg.norm(new_v2)
        new_v3 /= np.linalg.norm(new_v3)
        
        pts[v1_idx] = new_v1
        pts[v2_idx] = new_v2
        pts[v3_idx] = new_v3
        
        return pts

    # Main Optimization Loop
    # We will use a hybrid approach:
    # 1. Identify the worst face (largest circumradius).
    # 2. Attempt to "relax" it by moving its vertices.
    # 3. If the global radius decreases, accept.
    # 4. If stuck, perform a random perturbation or a "flip" (though scipy ConvexHull doesn't easily allow flipping, 
    #    we can simulate a flip by moving points across the edge).
    
    # To make this robust, we will run this loop for a fixed number of iterations or until time runs out.
    # We also keep a history of best points.
    
    pts = best_points.copy()
    current_rad, _, _ = compute_covering_radius(pts)
    best_rad = current_rad
    
    # Pre-compute adjacency? ConvexHull.simplices gives faces. 
    # We can find neighbors by shared edges.
    
    def get_neighbors(hull, pts):
        # Returns dict: face_index -> list of neighbor face indices
        # This is expensive to compute every time, so we might skip explicit neighbor checks
        # and rely on the global radius check.
        pass

    # Strategy: 
    # 1. Start with Fibonacci.
    # 2. Iteratively:
    #    a. Compute hull and worst face.
    #    b. Relax worst face.
    #    c. Recompute hull and radius.
    #    d. If improved, keep. Else, revert and try a different move (e.g., random kick on worst face vertices).
    
    max_iterations = 5000
    for i in range(max_iterations):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        rad, hull, worst_face = compute_covering_radius(pts)
        
        if rad < best_rad:
            best_rad = rad
            best_points = pts.copy()
            
        if worst_face is None:
            break
            
        # Relax the worst face
        pts_relaxed = relax_face(pts.copy(), worst_face, rad, hull)
        
        rad_relaxed, _, _ = compute_covering_radius(pts_relaxed)
        
        if rad_relaxed < rad:
            pts = pts_relaxed
        else:
            # Try a more aggressive move: push vertices towards the center more strongly
            # Or try moving one vertex specifically.
            # Let's try a random small perturbation on the worst face vertices to escape local minima
            v1_idx, v2_idx, v3_idx = worst_face
            
            # Perturb
            noise_scale = 0.01
            pts_perturbed = pts.copy()
            pts_perturbed[v1_idx] += np.random.normal(0, noise_scale, 3)
            pts_perturbed[v2_idx] += np.random.normal(0, noise_scale, 3)
            pts_perturbed[v3_idx] += np.random.normal(0, noise_scale, 3)
            
            # Normalize
            pts_perturbed[v1_idx] /= np.linalg.norm(pts_perturbed[v1_idx])
            pts_perturbed[v2_idx] /= np.linalg.norm(pts_perturbed[v2_idx])
            pts_perturbed[v3_idx] /= np.linalg.norm(pts_perturbed[v3_idx])
            
            rad_perturbed, _, _ = compute_covering_radius(pts_perturbed)
            
            if rad_perturbed < rad:
                pts = pts_perturbed
            else:
                # If neither works, we are likely in a local minimum.
                # Perform a larger random restart on a subset of points or full restart?
                # Full restart is risky. Let's try a "simulated annealing" style acceptance.
                # But for simplicity and determinism, let's just break and try a new init if we stall.
                # However, the task asks for a single construct() call.
                # Let's try to move the vertices of the worst face towards the center of the 
                # *adjacent* faces to balance the load? That's complex.
                
                # Alternative: Move the vertex of the worst face that is farthest from the center
                # even more aggressively.
                v1, v2, v3 = pts[v1_idx], pts[v2_idx], pts[v3_idx]
                n = np.cross(v2 - v1, v3 - v1)
                n = n / np.linalg.norm(n)
                
                d1 = np.arccos(np.clip(np.dot(n, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(n, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(n, v3), -1.0, 1.0))
                
                # Move the vertex with the largest d towards the center
                if d1 >= d2 and d1 >= d3:
                    idx = v1_idx
                    v = v1
                elif d2 >= d1 and d2 >= d3:
                    idx = v2_idx
                    v = v2
                else:
                    idx = v3_idx
                    v = v3
                    
                # Move towards n
                step = 0.1
                new_v = v + step * (n - v)
                new_v /= np.linalg.norm(new_v)
                
                pts_single = pts.copy()
                pts_single[idx] = new_v
                rad_single, _, _ = compute_covering_radius(pts_single)
                
                if rad_single < rad:
                    pts = pts_single
                else:
                    # Stuck. Break to save time? Or try a global jitter.
                    # Let's try a global jitter of 1% of all points.
                    if i % 10 == 0:
                         pts_jitter = pts + np.random.normal(0, 0.005, pts.shape)
                         pts_jitter /= np.linalg.norm(pts_jitter, axis=1, keepdims=True)
                         rad_jitter, _, _ = compute_covering_radius(pts_jitter)
                         if rad_jitter < rad:
                             pts = pts_jitter
                         else:
                             # If we are stuck for a long time, we might want to restart from a better init.
                             # But we only have one init. 
                             # Let's just break to avoid wasting time on a dead end.
                             # Actually, let's continue but with smaller steps.
                             pass
                             
    # Final Polish: Use the best_points found
    # One more pass to ensure validity
    final_rad, _, _ = compute_covering_radius(best_points)
    if final_rad < np.inf:
         return best_points
    else:
         return points # Fallback to original

# EVOLVE-BLOCK-END
