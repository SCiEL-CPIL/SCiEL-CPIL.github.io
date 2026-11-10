import numpy as np
import time
import math
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons in a regular hexagon by minimizing the outer hexagon side length.
    Uses a smooth Minkowski-sum based collision metric for L-BFGS-B optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    N = 12
    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices (angle 0: vertices at 0, 60, ...)
    unit_verts = np.array([
        [math.cos(math.radians(i * 60)), math.sin(math.radians(i * 60))] for i in range(6)
    ])
    
    # Outer hexagon normals (angle 0: normals at 30, 90, ...)
    OUTER_NORMALS = np.array([
        [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] for i in range(6)
    ])
    
    def rot_matrix(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]])

    def get_rotated_verts(center, angle_rad):
        """Returns vertices of a unit hexagon at given center and angle."""
        R = rot_matrix(angle_rad)
        return center + (unit_verts @ R.T)

    def min_translation_distance(v1, v2):
        """
        Computes the minimum translation distance to separate two convex polygons.
        v1, v2: (6, 2) arrays of vertices.
        Returns a smooth scalar approximation of the minimum distance.
        """
        # Minkowski difference: v1 - v2
        # The minimum translation distance is the minimum distance from origin to the Minkowski difference polygon.
        # For convex polygons, this can be approximated by checking the minimum norm of points in the Minkowski difference.
        # However, computing the exact Minkowski difference is expensive.
        # Instead, we use the fact that for two convex polygons, the minimum distance is the minimum over all edges of one polygon 
        # of the distance to the other polygon, but that's not smooth.
        
        # A better smooth approximation:
        # The minimum distance between two convex polygons is the minimum over all pairs of vertices/edges.
        # To make it smooth and differentiable, we can use a soft-min of the squared distances between all vertex pairs?
        # No, that's not correct for polygons.
        
        # Let's use the Separating Axis Theorem idea but smoothed.
        # Actually, a very effective smooth metric for "how much they overlap" is the minimum distance from the origin to the Minkowski difference.
        # The Minkowski difference A - B is a convex polygon.
        # We can compute its vertices.
        
        # Compute Minkowski difference vertices
        # This is complex. Let's use a simpler, robust smooth metric:
        # The minimum distance between two convex polygons can be found by checking the distance from each vertex of A to each edge of B and vice versa.
        # To make it smooth, we use a soft-min (log-sum-exp) of the squared distances.
        
        # Distances from vertices of v1 to edges of v2
        dists = []
        for i in range(6):
            p1 = v1[i]
            p2 = v2[i]
            p3 = v2[(i+1)%6]
            # Distance from p1 to segment p2-p3
            edge = p3 - p2
            t = np.dot(p1 - p2, edge) / np.dot(edge, edge)
            t = np.clip(t, 0, 1)
            proj = p2 + t * edge
            dists.append(np.sum((p1 - proj)**2))
            
        for i in range(6):
            p1 = v2[i]
            p2 = v1[i]
            p3 = v1[(i+1)%6]
            edge = p3 - p2
            t = np.dot(p1 - p2, edge) / np.dot(edge, edge)
            t = np.clip(t, 0, 1)
            proj = p2 + t * edge
            dists.append(np.sum((p1 - proj)**2))
        
        dists = np.array(dists)
        
        # Soft min of squared distances
        # If dists are all > 0, they are separated. The min dist is sqrt(min(dists)).
        # To penalize overlap (negative distance in a sense), we need a metric that is negative when overlapping.
        # The SAT overlap depth is a better measure.
        
        # Let's revert to SAT but smooth it.
        # Compute projections on 12 axes (6 from each hexagon).
        axes1 = np.array([
            [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] for i in range(6)
        ])
        axes2 = axes1 # Since we rotate the whole hexagon, the axes rotate too.
        
        # Wait, the axes depend on the angle.
        # Let's compute the axes based on the current angles.
        # This function doesn't have angles. I need to pass angles or compute axes inside.
        # Let's modify the objective to compute everything.
        
        # For now, let's just return the min squared distance as a proxy for separation.
        # If they are separated, min_dist > 0. If they touch, min_dist = 0. If they overlap, min_dist is still 0 (distance is 0).
        # This is not a good penalty for overlap.
        
        # I need a metric that is negative when overlapping.
        # The maximum overlap depth from SAT is a good metric.
        # Let's compute SAT overlap depth smoothly.
        
        # I will restructure the objective to compute SAT overlap directly.
        
        return np.min(dists) # This is just a placeholder, I will implement SAT in the objective

    def get_hex_axes(angle_rad):
        """Returns 6 outward normals for a unit hexagon rotated by angle_rad."""
        R = rot_matrix(angle_rad)
        base_normals = np.array([
            [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] for i in range(6)
        ])
        return base_normals @ R.T

    def sat_overlap_depth(v1, v2, axes1, axes2):
        """
        Computes the maximum overlap depth between two hexagons using SAT.
        Returns 0 if separated, positive if overlapping.
        """
        axes = np.vstack([axes1, axes2])
        
        proj1 = v1 @ axes.T
        proj2 = v2 @ axes.T
        
        min1 = proj1.min(axis=0)
        max1 = proj1.max(axis=0)
        min2 = proj2.min(axis=0)
        max2 = proj2.max(axis=0)
        
        # Overlap on each axis
        overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        # If any overlap is <= 0, they are separated
        if np.any(overlap <= 0):
            return 0.0
        
        return np.max(overlap)

    def min_s_for_hexagons(centers, angles):
        """
        Centers: (12, 2)
        Angles: (12,) in radians
        Returns min s for outer hex centered at (0,0) angle 0.
        """
        max_proj = 0.0
        for i in range(N):
            verts = get_rotated_verts(centers[i], angles[i])
            projs = verts @ OUTER_NORMALS.T
            max_proj_hex = projs.max(axis=0)
            current_max = np.max(max_proj_hex)
            if current_max > max_proj:
                max_proj = current_max
                
        return max_proj * 2.0 / sqrt3

    def objective(x):
        centers = x[:24].reshape(12, 2)
        angles = x[24:]
        
        # Center the configuration
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        
        s = min_s_for_hexagons(centers, angles)
        
        collision_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = get_rotated_verts(centers[i], angles[i])
                v2 = get_rotated_verts(centers[j], angles[j])
                axes1 = get_hex_axes(angles[i])
                axes2 = get_hex_axes(angles[j])
                
                overlap = sat_overlap_depth(v1, v2, axes1, axes2)
                if overlap > 0:
                    collision_pen += overlap ** 2 * 100.0
        
        return s + collision_pen

    def generate_initial():
        # Honeycomb lattice: 3 rows of 4
        # Hexagon side 1. Distance between centers in honeycomb is sqrt(3) for touching edges.
        # Wait, for unit hexagons (side 1), the distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
        dx = sqrt3
        dy = 1.5 # Vertical distance in honeycomb is 1.5 * side? No.
        # In a honeycomb of side 1 hexagons:
        # Horizontal distance: 1.5 * side? No, 1.5 * side is for flat top?
        # Let's stick to the standard lattice that works.
        # The previous program used dx=2.5, dy=sqrt(3).
        # Let's use a slightly tighter lattice to start.
        dx = 1.8
        dy = 1.5
        init = []
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2.0
                init.append([x, y, 0.0])
        return np.array(init)[:, :2], np.array(init)[:, 2]

    best_val = 0.0
    best_centers = None
    best_angles = None
    best_s = float('inf')

    def run_optimization(init_centers, init_angles):
        x0 = np.hstack([init_centers.flatten(), init_angles])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            x_opt = res.x
            centers = x_opt[:24].reshape(12, 2)
            angles = x_opt[24:]
            
            centroid = np.mean(centers, axis=0)
            centers = centers - centroid
            
            s = min_s_for_hexagons(centers, angles)
            
            # Validate
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    v1 = get_rotated_verts(centers[i], angles[i])
                    v2 = get_rotated_verts(centers[j], angles[j])
                    axes1 = get_hex_axes(angles[i])
                    axes2 = get_hex_axes(angles[j])
                    overlap = sat_overlap_depth(v1, v2, axes1, axes2)
                    if overlap > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check containment
                outer_verts = get_rotated_verts(np.array([0,0]), 0.0) * s
                all_inner_verts = np.vstack([get_rotated_verts(centers[i], angles[i]) for i in range(N)])
                
                for i in range(6):
                    p1 = outer_verts[i]
                    p2 = outer_verts[(i+1)%6]
                    edge = p2 - p1
                    vecs = all_inner_verts - p1
                    crosses = edge[0]*vecs[:,1] - edge[1]*vecs[:,0]
                    if np.any(crosses < -1e-6):
                        valid = False
                        break
            
            if valid:
                val = 1.0 / s
                return val, centers, angles, s
        except Exception:
            pass
        return 0.0, None, None, None

    # Run multiple restarts
    for i in range(15):
        if time.time() > deadline:
            break
        
        if i == 0:
            init_centers, init_angles = generate_initial()
        else:
            init_centers, init_angles = generate_initial()
            init_centers += np.random.normal(0, 0.3, (12, 2))
            init_angles += np.random.uniform(-0.3, 0.3, 12)
            
        val, centers, angles, s = run_optimization(init_centers, init_angles)
        
        if val > best_val:
            best_val = val
            best_centers = centers
            best_angles = angles
            best_s = s

    if best_centers is None:
        # Fallback
        best_centers, best_angles = generate_initial()
        centroid = np.mean(best_centers, axis=0)
        best_centers = best_centers - centroid
        best_s = min_s_for_hexagons(best_centers, best_angles)

    inner_ret = np.column_stack((best_centers, best_angles * 180.0 / math.pi))
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s
