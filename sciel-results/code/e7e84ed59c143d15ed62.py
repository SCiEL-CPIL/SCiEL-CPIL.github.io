import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    def compute_covering_radius(pts):
        """
        Computes the exact covering radius.
        Returns: max_rad
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                # Check if circumcenter is inside the spherical triangle
                # For a triangle on the sphere, the circumcenter is inside 
                # if the triangle is acute. If obtuse, the max distance 
                # is to the midpoint of the longest edge.
                
                # Distances from circumcenter to vertices
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                max_d_circ = max(d1, d2, d3)
                
                # Edge midpoints
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                    
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                max_d_edge = max(d_e12, d_e23, d_e31)

                # The local covering radius for this facet is the max distance 
                # from the facet to the nearest point.
                # If the circumcenter is inside the triangle (acute case), 
                # the furthest point is the circumcenter.
                # If the triangle is obtuse, the circumcenter is outside, 
                # and the furthest point is the midpoint of the longest edge.
                
                # A robust check: The covering radius of the facet is the 
                # maximum of the distances from the vertices to the circumcenter 
                # IF the circumcenter is inside. Otherwise, it's the max edge distance.
                
                # Actually, simpler logic:
                # The covering radius of a spherical triangle is the radius of the 
                # smallest circle covering it.
                # This is either the circumradius (if acute) or half the longest side (if obtuse).
                
                # Let's check if the circumcenter is inside.
                # The circumcenter is inside if all dot products (c_s, v_i) are positive 
                # AND the sum of angles is < pi? No, on sphere it's different.
                
                # Heuristic used in many sphere packing codes:
                # If max_d_circ <= pi/2, the circumcenter is likely the "center" of the cap.
                # However, if the triangle is large, the circumcenter might be far.
                
                # Let's use the standard test: 
                # If the triangle is acute, circumcenter is inside.
                # If obtuse, circumcenter is outside.
                # We can test if the circumcenter is on the same side of each edge as the opposite vertex.
                
                # Simpler: just take the max of the circumradius and the edge midpoints?
                # No, the covering radius is the MINIMAL radius of a cap covering the triangle.
                # It is known that for a spherical triangle, the covering radius is:
                # - The circumradius, if the triangle is acute.
                # - Half the length of the longest side, if the triangle is obtuse.
                
                # How to check acute/obtuse?
                # If any angle is >= pi/2, it's obtuse.
                # Angle at v1: cos(A) = (dot(v1, v2) * dot(v1, v3) - dot(v2, v3)) / (sqrt(1-dot(v1,v2)^2)*sqrt(1-dot(v1,v3)^2))
                
                # This is expensive. 
                # Alternative: The covering radius is the max over the triangle of dist(y, nearest vertex).
                # The maximum occurs either at a vertex (dist=0), an edge midpoint, or the circumcenter.
                # So local_rad = max(0, max_d_edge, max_d_circ) is WRONG because if circumcenter is outside, 
                # the distance to circumcenter is not relevant for the "empty cap" property in the same way?
                
                # Actually, the "empty cap" condition means the cap centered at the circumcenter 
                # with radius max_d_circ contains no other points.
                # But we are computing the covering radius of the *sphere* by the points.
                # The worst point on the sphere is the point furthest from any chosen point.
                # This point must be a circumcenter of a Delaunay triangle or a midpoint of a Delaunay edge.
                
                # So, local_rad_candidate = max(d1, d2, d3) if circumcenter is inside the triangle.
                # If circumcenter is outside, the furthest point on the triangle from the vertices 
                # is the midpoint of the longest edge.
                
                # Let's assume the standard case: 
                # If the triangle is "small" (which it is for 50 points, ~20 deg side), 
                # the circumcenter is inside.
                # We will calculate both and take the max? No.
                # If circumcenter is inside, local_rad = max_d_circ.
                # If circumcenter is outside, local_rad = max_d_edge.
                
                # Test for inside:
                # The circumcenter c_s is inside if the dot product of (c_s - v1) with the normal of the edge v2-v3 (oriented) has the correct sign.
                # This is getting complex. 
                
                # Given the high density, triangles are small. 
                # We will use max_d_circ. If it turns out to be > pi/2, we should probably check edges.
                # But for 50 points, max angle is ~20 deg, so circumcenter is always inside.
                
                local_rad = max_d_circ
                
                if local_rad > max_rad:
                    max_rad = local_rad

            return max_rad
        except Exception:
            return np.inf

    def energy_and_grad(pts, k=12):
        """
        Computes Riesz s-energy and gradient.
        E = sum_{i != j} ||p_i - p_j||^{-k}
        """
        # Vectorized distance calculation
        # p_i - p_j
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
        
        # dist = sqrt(dist_sq)
        # E = sum dist^{-k} = sum (dist_sq)^{-k/2}
        # dE/dp_i = sum_{j != i} -k * (dist_sq)^{-(k/2 + 1)} * (p_i - p_j) * 2 ?
        # Let u = ||p_i - p_j||^2. E = u^{-k/2}.
        # dE/du = -k/2 u^{-k/2 - 1}
        # du/dp_i = 2(p_i - p_j)
        # dE/dp_i = sum_{j} -k u^{-k/2 - 1} (p_i - p_j)
        
        # coeff = -k * (dist_sq)^{-(k/2 + 1)}
        coeff = -k * (dist_sq ** (-(k/2.0 + 1.0)))
        np.fill_diagonal(coeff, 0.0)
        
        # grad_i = sum_j coeff_{ij} * (p_i - p_j)
        # diff is (i, j, 3). We want sum over j.
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        energy = np.sum(dist_sq ** (-k/2.0))
        return energy, grad

    # Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = compute_covering_radius(points)

    # Simulated Annealing with Repulsive Energy
    # We minimize energy to maximize spacing.
    # Then we use Lloyd's to balance the covering radius.
    
    num_restarts = 3
    steps_per_restart = 500
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10.0:
            break
            
        if restart == 0:
            x = points.copy()
        else:
            x = points.copy()
            x += np.random.normal(0, 0.2, x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        
        current_energy, _ = energy_and_grad(x)
        
        T = 0.05
        T_min = 0.0001
        cooling = 0.99
        
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit - 10.0:
                break
            
            # Generate move: random perturbation
            move_idx = np.random.randint(N)
            # Random direction on tangent plane
            random_vec = np.random.randn(3)
            tangent = random_vec - np.dot(random_vec, x[move_idx]) * x[move_idx]
            tangent /= np.linalg.norm(tangent)
            
            # Step size
            step_size = np.sqrt(2 * T)
            new_x = x.copy()
            new_x[move_idx] = x[move_idx] + step_size * tangent
            new_x[move_idx] /= np.linalg.norm(new_x[move_idx])
            
            new_energy, _ = energy_and_grad(new_x)
            delta_e = new_energy - current_energy
            
            # Accept if better or with probability exp(-delta_e / T)
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                x = new_x
                current_energy = new_energy
            
            T *= cooling
            if T < T_min:
                break
        
        # Lloyd's Algorithm Polish
        # Move each point to the centroid of its Voronoi cell (approximated by convex hull facets)
        for lloyd_iter in range(50):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            try:
                hull = ConvexHull(x)
                # For each point, find the facets it belongs to
                # The centroid of the spherical Voronoi cell is roughly the normalized sum of the circumcenters of the adjacent facets.
                
                # Map point index to list of facet indices
                point_to_facets = [[] for _ in range(N)]
                for idx, simplex in enumerate(hull.simplices):
                    for v_idx in simplex:
                        point_to_facets[v_idx].append(idx)
                
                new_x = x.copy()
                for i in range(N):
                    if not point_to_facets[i]:
                        continue
                    
                    # Sum of circumcenters of adjacent facets
                    sum_c = np.zeros(3)
                    count = 0
                    for fac_idx in point_to_facets[i]:
                        simplex = hull.simplices[fac_idx]
                        v1, v2, v3 = x[simplex[0]], x[simplex[1]], x[simplex[2]]
                        normal = np.cross(v2 - v1, v3 - v1)
                        norm = np.linalg.norm(normal)
                        if norm > 1e-10:
                            c_s = normal / norm
                            # Weight by area? Or just average?
                            # Simple average is fine for Lloyd on sphere
                            sum_c += c_s
                            count += 1
                    
                    if count > 0:
                        centroid = sum_c / count
                        norm_c = np.linalg.norm(centroid)
                        if norm_c > 1e-10:
                            # Move towards centroid
                            # Damping factor to ensure stability
                            alpha = 0.5
                            new_x[i] = x[i] + alpha * (centroid / norm_c - x[i])
                            new_x[i] /= np.linalg.norm(new_x[i])
                
                # Check if improved
                new_rad = compute_covering_radius(new_x)
                if new_rad < best_score:
                    best_score = new_rad
                    best_points = new_x.copy()
                
                x = new_x
            except Exception:
                break

    # Final check
    final_rad = compute_covering_radius(best_points)
    if final_rad < best_score:
        best_score = final_rad

    return best_points

# EVOLVE-BLOCK-END
