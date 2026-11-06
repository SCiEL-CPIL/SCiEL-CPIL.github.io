"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """
        Compute the covering radius of the configuration on the unit sphere.
        The covering radius is the maximum circumradius of the spherical triangles
        (facets of the convex hull) formed by the points.
        """
        try:
            # Ensure points are on the unit sphere
            pts = normalize(pts)
            
            # Check for coincident points or degeneracy
            if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
                return 180.0
                
            # Compute Convex Hull
            hull = ConvexHull(pts)
            
            max_r = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute the circumcenter of the spherical triangle (p0, p1, p2).
                # The circumcenter c is equidistant from p0, p1, p2.
                # It lies in the plane spanned by p0, p1, p2? No, it lies on the sphere.
                # The vector c is a linear combination of p0, p1, p2: c = a*p0 + b*p1 + c*p2
                # Such that |c - p0|^2 = |c - p1|^2 = |c - p2|^2.
                # This implies c.dot(p0) = c.dot(p1) = c.dot(p2) = R_circ^2 (where R_circ is the chordal distance squared? No.)
                # Let d be the angular distance. cos(d) = c.dot(p_i).
                # So we want c.dot(p0) = c.dot(p1) = c.dot(p2).
                
                # Let M be the matrix with rows p0, p1, p2.
                # We want to solve M @ w = [1, 1, 1].T ? No.
                # c = sum w_i p_i.
                # c . p_j = sum_i w_i (p_i . p_j) = 1 for all j=0,1,2.
                # Let G be the Gram matrix of p0, p1, p2. G_ij = p_i . p_j.
                # Then G @ w = [1, 1, 1].T.
                
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                
                try:
                    w = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    # Degenerate triangle, use centroid
                    centroid = np.sum(p_face, axis=0)
                    centroid /= np.linalg.norm(centroid)
                    dists = np.arccos(np.clip(pts @ centroid, -1.0, 1.0))
                    min_dist = np.min(dists)
                    if min_dist > max_r:
                        max_r = min_dist
                    continue
                    
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    centroid = np.sum(p_face, axis=0)
                    centroid /= np.linalg.norm(centroid)
                    dists = np.arccos(np.clip(pts @ centroid, -1.0, 1.0))
                    min_dist = np.min(dists)
                    if min_dist > max_r:
                        max_r = min_dist
                    continue
                    
                c /= c_norm
                
                # The angular distance from c to any of p0, p1, p2 is the circumradius.
                # We check if c is inside the spherical triangle. 
                # For convex hull facets, the circumcenter of the face corresponds to the 
                # "worst point" in the Voronoi cell only if the circumcenter lies within the triangle.
                # If it lies outside, the worst point is on the edge. 
                # However, for the convex hull of points on a sphere, the circumcenter of a face 
                # is the center of the smallest cap containing the three vertices.
                # The covering radius is the max over all caps.
                # It is known that the covering radius is the max circumradius of the Delaunay triangles.
                
                # Compute angular distance from c to p0
                cos_d = np.clip(c @ p0, -1.0, 1.0)
                d = np.arccos(cos_d)
                
                if d > max_r:
                    max_r = d
                    
            return np.degrees(max_r)
            
        except Exception:
            return 180.0

    def covering_energy(x, beta=50.0):
        """
        Soft-max approximation of the covering radius.
        We sample a set of points on the sphere and compute the max distance to nearest config point.
        To make it differentiable, we use a soft-max (log-sum-exp) of the distances.
        Actually, a better energy for covering is the "max-min" distance.
        Let's use a fixed grid of test points.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        # Use a pre-computed set of test points (Fibonacci sphere)
        # This is expensive to compute every time, so we use a smaller set or cached one.
        # For efficiency in the loop, let's use a moderate number of test points.
        # However, N=50, so the Voronoi cells are large.
        
        # Alternative: Use the convex hull circumradii directly? 
        # The convex hull changes discretely, so it's not differentiable.
        
        # Let's stick to the soft-max of distances to a grid.
        # Generate grid once outside? No, we need it inside.
        # Let's use 2000 test points.
        
        if not hasattr(covering_energy, 'test_pts'):
            # Generate Fibonacci sphere points
            n_test = 2000
            i = np.arange(n_test) + 0.5
            z = 1.0 - 2.0 * i / n_test
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            covering_energy.test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
        test_pts = covering_energy.test_pts
        
        # Compute distances from test_pts to pts
        # dists[i, j] = arccos(test_pts[i] . pts[j])
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Min distance for each test point
        min_dists = np.min(dists, axis=1)
        
        # Soft max: (1/beta) * log(sum(exp(beta * min_dists)))
        # To avoid overflow, subtract max
        max_d = np.max(min_dists)
        if max_d == 0:
            return 0.0
        exp_dists = np.exp(beta * (min_dists - max_d))
        soft_max = max_d + (1.0 / beta) * np.log(np.sum(exp_dists))
        
        return soft_max

    def covering_energy_grad(x, beta=50.0):
        """
        Vectorized gradient of the soft-max covering energy.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        test_pts = covering_energy.test_pts
        n_test = len(test_pts)
        
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_idx = np.argmin(dists, axis=1)
        min_dists = np.take_along_axis(dists, min_idx[:, None], axis=1).flatten()
        
        max_d = np.max(min_dists)
        if max_d == 0:
            return np.zeros_like(x)
            
        exp_dists = np.exp(beta * (min_dists - max_d))
        weights = exp_dists / np.sum(exp_dists)
        
        # Vectorized gradient computation
        # grad_pts[j] = sum_{i: min_idx[i]==j} weights[i] * beta * (-1/sqrt(1-dots[i,j]^2)) * test_pts[i]
        
        # Create a mask for each point j
        # We can use np.add.at or bincount-like logic
        # Let's compute the factor for each test point i
        # factor_i = weights[i] * beta / sqrt(1 - dots[i, min_idx[i]]^2)
        
        u_min = min_dists # This is dists, not dots. We need dots[i, min_idx[i]]
        # dots[i, min_idx[i]] is the cosine of the distance
        cos_min = np.take_along_axis(dots, min_idx[:, None], axis=1).flatten()
        
        denom = np.sqrt(1.0 - cos_min * cos_min + 1e-12)
        factors = weights * beta / denom
        
        # grad_pts[j] = sum_{i: min_idx[i]==j} factors[i] * test_pts[i]
        # Note: The derivative of arccos(u) is -1/sqrt(1-u^2).
        # So the sign is negative.
        # grad_pts[j] = - sum ...
        
        grad_pts = np.zeros_like(pts)
        np.add.at(grad_pts, min_idx, -factors[:, None] * test_pts)
        
        # Chain rule for normalization
        v = x.reshape(-1, 3)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        
        grad_v = np.zeros_like(v)
        for k in range(N):
            n_k = norms[k, 0]
            v_k = v[k]
            dot_vg = np.dot(v_k, grad_pts[k])
            grad_v[k] = grad_pts[k] / n_k - v_k * (dot_vg / (n_k ** 3))
            
        return grad_v.flatten()

    # Initial configuration: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = init_pts.flatten()
    
    best_cr = compute_covering_radius(init_pts)
    best_pts = init_pts
    
    # Optimization loop
    # We use L-BFGS-B with the gradient.
    # Since the landscape is complex, we do a few restarts or just one long run?
    # Given the time limit, let's try one run with a good number of iterations.
    
    try:
        res = minimize(
            covering_energy, 
            x0, 
            jac=covering_energy_grad, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
        )
        
        if res.fun < np.inf:
            pts_opt = res.x.reshape(-1, 3)
            pts_opt = normalize(pts_opt)
            cr_opt = compute_covering_radius(pts_opt)
            
            if cr_opt < best_cr:
                best_cr = cr_opt
                best_pts = pts_opt
                
    except Exception:
        pass
        
    # Refinement: Local search on the hull facets
    # If we have time, we can try to move points to reduce the max circumradius.
    # This is a heuristic.
    
    def lloyd_step(pts):
        """
        Perform one Lloyd's iteration: move each point to the centroid of its Voronoi cell.
        Approximation: Centroid of the point and the circumcenters of adjacent hull faces.
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return pts
            
        new_pts = pts.copy()
        
        # Map point index to list of face indices
        point_faces = [[] for _ in range(N)]
        for i, face in enumerate(hull.simplices):
            for idx in face:
                point_faces[idx].append(i)
                
        for i in range(N):
            if not point_faces[i]:
                continue
                
            # Sum of the point itself and the circumcenters of adjacent faces
            vec_sum = pts[i].copy()
            
            for face_idx in point_faces[i]:
                face = hull.simplices[face_idx]
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                try:
                    w = np.linalg.solve(G, rhs)
                    c = np.sum(w[:, None] * p_face, axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm > 1e-10:
                        c /= c_norm
                        vec_sum += c
                except np.linalg.LinAlgError:
                    continue
                    
            new_norm = np.linalg.norm(vec_sum)
            if new_norm > 1e-10:
                new_pts[i] = vec_sum / new_norm
                
        return new_pts

    # Phase 2: Lloyd's Iteration
    if time.time() < deadline - 15:
        current_pts = best_pts.copy()
        current_cr = best_cr
        
        for iter in range(50): # Max 50 iterations
            if time.time() > deadline - 5:
                break
                
            current_pts = lloyd_step(current_pts)
            current_pts = normalize(current_pts)
            
            # Check for validity (coincident points, etc.)
            # Simple check: if any two points are too close
            # This is O(N^2) but N=50 is small
            min_dist = 1.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(current_pts[i] - current_pts[j])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist < 1e-6:
                # Points collapsed, revert or break
                break
                
            cr = compute_covering_radius(current_pts)
            if cr < current_cr:
                current_cr = cr
                if cr < best_cr:
                    best_cr = cr
                    best_pts = current_pts.copy()
            else:
                # If it gets worse, we might be in a bad local minimum or oscillating
                # We can break or continue with smaller steps. 
                # For simplicity, break if no improvement for a few steps?
                # Or just break if it increases significantly.
                if cr > current_cr + 1e-4:
                    break

    # Phase 3: Final Polish with Gradient Descent on the best found
    if time.time() < deadline - 5:
        x0_final = best_pts.flatten()
        try:
            res_final = minimize(
                covering_energy, 
                x0_final, 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res_final.fun < np.inf:
                pts_final = res_final.x.reshape(-1, 3)
                pts_final = normalize(pts_final)
                cr_final = compute_covering_radius(pts_final)
                
                if cr_final < best_cr:
                    best_cr = cr_final
                    best_pts = pts_final
                    
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
