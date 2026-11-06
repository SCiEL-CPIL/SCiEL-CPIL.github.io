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
            
            # Check if all points are in a closed hemisphere
            # If so, the covering radius is at least 90 degrees
            # Simple check: if the origin is not in the convex hull, points are in a hemisphere
            hull = ConvexHull(pts)
            # Check if origin is inside the convex hull
            # The origin is inside if for all facets, the origin is on the same side as the interior
            # For a facet with normal n and point p on facet, origin is inside if n·(0 - p) <= 0
            # i.e., n·p >= 0 for all facets (since normals point outward)
            origin_inside = True
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                normal = np.cross(p1 - p0, p2 - p0)
                n_norm = np.linalg.norm(normal)
                if n_norm < 1e-15:
                    continue
                normal /= n_norm
                # Ensure normal points outward (away from origin)
                # For points on unit sphere, outward normal should have positive dot with centroid
                centroid = (p0 + p1 + p2) / 3.0
                if np.dot(normal, centroid) < 0:
                    normal = -normal
                if np.dot(normal, -p0) > 1e-10:
                    # Origin is on the "outside" of this facet
                    origin_inside = False
                    break
            
            if not origin_inside:
                # Points are in a closed hemisphere, covering radius >= 90 degrees
                # But we still compute the actual value for consistency
                pass
            
            max_r = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute the circumcenter of the spherical triangle (p0, p1, p2).
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                
                try:
                    w = np.linalg.solve(G, rhs)
                except np.linalg.LinAlgError:
                    continue
                    
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm < 1e-10:
                    continue
                    
                c /= c_norm
                
                # Compute angular distance from c to p0
                cos_d = np.clip(c @ p0, -1.0, 1.0)
                d = np.arccos(cos_d)
                
                if d > max_r:
                    max_r = d
                    
            return np.degrees(max_r)
            
        except Exception:
            return 180.0

    def get_circumcenters(pts):
        """
        Compute circumcenters for all hull facets.
        Returns array of shape (n_facets, 3)
        """
        hull = ConvexHull(pts)
        centers = []
        for face in hull.simplices:
            p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
            p_face = np.array([p0, p1, p2])
            G = p_face @ p_face.T
            rhs = np.array([1.0, 1.0, 1.0])
            try:
                w = np.linalg.solve(G, rhs)
                c = np.sum(w[:, None] * p_face, axis=0)
                c_norm = np.linalg.norm(c)
                if c_norm > 1e-10:
                    centers.append(c / c_norm)
            except:
                pass
        return np.array(centers)

    def lloyd_polish(pts, max_iters=20, step_size=0.1):
        """
        Lloyd's algorithm for sphere covering.
        Moves each point towards the centroid of the circumcenters of its adjacent facets.
        """
        pts = normalize(pts)
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            try:
                hull = ConvexHull(pts)
            except:
                break
                
            # Accumulate sum of circumcenters for each vertex
            sum_centers = np.zeros_like(pts)
            counts = np.zeros(N)
            
            for face in hull.simplices:
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
                        for v_idx in face:
                            sum_centers[v_idx] += c
                            counts[v_idx] += 1
                except:
                    pass
            
            # Update points
            moved = False
            for i in range(N):
                if counts[i] > 0:
                    target = sum_centers[i] / counts[i]
                    target_norm = np.linalg.norm(target)
                    if target_norm > 1e-10:
                        target /= target_norm
                        new_pt = pts[i] + step_size * (target - pts[i])
                        new_pt_norm = np.linalg.norm(new_pt)
                        if new_pt_norm > 1e-10:
                            pts[i] = new_pt / new_pt_norm
                            moved = True
            
            if not moved:
                break
                
        return pts

    def covering_energy(x, beta=50.0):
        """
        Soft-max approximation of the covering radius.
        """
        pts = x.reshape(-1, 3)
        pts = normalize(pts)
        
        if not hasattr(covering_energy, 'test_pts'):
            # Generate Fibonacci sphere points - Increased to 8000 for better resolution
            n_test = 8000
            i = np.arange(n_test) + 0.5
            z = 1.0 - 2.0 * i / n_test
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            covering_energy.test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            
        test_pts = covering_energy.test_pts
        
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        min_dists = np.min(dists, axis=1)
        
        max_d = np.max(min_dists)
        if max_d == 0:
            return 0.0
        # Adaptive beta: increase as we get closer to optimal
        current_beta = max(beta, 20.0 / max(max_d, 1e-6))
        exp_dists = np.exp(current_beta * (min_dists - max_d))
        soft_max = max_d + (1.0 / current_beta) * np.log(np.sum(exp_dists))
        
        return soft_max

    def covering_energy_grad(x, beta=50.0):
        """
        Gradient of the soft-max covering energy.
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
            
        # Adaptive beta matching covering_energy
        current_beta = max(beta, 20.0 / max(max_d, 1e-6))
        exp_dists = np.exp(current_beta * (min_dists - max_d))
        weights = exp_dists / np.sum(exp_dists)
        
        grad_pts = np.zeros_like(pts)
        
        # Vectorized gradient computation
        for i in range(n_test):
            j = min_idx[i]
            w = weights[i]
            if w == 0:
                continue
            
            u = dots[i, j]
            denom = np.sqrt(1.0 - u * u + 1e-12)
            grad_u = -1.0 / denom
            grad_pts[j] += w * current_beta * grad_u * test_pts[i]
            
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
    try:
        res = minimize(
            covering_energy, 
            x0, 
            jac=covering_energy_grad, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
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
    
    # Second optimization pass from a different starting point if time permits
    if time.time() < deadline - 30:
        # Use a slightly perturbed Fibonacci sphere
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5  # Shifted
        r = np.sqrt(1.0 - z * z)
        init_pts2 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        x0_2 = init_pts2.flatten()
        
        try:
            res2 = minimize(
                covering_energy, 
                x0_2, 
                jac=covering_energy_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if res2.fun < np.inf:
                pts_opt2 = res2.x.reshape(-1, 3)
                pts_opt2 = normalize(pts_opt2)
                cr_opt2 = compute_covering_radius(pts_opt2)
                
                if cr_opt2 < best_cr:
                    best_cr = cr_opt2
                    best_pts = pts_opt2
                    
        except Exception:
            pass
        
    # Refinement: Lloyd's Algorithm
    if time.time() < deadline - 15:
        pts_lloyd = lloyd_polish(best_pts, max_iters=50, step_size=0.2)
        cr_lloyd = compute_covering_radius(pts_lloyd)
        if cr_lloyd < best_cr:
            best_cr = cr_lloyd
            best_pts = pts_lloyd
            
        # Second pass with smaller step
        if time.time() < deadline - 10:
            pts_lloyd2 = lloyd_polish(best_pts, max_iters=50, step_size=0.05)
            cr_lloyd2 = compute_covering_radius(pts_lloyd2)
            if cr_lloyd2 < best_cr:
                best_cr = cr_lloyd2
                best_pts = pts_lloyd2
    
    # Micro-polish: iteratively optimize the worst facet's circumcenter
    def micro_polish(pts, max_iters=200, tol=1e-10):
        """
        Iteratively move points to reduce the maximum circumradius of hull facets.
        Focuses on the worst (largest circumradius) facet each iteration.
        """
        pts = normalize(pts)
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
                
            try:
                hull = ConvexHull(pts)
            except:
                break
            
            # Find the facet with maximum circumradius
            max_r = 0.0
            max_face_idx = -1
            max_circumcenter = None
            max_face = None
            
            for idx, face in enumerate(hull.simplices):
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                p_face = np.array([p0, p1, p2])
                G = p_face @ p_face.T
                rhs = np.array([1.0, 1.0, 1.0])
                try:
                    w = np.linalg.solve(G, rhs)
                    c = np.sum(w[:, None] * p_face, axis=0)
                    c_norm = np.linalg.norm(c)
                    if c_norm < 1e-10:
                        continue
                    c /= c_norm
                    cos_d = np.clip(c @ p0, -1.0, 1.0)
                    d = np.arccos(cos_d)
                    if d > max_r:
                        max_r = d
                        max_face_idx = idx
                        max_circumcenter = c
                        max_face = face
                except:
                    pass
            
            if max_face_idx < 0:
                break
            
            # Move the three vertices of the worst facet toward the circumcenter
            # Use a small step to avoid oscillation
            step = 0.01
            for v_idx in max_face:
                target = max_circumcenter
                new_pt = pts[v_idx] + step * (target - pts[v_idx])
                new_pt_norm = np.linalg.norm(new_pt)
                if new_pt_norm > 1e-10:
                    pts[v_idx] = new_pt / new_pt_norm
            
            # Check convergence
            if max_r < np.radians(18.31):  # Close to best known
                break
                
        return pts
    
    if time.time() < deadline - 5:
        pts_micro = micro_polish(best_pts, max_iters=200)
        cr_micro = compute_covering_radius(pts_micro)
        if cr_micro < best_cr:
            best_cr = cr_micro
            best_pts = pts_micro
    
    # Final Lloyd pass with very small step
    if time.time() < deadline - 2:
        pts_final = lloyd_polish(best_pts, max_iters=30, step_size=0.01)
        cr_final = compute_covering_radius(pts_final)
        if cr_final < best_cr:
            best_cr = cr_final
            best_pts = pts_final

    return best_pts
# EVOLVE-BLOCK-END
