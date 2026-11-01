import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin

def construct():
    """
    Constructs a configuration of 50 points on the unit sphere minimizing the covering radius.
    """
    # 1. Generate a dense grid of test points on the sphere
    # Using a Fibonacci spiral for uniform distribution
    M = 2000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Precompute Y for faster dot products
    # Y shape: (M, 3)
    
    best_points = None
    best_score = np.inf
    
    # Try a few restarts to avoid local minima
    n_restarts = 8
    
    for k in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Initial configuration: Fibonacci spiral with random rotation/jitter
        i_init = np.arange(N) + 0.5
        z_init = 1.0 - 2.0 * i_init / N
        phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
        r_init = np.sqrt(1.0 - z_init * z_init)
        P0 = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
        
        # Add random jitter
        if k > 0:
            # Random rotation
            from scipy.spatial.transform import Rotation
            rot = Rotation.random(random_state=k)
            P0 = rot.apply(P0)
            # Small noise
            P0 += np.random.normal(0, 0.1, P0.shape)
            
        # Normalize
        norms = np.linalg.norm(P0, axis=1, keepdims=True)
        P0 = P0 / norms
        
        # Define objective and gradient
        def objective_and_grad(x):
            """
            x: (N, 3) Cartesian coordinates (not necessarily normalized)
            Returns: (loss, gradient)
            """
            P = x.reshape(N, 3)
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms < 1e-8, 1.0, norms)
            U = P / norms  # (N, 3) unit vectors
            
            # Compute angular distances between Y (M,3) and U (N,3)
            # D[i, j] = arccos(clip(Y[i] . U[j], -1, 1))
            # Dot products: (M, N)
            dots = Y @ U.T  # (M, N)
            dots = np.clip(dots, -0.999999, 0.999999)
            
            # Soft-min of squared distances
            # dist^2 = 2 * (1 - cos(dist)) = 2 * (1 - dots)
            sq_dists = 2.0 * (1.0 - dots)  # (M, N)
            
            # We want to minimize the sum over Y of the soft-min over j of sq_dists
            # softmin_j(v_j) = -1/beta * log(sum(exp(-beta * v_j)))
            beta = 50.0  # Temperature parameter
            
            # For numerical stability, subtract max
            # v = -beta * sq_dists
            v = -beta * sq_dists  # (M, N)
            v_max = np.max(v, axis=1, keepdims=True)
            v_shifted = v - v_max
            exp_v = np.exp(v_shifted)
            sum_exp = np.sum(exp_v, axis=1, keepdims=True)
            
            # loss = sum( -1/beta * log(sum_exp) + v_max )  (since -1/beta * log(sum(exp(v))) = -1/beta * (log(sum(exp(v_shifted))) + v_max))
            # Actually: -1/beta * log(sum(exp(v))) = -1/beta * (v_max + log(sum(exp(v_shifted))))
            loss_term = -1.0 / beta * (v_max[:, 0] + np.log(sum_exp[:, 0]))
            loss = np.sum(loss_term)
            
            # Gradient calculation
            # d(loss)/d(P)
            # Let L = sum_i l_i, where l_i = -1/beta * (v_max_i + log(sum_j exp(v_ij - v_max_i)))
            # Let w_ij = exp(v_ij - v_max_i) / sum_j exp(v_ij - v_max_i)  (softmax weights)
            # d(l_i)/d(v_ij) = -1/beta * (w_ij - delta(i==argmax) * ... ) 
            # Wait, the derivative of softmin is:
            # d/dv_j (-1/beta log sum exp(-beta v)) = -1/beta * (1/sum) * sum_k exp(-beta v_k) * (-beta) * delta_{jk}
            # = exp(-beta v_j) / sum_k exp(-beta v_k) = w_j
            # So d(l_i)/d(sq_dist_ij) = w_ij
            # Note: sq_dist_ij = 2(1 - dots_ij)
            # d(sq_dist_ij)/d(dots_ij) = -2
            # d(dots_ij)/d(U_j) = -Y_i
            # d(U_j)/d(P_j) = (I - U_j U_j^T) / |P_j|
            
            # Compute weights w (M, N)
            w = exp_v / sum_exp  # (M, N)
            
            # Gradient of loss w.r.t sq_dists: (M, N)
            grad_sq_dists = w  # d(loss)/d(sq_dist)
            
            # Gradient w.r.t dots: (M, N)
            grad_dots = -2.0 * grad_sq_dists
            
            # Gradient w.r.t U: 
            # d(loss)/d(U_j) = sum_i grad_dots[i, j] * (-Y_i)
            # = - (grad_dots.T @ Y)
            grad_U = -(grad_dots.T @ Y)  # (N, 3)
            
            # Gradient w.r.t P:
            # d(U_j)/d(P_j) = (I - U_j U_j^T) / norm_j
            # grad_P_j = (I - U_j U_j^T) @ grad_U_j / norm_j
            # = (grad_U_j - U_j (U_j . grad_U_j)) / norm_j
            
            # U_j . grad_U_j
            proj = np.sum(U * grad_U, axis=1, keepdims=True)  # (N, 1)
            grad_P = (grad_U - U * proj) / norms  # (N, 3)
            
            return loss, grad_P.flatten()

        # Run L-BFGS-B
        # We might run it in stages with increasing beta if time permits, but for now fixed beta=50 is a good start.
        # To allow for better convergence, we can run it for a limited number of iterations.
        
        try:
            res = minimize(
                objective_and_grad,
                P0.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            P_opt = res.x.reshape(N, 3)
            # Normalize final result
            norms_opt = np.linalg.norm(P_opt, axis=1, keepdims=True)
            P_opt = P_opt / norms_opt
            
            # Evaluate actual covering radius (using hull for accuracy)
            score = _compute_covering_radius(P_opt)
            
            if score < best_score:
                best_score = score
                best_points = P_opt
                
        except Exception:
            pass

    # If no valid points found, return initial
    if best_points is None:
        i_init = np.arange(N) + 0.5
        z_init = 1.0 - 2.0 * i_init / N
        phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
        r_init = np.sqrt(1.0 - z_init * z_init)
        best_points = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])

    # Final Polish: Local search on the worst facet
    if time.time() - START_TIME < TIME_LIMIT - 5:
        best_points = _local_polish(best_points, time_limit=TIME_LIMIT)
        
    return best_points

def _compute_covering_radius(points):
    """
    Computes the covering radius by finding the maximum circumradius of the convex hull facets.
    """
    try:
        hull = ConvexHull(points)
        max_rad = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            # Compute circumradius of triangle p1, p2, p3
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
            if area < 1e-10:
                continue
            R = (a * b * c) / (4.0 * area)
            if R > max_rad:
                max_rad = R
        # Convert radius to angular distance?
        # The distance from center to circumcenter is R.
        # The angular distance is arccos(1 - R^2/2)? No.
        # The circumcenter C is at distance R from the vertices.
        # The distance from origin to C is h.
        # For a triangle on unit sphere, the circumcenter of the spherical triangle is not the same as the planar circumcenter projected.
        # However, the "covering radius" is the max angular distance from any point on sphere to nearest p_i.
        # The worst point is the circumcenter of a facet of the convex hull of the points (in R3).
        # Let the facet be defined by p1, p2, p3.
        # The circumcenter C of the triangle p1,p2,p3 in R3 lies on the line passing through the origin and the normal of the plane? No.
        # The circumcenter C is the point equidistant from p1, p2, p3.
        # We want the angular distance from C (normalized to unit sphere) to the nearest p_i?
        # Actually, the point on the sphere farthest from the set {p_i} is the circumcenter of a facet of the convex hull of {p_i}.
        # Let C be the circumcenter of the triangle p1,p2,p3.
        # The distance from origin to C is not necessarily 1.
        # The point on the sphere is C / ||C||.
        # The angular distance from C/||C|| to p1 is arccos( (C/||C||) . p1 ).
        # Since C is equidistant from p1,p2,p3, this distance is the same for all three.
        # Let's compute it.
        
        max_ang = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            # Solve for C: |C-p1|^2 = |C-p2|^2 = |C-p3|^2
            # 2 C . (p2-p1) = |p2|^2 - |p1|^2 = 0
            # 2 C . (p3-p1) = 0
            # So C is perpendicular to p2-p1 and p3-p1.
            # C is parallel to n = (p2-p1) x (p3-p1)
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n_unit = n / n_norm
            # C = t * n_unit
            # |t n_unit - p1|^2 = t^2 - 2 t (n_unit . p1) + 1
            # We want to minimize/maximize?
            # The circumcenter is the unique point.
            # t = (n_unit . p1)  (since p1 is on unit sphere, and C is on the normal line through origin? 
            # Wait, the plane of the triangle does not pass through the origin.
            # The circumcenter C lies on the line passing through the origin and the normal vector?
            # No, C lies on the line perpendicular to the plane passing through the centroid? No.
            # C is the intersection of the plane's normal line passing through the triangle's centroid? No.
            # C is the point such that C . (p2-p1) = 0 and C . (p3-p1) = 0.
            # This implies C is parallel to n = (p2-p1) x (p3-p1).
            # So C = t * n_unit.
            # Condition: |C - p1|^2 = |C - p2|^2.
            # t^2 - 2t (n_unit . p1) + 1 = t^2 - 2t (n_unit . p2) + 1
            # n_unit . p1 = n_unit . p2 (This is true if n_unit is normal to the plane defined by p1,p2,p3? No, n is normal to the vectors p2-p1, p3-p1. So n . (p2-p1) = 0 => n . p2 = n . p1. Yes.)
            # So this condition is automatically satisfied for any t?
            # No, we need C to be the circumcenter, meaning distance to p1, p2, p3 is the same.
            # The locus of points equidistant from p1 and p2 is the plane perpendicular to p2-p1 passing through midpoint.
            # The intersection of these two planes is the line parallel to n.
            # So C = t * n_unit is on this line.
            # We need to find t.
            # Actually, the circumcenter is not uniquely determined by just being on the normal line?
            # Wait, the circumcenter of a triangle in 3D is the center of the unique circle passing through the 3 points.
            # The circle lies in the plane of the triangle.
            # The center C must lie in the plane of the triangle.
            # My previous deduction that C is parallel to n (normal to plane) is WRONG.
            # C lies IN the plane.
            # So C = p1 + u (p2-p1) + v (p3-p1).
            # And |C-p1| = |C-p2| = |C-p3|.
            
            # Let's use the formula for circumcenter of triangle with vertices on unit sphere.
            # It's easier to compute the circumradius R of the planar triangle p1,p2,p3.
            # The distance from the origin to the plane is h.
            # The distance from the origin to the circumcenter C is d.
            # d^2 = R^2 - h^2? No.
            # Let's just compute C explicitly.
            
            # Solve linear system for C:
            # 2 C . (p2-p1) = 0
            # 2 C . (p3-p1) = 0
            # C . n_unit = h_plane (distance from origin to plane)
            # n_unit = n / ||n||
            # h_plane = n_unit . p1
            # C = h_plane * n_unit ?
            # Let's check: C . (p2-p1) = h_plane n_unit . (p2-p1) = 0. Yes.
            # So C = h_plane * n_unit.
            # This C is the projection of the origin onto the plane? No, it's the point on the normal line closest to the origin.
            # Is this the circumcenter?
            # The circumcenter is the point in the plane equidistant from vertices.
            # The point C = h_plane * n_unit is the foot of the perpendicular from the origin to the plane.
            # Is the foot of the perpendicular the circumcenter?
            # Only if the triangle is right-angled? No.
            # The circumcenter is NOT generally the foot of the perpendicular from the origin.
            
            # Correct method:
            # C = p1 + alpha (p2-p1) + beta (p3-p1)
            # |C-p1|^2 = |C-p2|^2 => 2 alpha ||p2-p1||^2 + 2 beta (p2-p1).(p3-p1) - ||p2-p1||^2 = 0
            # |C-p1|^2 = |C-p3|^2 => 2 alpha (p2-p1).(p3-p1) + 2 beta ||p3-p1||^2 - ||p3-p1||^2 = 0
            
            v1 = p2 - p1
            v2 = p3 - p1
            A = np.array([[np.dot(v1, v1), np.dot(v1, v2)], [np.dot(v1, v2), np.dot(v2, v2)]])
            b = np.array([0.5 * np.dot(v1, v1), 0.5 * np.dot(v2, v2)])
            try:
                coeffs = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            C = p1 + coeffs[0] * v1 + coeffs[1] * v2
            
            C_norm = np.linalg.norm(C)
            if C_norm < 1e-10:
                continue
            C_unit = C / C_norm
            # Angular distance from C_unit to p1
            dot = np.dot(C_unit, p1)
            dot = np.clip(dot, -1.0, 1.0)
            ang = np.arccos(dot)
            if ang > max_ang:
                max_ang = ang
                
        return np.degrees(max_ang)
    except Exception:
        return np.inf

def _local_polish(points, time_limit=115.0):
    """
    Local polish using gradient descent on the max distance to hull vertices/circumcenters.
    """
    P = points.copy()
    step_size = 0.05
    for _ in range(50):
        if time.time() > time_limit - 2:
            break
        # Find worst point
        hull = ConvexHull(P)
        max_ang = 0.0
        worst_facet = None
        worst_C = None
        for simplex in hull.simplices:
            p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            A = np.array([[np.dot(v1, v1), np.dot(v1, v2)], [np.dot(v1, v2), np.dot(v2, v2)]])
            b = np.array([0.5 * np.dot(v1, v1), 0.5 * np.dot(v2, v2)])
            try:
                coeffs = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            C = p1 + coeffs[0] * v1 + coeffs[1] * v2
            C_norm = np.linalg.norm(C)
            if C_norm < 1e-10:
                continue
            C_unit = C / C_norm
            dot = np.clip(np.dot(C_unit, p1), -1.0, 1.0)
            ang = np.arccos(dot)
            if ang > max_ang:
                max_ang = ang
                worst_facet = simplex
                worst_C = C_unit
        
        if worst_facet is None:
            break
            
        # Move points in the worst facet to reduce the distance
        # Gradient of distance wrt p_i is -C_unit (projected to tangent space)
        # We want to move p_i towards C_unit? No, C_unit is the far point.
        # We want to reduce the angular distance.
        # Moving p_i towards C_unit reduces the distance.
        # But p_i must stay on sphere.
        # Tangent direction: C_unit - (C_unit . p_i) p_i
        for idx in worst_facet:
            p_i = P[idx]
            proj = np.dot(C_unit, p_i)
            tangent_dir = C_unit - proj * p_i
            norm_dir = np.linalg.norm(tangent_dir)
            if norm_dir > 1e-8:
                tangent_dir = tangent_dir / norm_dir
                P[idx] = P[idx] + step_size * tangent_dir
                P[idx] = P[idx] / np.linalg.norm(P[idx])
                
    return P

# Ensure construct is accessible
if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_compute_covering_radius(pts)}")
