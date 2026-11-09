import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _compute_min_triangle_area(points):
    """
    Compute the minimum triangle area among all C(n,3) triangles.
    """
    n = len(points)
    min_area = float('inf')
    # Pre-compute indices to avoid list generation overhead
    # C(13,3) = 286
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                area = 0.5 * abs(pi[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    min_area = _compute_min_triangle_area(points)
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    return min_area / hull_area

def _cholesky_with_jitter(G, jitter=1e-9):
    """
    Compute Cholesky factorization of a symmetric matrix G.
    If it fails due to non-PSD, add jitter to the diagonal.
    Returns (L, success) where G = L @ L.T
    """
    try:
        L = np.linalg.cholesky(G)
        return L, True
    except np.linalg.LinAlgError:
        # Add jitter
        n = G.shape[0]
        G_jittered = G + np.eye(n) * jitter
        # If still fails, project to nearest PSD via eigen decomposition
        try:
            L = np.linalg.cholesky(G_jittered)
            return L, True
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(G)
            w = np.maximum(w, 0)
            G_psd = V @ np.diag(w) @ V.T
            L = np.linalg.cholesky(G_psd + np.eye(n) * 1e-12)
            return L, False

def _sdpsolve_heilbronn(n_points, time_budget=500):
    """
    Solve the Heilbronn problem using an SDP-like approach.
    
    We parameterize the points P (n x 2) via their Gram matrix G = P P^T (n x n).
    Constraints:
    1. G is PSD.
    2. Rank(G) <= 2. (Enforced by projecting to rank 2 via eigen decomposition)
    3. Points lie on the unit circle: diag(G) = 1. (Fixes hull area to pi)
    4. Maximize delta such that for all (i,j,k): Area(i,j,k) >= delta.
    
    Area^2(i,j,k) = det(G_sub) where G_sub is the 3x3 submatrix of G for indices i,j,k.
    Actually, Area^2 = 0.25 * (2*G_ii G_jj + 2*G_jj G_kk + 2*G_kk G_ii - G_ii^2 - G_jj^2 - G_kk^2 - 4*G_ij^2 - 4*G_jk^2 - 4*G_ki^2)
    Wait, the formula for squared area of triangle with vertices vectors u, v, w is:
    4 * Area^2 = 2(u.u v.v + v.v w.w + w.w u.u) - (u.u)^2 - (v.v)^2 - (w.w)^2
    Since u.u = 1, v.v = 1, w.w = 1:
    4 * Area^2 = 2(1 + 1 + 1) - 1 - 1 - 1 = 3.
    Wait, this assumes the origin is the centroid? No, the area is translation invariant.
    The formula using Gram matrix of differences is safer.
    Let's use the determinant of the 3x3 matrix formed by the points in homogeneous coordinates?
    Or simply: Area^2 = 0.25 * ( (p_j - p_i) x (p_k - p_i) )^2
    = 0.25 * ( |p_j-p_i|^2 |p_k-p_i|^2 - ((p_j-p_i) . (p_k-p_i))^2 )
    
    Let d_ij = |p_i - p_j|^2 = G_ii + G_jj - 2 G_ij = 1 + 1 - 2 G_ij = 2(1 - G_ij).
    Let dot_ij = (p_j - p_i) . (p_k - p_i) = p_j.p_k - p_j.p_i - p_i.p_k + p_i.p_i
    = G_jk - G_ji - G_ik + G_ii = G_jk - G_ij - G_ik + 1.
    
    So 4 * Area^2 = d_ij * d_ik - dot_ij^2.
    
    We want to maximize delta = min(4 * Area^2).
    
    We will use a projected gradient ascent on G.
    1. Start with random PSD rank-2 matrix G with diag 1.
    2. Compute gradient of min_area w.r.t G.
       This is tricky because of the min. We use the subgradient of the active constraint.
       Let (i,j,k) be the triangle with minimum area.
       We increase the area of this triangle by moving G.
       
    Alternatively, we can use a simple iterative scheme:
    1. Extract points P from G (via eigendecomposition).
    2. Find min triangle (i,j,k).
    3. Update P by moving one of the points to increase the area of (i,j,k).
    4. Re-orthonormalize/project P to satisfy constraints?
    
    Actually, the "SDP" framing suggests optimizing G directly.
    Let's stick to the P-space but use the structural insight:
    Optimize angles theta_i on the circle.
    Area(i,j,k) = 0.5 * | sin(theta_j - theta_i) + sin(theta_k - theta_j) + sin(theta_i - theta_k) |
    (This is the area of the triangle formed by points on the unit circle).
    
    So the problem reduces to:
    Maximize min_{i,j,k} | sin(t_j - t_i) + sin(t_k - t_j) + sin(t_i - t_k) |
    Subject to t_i in [0, 2pi).
    
    This is a 13-dimensional non-convex optimization.
    We can use Nelder-Mead or Powell on the angles!
    This is MUCH simpler and more robust than 2D coordinates because:
    1. The hull area is fixed (pi).
    2. The points are always on the hull (convex).
    3. The dimension is 13 instead of 26.
    4. The landscape might be smoother.
    
    Let's implement this.
    """
    
    def area_from_angles(t):
        """Compute min triangle area for points on unit circle with angles t."""
        n = len(t)
        min_area = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Area = 0.5 * |sin(t_j - t_i) + sin(t_k - t_j) + sin(t_i - t_k)|
                    # Note: The order of indices in the sine sum matters for the sign, but we take abs.
                    # The formula is 0.5 * | sin(t_j - t_i) + sin(t_k - t_j) + sin(t_i - t_k) |
                    # Let's verify:
                    # Vertices: (cos t_i, sin t_i), etc.
                    # Area = 0.5 | x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j) |
                    # = 0.5 | cos t_i (sin t_j - sin t_k) + cos t_j (sin t_k - sin t_i) + cos t_k (sin t_i - sin t_j) |
                    # Using trig identities, this equals 0.5 | sin(t_j - t_i) + sin(t_k - t_j) + sin(t_i - t_k) |
                    # Let's double check with t_i=0, t_j=pi/2, t_k=pi.
                    # sin(pi/2) + sin(pi/2) + sin(-pi) = 1 + 1 + 0 = 2. Area = 1. Correct (right triangle legs 1,1).
                    
                    val = np.sin(t[j] - t[i]) + np.sin(t[k] - t[j]) + np.sin(t[i] - t[k])
                    area = 0.5 * abs(val)
                    if area < min_area:
                        min_area = area
        return min_area

    def negative_area(t):
        return -area_from_angles(t)

    # Use scipy.optimize.minimize with Powell or Nelder-Mead
    from scipy.optimize import minimize
    
    best_t = None
    best_score = 0.0
    
    # Multistart
    n_starts = 20
    for s in range(n_starts):
        if time.time() > time_budget + time.time() - 10: # check remaining time
             break
             
        # Initial guess: regular polygon + noise
        t0 = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        if s > 0:
            t0 += np.random.uniform(-0.5, 0.5, n_points)
            t0 = np.sort(t0) # keep them ordered? The formula is symmetric, but sorting helps stability
        
        res = minimize(negative_area, t0, method='Powell', 
                       options={'maxiter': 2000, 'disp': False})
        
        t_opt = res.x
        min_area_val = area_from_angles(t_opt)
        
        # Convert to points
        points = np.column_stack((np.cos(t_opt), np.sin(t_opt)))
        
        # Calculate actual score (hull area is pi)
        score = min_area_val / np.pi
        
        if score > best_score:
            best_score = score
            best_t = t_opt.copy()
            best_points = points.copy()
            
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Deterministic geometric optimization using angle-based SDP relaxation.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Strategy: Optimize angles on the unit circle
    # This fixes the hull area to pi, simplifying the objective to maximizing min triangle area.
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: Angle-based optimization
    try:
        points, score = _sdpsolve_heilbronn(13, time_budget=400)
        if points is not None:
            best_points = points
            best_score = score
    except Exception:
        pass
    
    # Phase 2: Refine in 2D coordinates if needed?
    # The angle-based solution is already convex (points on circle).
    # We can try to "inflate" the points slightly off the circle to see if we can increase
    # the min area relative to the hull area.
    # However, for points on a circle, the hull area is fixed.
    # If we move them off the circle, the hull area might decrease (if we move inwards) or stay same (if outwards).
    # Moving inwards usually reduces triangle areas more than it reduces hull area?
    # Actually, the optimal Heilbronn configuration for N=13 is not necessarily on a circle.
    # But it's a very good starting point.
    
    if best_points is None:
        # Fallback: Regular polygon
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
    else:
        # Local refinement in 2D to potentially improve score
        # We use a simple hill climbing step to adjust radial distances
        # while keeping angles fixed? Or full 2D?
        
        # Let's do a quick 2D refinement using the same logic as the seed program
        # but starting from the angle-optimized points.
        
        current_points = best_points.copy()
        current_score = best_score
        
        # Simple gradient-free step
        # We will try to push the points of the minimum triangle apart.
        
        for it in range(200):
            if time.time() > deadline:
                break
                
            min_area, i, j, k = _compute_min_triangle_area_and_indices(current_points)
            
            if min_area < 1e-9:
                break
                
            # Find the point in (i,j,k) that is "most central" or closest to the other two's line
            # and push it away.
            
            # Distances to lines
            def dist_to_line(p, a, b):
                area = 0.5 * abs(a[0]*(b[1]-p[1]) + b[0]*(p[1]-a[1]) + p[0]*(a[1]-b[1]))
                length = np.linalg.norm(b - a)
                if length < 1e-9:
                    return 0
                return 2 * area / length
            
            d_i = dist_to_line(current_points[i], current_points[j], current_points[k])
            d_j = dist_to_line(current_points[j], current_points[i], current_points[k])
            d_k = dist_to_line(current_points[k], current_points[i], current_points[j])
            
            if d_i <= d_j and d_i <= d_k:
                move_idx = i
                v = current_points[k] - current_points[j]
                n = np.array([-v[1], v[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    n = n / n_norm
                    cross = v[0]*(current_points[i][1]-current_points[j][1]) - v[1]*(current_points[i][0]-current_points[j][0])
                    direction = n if cross > 0 else -n
                    current_points[move_idx] += direction * 0.01
            elif d_j <= d_i and d_j <= d_k:
                move_idx = j
                v = current_points[k] - current_points[i]
                n = np.array([-v[1], v[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    n = n / n_norm
                    cross = v[0]*(current_points[j][1]-current_points[i][1]) - v[1]*(current_points[j][0]-current_points[i][0])
                    direction = n if cross > 0 else -n
                    current_points[move_idx] += direction * 0.01
            else:
                move_idx = k
                v = current_points[j] - current_points[i]
                n = np.array([-v[1], v[0]])
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    n = n / n_norm
                    cross = v[0]*(current_points[k][1]-current_points[i][1]) - v[1]*(current_points[k][0]-current_points[i][0])
                    direction = n if cross > 0 else -n
                    current_points[move_idx] += direction * 0.01
            
            new_score = _score(current_points)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Revert
                current_points[move_idx] -= direction * 0.01
                # Try a different point or break?
                # For simplicity, just break if we can't improve easily
                if it > 50:
                    break

    # Final sanity check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points

def _compute_min_triangle_area_and_indices(points):
    n = len(points)
    min_area = float('inf')
    min_i, min_j, min_k = 0, 1, 2
    
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                area = 0.5 * abs(pi[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj[1]))
                if area < min_area:
                    min_area = area
                    min_i, min_j, min_k = i, j, k
                    
    return min_area, min_i, min_j, min_k
