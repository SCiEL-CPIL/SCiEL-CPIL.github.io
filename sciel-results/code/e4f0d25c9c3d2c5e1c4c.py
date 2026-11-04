"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, brentq

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _generate_two_ring_points(r, theta):
    """
    Generate 14 points: 7 on unit circle (outer), 7 on circle of radius r (inner).
    Outer angles: 0, 2pi/7, ...
    Inner angles: theta, theta + 2pi/7, ...
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _analyze_geometry(r, theta):
    """
    Analytically compute the minimum triangle area components for the two-ring structure.
    
    Categories of triangles:
    1. Pure Outer (OOO): 3 points from outer ring.
       Min area occurs for 3 consecutive points (or closest).
       Area_OOO = 0.5 * |cross(P1, P2)| * |sin(angle_diff)| ... actually simpler:
       For regular heptagon on unit circle, min area is formed by 3 consecutive vertices.
       Area_OOO = 0.5 * 1 * 1 * sin(2*pi/7) * (1 + cos(2*pi/7)) ? No.
       Let's just compute it numerically for the specific configuration to be safe, 
       but since it's fixed for outer ring, we can precompute or compute quickly.
       
    2. Pure Inner (III): 3 points from inner ring.
       Area_III = r^2 * Area_OOO (since it's a scaled version).
       
    3. Mixed (OOI or OII):
       - OOI: 2 outer, 1 inner.
       - OII: 1 outer, 2 inner.
       
    The global minimum is min(Area_OOO, Area_III, Min_Area_OOI, Min_Area_OII).
    
    Note: Area_OOO is constant for the outer ring (unit radius).
    Area_III depends on r.
    Mixed areas depend on r and theta.
    """
    # Precompute angles
    d = 2 * np.pi / 7
    
    # 1. Pure Outer Area (fixed)
    # Vertices at angles 0, d, 2d.
    # Area = 0.5 * | (P1-P0) x (P2-P0) |
    p0 = np.array([1.0, 0.0])
    p1 = np.array([np.cos(d), np.sin(d)])
    p2 = np.array([np.cos(2*d), np.sin(2*d)])
    area_ooo = 0.5 * np.abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0]))
    
    # 2. Pure Inner Area
    area_iii = r**2 * area_ooo
    
    # 3. Mixed Areas
    # We need to find the min area among all OOI and OII triangles.
    # Due to symmetry, we can just check a representative set.
    # Outer indices: 0..6. Inner indices: 0..6 (rotated by theta).
    
    min_mixed = float('inf')
    
    # Check OOI: 2 Outer, 1 Inner
    # By symmetry, we can fix the two outer points to be adjacent (indices 0, 1) 
    # or separated by k. The "worst case" (min area) usually involves the inner point 
    # being "between" the outer points in angular projection, or the outer points being close.
    # Actually, for a fixed inner point, the min area OOI triangle is with the two outer points 
    # closest to it in angle? No, it's about the base height.
    # Let's just iterate over all combinations of 2 outer and 1 inner, but exploit symmetry.
    # There are C(7,2)*7 = 21*7 = 147 such triangles.
    # And C(7,1)*C(7,2) = 7*21 = 147 OII triangles.
    # Total 294 mixed triangles. This is very fast to compute numerically.
    
    outer_pts = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)), 
                                  np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False))])
    inner_pts = r * np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False) + theta), 
                                      np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False) + theta)])
    
    # Vectorized check for all mixed triangles
    # OOI: i, j from outer, k from inner
    # We can loop over the 7 inner points and find the min area triangle formed by it and any 2 outer points.
    # Area(P_i, P_j, Q_k) = 0.5 * | (P_i - Q_k) x (P_j - Q_k) |
    
    for k in range(7):
        qk = inner_pts[k]
        # Compute areas for all pairs of outer points with qk
        # Vectorized:
        # v1 = outer_pts - qk (shape 7x2)
        # We need cross product of v1[i] and v1[j] for all i<j.
        # This is equivalent to finding the min area triangle in a set of 7 points on unit circle 
        # with a fixed point qk? No, qk is the vertex, the other two are on unit circle.
        # Area = 0.5 * | (P_i - Q) x (P_j - Q) |.
        # This is 0.5 * | P_i x P_j - Q x (P_i + P_j) |? No.
        # (P_i - Q) x (P_j - Q) = P_i x P_j - P_i x Q - Q x P_j + Q x Q
        # = P_i x P_j - P_i x Q + P_j x Q (since Q x Q = 0 and -Q x P_j = P_j x Q)
        # Wait, (a-b)x(c-b) = a x c - a x b - b x c + b x b = a x c - a x b + c x b.
        
        # Let's just compute it directly for the 21 pairs for each of the 7 inner points.
        # 7 * 21 = 147 operations. Negligible.
        
        for i in range(7):
            for j in range(i+1, 7):
                pi = outer_pts[i]
                pj = outer_pts[j]
                # Area = 0.5 * | (pi - qk) x (pj - qk) |
                v1 = pi - qk
                v2 = pj - qk
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_mixed:
                    min_mixed = area
    
    # OII: 1 Outer, 2 Inner
    # Symmetric to OOI if we swap roles? 
    # Area(P_i, Q_j, Q_k) = 0.5 * | (P_i - Q_j) x (P_k - Q_j) |? No, (Q_j - P_i) x (Q_k - P_i).
    # This is not simply r^2 times OOI because the "fixed" point is on the outer ring.
    # We must check these too.
    
    for i in range(7):
        pi = outer_pts[i]
        for j in range(7):
            for k in range(j+1, 7):
                qj = inner_pts[j]
                qk = inner_pts[k]
                v1 = qj - pi
                v2 = qk - pi
                area = 0.5 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_mixed:
                    min_mixed = area

    # The minimum triangle area is the min of the pure and mixed components
    min_area = min(area_ooo, area_iii, min_mixed)
    
    # Convex hull area
    # The convex hull of the union of two concentric regular heptagons.
    # If r < 1, the outer ring determines the hull, UNLESS the inner ring points stick out?
    # No, inner ring is strictly inside if r < 1.
    # Wait, is it possible for an inner point to be outside the convex hull of the outer points?
    # The convex hull of the outer points is the regular heptagon inscribed in the unit circle.
    # Any point with radius r < 1 is inside the unit circle, but is it inside the heptagon?
    # The apothem of the unit heptagon is cos(pi/7) ≈ 0.9009.
    # If r > cos(pi/7), the inner points are OUTSIDE the convex hull of the outer points!
    # In that case, the convex hull is the convex hull of all 14 points.
    
    # Let's just compute the hull area numerically to be safe and accurate.
    # It's fast enough.
    pts = _generate_two_ring_points(r, theta)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        hull_area = 1.0 # Fallback
        
    return min_area / hull_area

def _optimize_r_for_theta(theta):
    """
    For a fixed theta, find the r that maximizes the minimum triangle area.
    The function f(r) = min(Area_OOO, Area_III(r), Min_Mixed(r, theta)) is concave-ish.
    Area_III increases with r.
    Min_Mixed usually decreases as r increases (points get closer to each other? or geometry changes).
    Actually, as r increases, inner points move outward.
    - Area_III increases.
    - OOI/OII areas change non-monotonically.
    
    We can use a simple 1D search for r in [0.1, 1.0].
    Since we have an analytic formula for the components (mostly), we can optimize this.
    However, the mixed part is computed numerically in _analyze_geometry.
    So _analyze_geometry is the objective function for r.
    """
    # We want to maximize min_area / hull_area.
    # Hull area changes with r if r > cos(pi/7).
    # Let's just maximize the raw min_area / hull_area using minimize_scalar.
    
    def neg_score(r):
        return -_analyze_geometry(r, theta)
        
    # r should be between 0 and 1.
    # If r is too small, Area_III is small.
    # If r is close to 1, points might coincide or mixed areas drop.
    
    res = minimize_scalar(neg_score, bounds=(0.2, 0.95), method='bounded', options={'xatol': 1e-6})
    return res.x, -res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 35s safety margin
    
    best_score = -float('inf')
    best_r = 0.6
    best_theta = np.pi / 7
    
    # 1. Coarse grid search over theta to find promising regions
    # theta is the relative rotation. Due to symmetry, theta in [0, pi/7] is sufficient.
    theta_vals = np.linspace(0, np.pi/7, 20)
    
    coarse_best = -float('inf')
    coarse_best_r = 0.6
    coarse_best_theta = np.pi/7
    
    for th in theta_vals:
        if time.time() > deadline - 30.0:
            break
        # Optimize r for this theta
        r_opt, score_opt = _optimize_r_for_theta(th)
        if score_opt > coarse_best:
            coarse_best = score_opt
            coarse_best_r = r_opt
            coarse_best_theta = th
            
    best_r = coarse_best_r
    best_theta = coarse_best_theta
    best_score = coarse_best
    
    # 2. Refine theta with a finer search around the best coarse theta
    if time.time() < deadline - 10.0:
        # Narrow down theta range
        theta_lo = max(0, best_theta - np.pi/70)
        theta_hi = min(np.pi/7, best_theta + np.pi/70)
        
        def neg_theta_score(th):
            # For each theta, we need to re-optimize r.
            # This is expensive. We can approximate by fixing r to the best r found, 
            # or re-optimizing r quickly.
            # Let's re-optimize r quickly with a tighter bound.
            r_opt, score_opt = _optimize_r_for_theta(th)
            return -score_opt
            
        res_th = minimize_scalar(neg_theta_score, bounds=(theta_lo, theta_hi), method='bounded', options={'xatol': 1e-6})
        best_theta = res_th.x
        # Get the corresponding best r
        best_r, best_score = _optimize_r_for_theta(best_theta)
        
        # 3. Final Polish: Joint optimization of r and theta?
        # Or just verify and maybe do a tiny local search on the 2D (r, theta) space
        # using the analytic score function.
        
        if time.time() < deadline - 5.0:
            def neg_2d(params):
                r, th = params
                if r < 0.1 or r > 0.99: return 1e9
                return -_analyze_geometry(r, th)
                
            x0 = [best_r, best_theta]
            # Use Nelder-Mead on the 2D space
            try:
                from scipy.optimize import minimize
                res = minimize(neg_2d, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100})
                if res.fun < -best_score:
                    best_r, best_theta = res.x
                    best_score = -res.fun
            except Exception:
                pass

    # Generate final points
    best_pts = _generate_two_ring_points(best_r, best_theta)
    
    # One final score check
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score
        
    return best_pts
# EVOLVE-BLOCK-END
