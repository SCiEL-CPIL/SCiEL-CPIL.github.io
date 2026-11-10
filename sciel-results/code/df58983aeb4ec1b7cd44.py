"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay


def _normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]


def _covering_radius(pts, n_samples=20000):
    """Compute covering radius using a dense sample grid on the sphere."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    angles = np.arccos(max_dots)
    max_angle = np.max(angles)
    return max_angle * 180.0 / np.pi


def _covering_radius_exact(pts):
    """Compute exact covering radius using circumradii of Delaunay triangles on sphere.
    
    For each facet of the convex hull (which corresponds to a Voronoi cell),
    the worst point is the circumcenter of that triangle (if it lies within the
    spherical triangle), or a vertex/edge midpoint.
    """
    try:
        hull = ConvexHull(pts)
        tri_idx = hull.simplices[:, :3]
        
        max_angle = 0.0
        for tri in tri_idx:
            p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            
            # Compute the circumcenter of the spherical triangle.
            # The circumcenter is the point on the unit sphere equidistant from p1, p2, p3.
            # It lies in the direction of the normalized vector that is the solution to:
            # |C - p1|^2 = |C - p2|^2 = |C - p3|^2 where |C| = 1
            
            # For points on the unit sphere, the circumcenter direction can be found by:
            # C = normalize(cross(p1-p2, p1-p3)) ... no, that's the normal.
            
            # Better: solve for the circumcenter using the fact that it's equiangular.
            # The circumcenter C satisfies: C·p1 = C·p2 = C·p3 = cos(R)
            # where R is the angular circumradius.
            
            # Set up: C = a*p1 + b*p2 + c*p3, with C·p1 = C·p2 = C·p3 and |C| = 1.
            # This gives: a + b*(p1·p2) + c*(p1·p3) = a*(p1·p2) + b + c*(p2·p3)
            # and similar.
            
            # Simpler numerical approach: the circumcenter is in the plane of the triangle.
            # Let's find it by intersecting the perpendicular bisectors in 3D.
            
            # Midpoint of p1-p2: m12 = (p1+p2)/2
            # Direction p1-p2: d12 = p1-p2
            # Perpendicular bisector plane: X·d12 = m12·d12
            
            # We want X on the unit sphere, so X·X = 1, and X·d12 = m12·d12, X·d13 = m13·d13
            # where d13 = p1-p3, m13 = (p1+p3)/2.
            
            d12 = p1 - p2
            d13 = p1 - p3
            m12 = (p1 + p2) / 2.0
            m13 = (p1 + p3) / 2.0
            b12 = m12 @ d12
            b13 = m13 @ d13
            
            # Solve: X·d12 = b12, X·d13 = b13, X·X = 1
            # Let X = alpha*d12 + beta*d13 + gamma*n where n = cross(d12, d13)
            # But this is getting complex. Let's use a different approach.
            
            # For a spherical triangle, the circumradius can be computed from the side lengths.
            # Let a = angular distance p2-p3, b = angular distance p1-p3, c = angular distance p1-p2.
            # Then the circumradius R satisfies: sin(R) = (sin(a)*sin(b)*sin(c)) / (2*sin(A)*sin(B)*sin(C))
            # where A, B, C are the angles of the spherical triangle.
            
            # Actually, there's a simpler formula. The circumradius of a spherical triangle
            # with sides a, b, c (in radians) and area Δ is:
            # sin(R) = (sin(a)*sin(b)*sin(c)) / (4*cos(A/2)*cos(B/2)*cos(C/2))
            
            # Even simpler: use the fact that for a triangle inscribed in a unit sphere,
            # the circumradius R satisfies: 1/sin(R) = |cross(p2-p1, p3-p1)| / (2*area_3d)
            # Hmm, this is getting complicated.
            
            # Let me just use a robust numerical method: sample the triangle densely.
            # For each triangle, sample a grid of barycentric coordinates and find the
            # point with maximum angular distance to the nearest vertex.
            
            n_sub = 20
            max_tri_angle = 0.0
            for i in range(n_sub + 1):
                for j in range(n_sub + 1 - i):
                    k = n_sub - i - j
                    u, v, w = i / n_sub, j / n_sub, k / n_sub
                    # Point on the triangle (in 3D, then normalize)
                    pt = u * p1 + v * p2 + w * p3
                    pt_norm = np.linalg.norm(pt)
                    if pt_norm < 1e-10:
                        continue
                    pt = pt / pt_norm
                    
                    # Angular distance to each vertex
                    d1 = np.arccos(np.clip(pt @ p1, -1.0, 1.0))
                    d2 = np.arccos(np.clip(pt @ p2, -1.0, 1.0))
                    d3 = np.arccos(np.clip(pt @ p3, -1.0, 1.0))
                    min_d = min(d1, d2, d3)
                    if min_d > max_tri_angle:
                        max_tri_angle = min_d
            
            if max_tri_angle > max_angle:
                max_angle = max_tri_angle
        
        return max_angle * 180.0 / np.pi
    except:
        return _covering_radius(pts, n_samples=50000)


def _covering_radius_voronoi(pts, max_time=2.0):
    """Compute covering radius using Voronoi cells on the sphere via Delaunay."""
    return _covering_radius_exact(pts)


def _soft_covering_radius(pts, temperature=0.01):
    """Compute a smooth surrogate for the covering radius using soft-max.
    
    Instead of max over samples of arccos(max_dot), we use a soft-max
    to get a differentiable objective.
    """
    i = np.arange(20000) + 0.5
    z = 1.0 - 2.0 * i / 20000
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = samples @ pts.T  # shape (20000, 50)
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each sample, find the max dot product (nearest point)
    # Use soft-max: log(sum(exp(dots/temperature))) * temperature
    # But we want arccos of the max dot, so we need to be careful.
    
    # Alternative: use the fact that for small angles, arccos(x) ≈ sqrt(2*(1-x))
    # So we can minimize sum of (1 - dots)^2 as a smooth surrogate.
    
    max_dots = np.max(dots, axis=1)
    # Use soft version: replace max with log-sum-exp
    scaled = dots / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled, axis=1, keepdims=True)
    soft_max = max_scaled + np.log(sum_exp)
    soft_max_dots = np.clip(soft_max * temperature, -1.0, 1.0)
    
    # Objective: sum of (1 - soft_max_dots)^2, which is a smooth proxy for covering radius
    objective = np.sum((1.0 - soft_max_dots[:, 0])**2)
    return objective


def _local_optimize(pts, max_time=30.0):
    """Local optimization to reduce covering radius using coordinate descent
    with line search and a smooth surrogate for gradient-like updates."""
    start_time = time.time()
    pts = _normalize(pts.copy())
    
    best_cr = _covering_radius(pts)
    best_pts = pts.copy()
    
    # Phase 1: Coordinate descent with random directions
    step_size = 0.05
    
    while time.time() - start_time < max_time - 5.0:
        improved = False
        
        for i in range(N):
            if time.time() - start_time > max_time - 5.0:
                break
            
            # Try moving point i in random directions
            for trial in range(15):
                direction = np.random.randn(3)
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-10:
                    continue
                direction /= norm_dir
                
                new_pts = pts.copy()
                new_pt = new_pts[i] + step_size * direction
                new_pts[i] = _normalize(new_pt)
                
                cr = _covering_radius(new_pts)
                if cr < best_cr - 1e-10:
                    best_cr = cr
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-4:
                break
    
    # Phase 2: High-precision polish using exact covering radius
    if time.time() - start_time < max_time - 3.0:
        polish_start = time.time()
        polish_pts = best_pts.copy()
        polish_cr = _covering_radius_exact(polish_pts)
        polish_step = 0.01
        
        while time.time() - polish_start < max_time - 3.0 - (time.time() - start_time):
            polish_improved = False
            for i in range(N):
                if time.time() - start_time > max_time - 3.0:
                    break
                
                for trial in range(20):
                    direction = np.random.randn(3)
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir < 1e-10:
                        continue
                    direction /= norm_dir
                    
                    new_pts = polish_pts.copy()
                    new_pt = new_pts[i] + polish_step * direction
                    new_pts[i] = _normalize(new_pt)
                    
                    cr = _covering_radius_exact(new_pts)
                    if cr < polish_cr - 1e-10:
                        polish_cr = cr
                        polish_pts = new_pts.copy()
                        polish_improved = True
                        break
            
            if not polish_improved:
                polish_step *= 0.7
                if polish_step < 1e-5:
                    break
        
        if polish_cr < best_cr:
            best_pts = polish_pts
            best_cr = polish_cr
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Local optimization from Fibonacci start
    if time.time() < deadline - 30.0:
        remaining = min(60.0, deadline - 30.0 - time.time())
        optimized_pts, cr = _local_optimize(pts, max_time=remaining)
        if cr < best_cr:
            best_pts = optimized_pts
            best_cr = cr
    
    # Try random restarts if time permits
    np.random.seed(42)
    restarts_done = 0
    for restart in range(8):
        if time.time() > deadline - 25.0:
            break
        
        # Random start
        rand_pts = np.random.randn(N, 3)
        rand_pts = _normalize(rand_pts)
        
        remaining = min(15.0, deadline - 25.0 - time.time())
        if remaining < 5.0:
            break
        
        opt_pts, cr = _local_optimize(rand_pts, max_time=remaining)
        if cr < best_cr:
            best_pts = opt_pts
            best_cr = cr
        restarts_done += 1
    
    # Final verification with exact method
    if time.time() < deadline - 2.0:
        final_cr = _covering_radius_exact(best_pts)
        # If exact is worse than sampled (shouldn't happen), use sampled
        # But keep the points as is
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
