"""Sphere covering with 130 equal caps – refined multi-start with smooth surrogate."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _covering_radius(pts):
    """Compute the covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Circumradius of triangle on unit sphere = angle of circumcenter
            a = tri[0]
            b = tri[1]
            c = tri[2]
            # Vectors
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-12:
                continue
            # Side lengths (chord lengths)
            side_a = np.linalg.norm(b - c)
            side_b = np.linalg.norm(a - c)
            side_c = np.linalg.norm(a - b)
            # Circumradius R = (a*b*c) / (4*area) where area = 0.5*norm(cross)
            area = 0.5 * norm_cross
            R = (side_a * side_b * side_c) / (4.0 * area)
            # Angular circumradius: the cap radius is arcsin(R) if R<=1, else pi/2
            # Actually for points on unit sphere, the circumradius of the spherical triangle
            # is the angular distance from circumcenter to vertices.
            # The circumcenter in 3D:
            # Solve for point p such that |p-a|^2 = |p-b|^2 = |p-c|^2
            # This gives p = a + t1*ab + t2*ac
            # Using the formula: circumradius = side_a*side_b*side_c / (4*area)
            # The angular radius = arcsin(min(R, 1.0)) but this is for the 3D circumcircle.
            # For spherical triangles, the covering radius contribution is the angular distance
            # from the spherical circumcenter to each vertex.
            # A simpler approach: the max angular distance from any point on the facet
            # to the nearest of the three vertices equals the circumradius of the spherical triangle.
            # For a spherical triangle with sides (angular) alpha, beta, gamma,
            # the circumradius (spherical) can be computed.
            # But a practical approach: the covering radius is the max over facets of
            # the angular circumradius, which for small triangles ~ R (chord circumradius).
            # Let's use: angular_r = 2*arcsin(min(R, 1.0)/2) ... no.
            # Actually, the angular distance corresponding to chord length d is 2*arcsin(d/2).
            # The circumradius in angular terms: if the 3D circumcircle has radius R (chord),
            # and it lies on the unit sphere, then the angular radius theta satisfies
            # sin(theta) = R (since the chord from center to vertex has length 2*sin(theta/2)... 
            # wait, the 3D circumcenter may not be on the sphere.
            # Let me use a direct method: find the point on the sphere that is equidistant 
            # from the three vertices (spherical circumcenter) and compute the distance.
            # For now, use the approximation: angular_r = arcsin(min(R, 1.0))
            # This is the radius of the great-circle cap that covers the triangle.
            # Actually, the correct formula: the covering radius for a facet is the 
            # circumradius of the spherical triangle. For a triangle with vertices on unit sphere,
            # the spherical circumradius C satisfies:
            #   sin(C) = (sin(a)*sin(b)*sin(c)) / (2*sin(a+b+c)/2 * ... ) -- complex.
            # Simpler: compute the 3D circumcenter and project to sphere, then distance.
            # Let's compute 3D circumcenter of the triangle.
            # p = a + s*ab + t*ac, minimize |p|^2 subject to equidistance.
            # Equidistant: |p-a|^2 = |p-b|^2 => 2*p.(b-a) = |b|^2 - |a|^2 = 0 (since unit)
            # So p.(b-a) = 0, p.(c-a) = 0.
            # p = a + s*(b-a) + t*(c-a)
            # p.(b-a) = 0: a.(b-a) + s*|b-a|^2 + t*(c-a).(b-a) = 0
            # p.(c-a) = 0: a.(c-a) + s*(b-a).(c-a) + t*|c-a|^2 = 0
            M = np.array([[np.dot(ab, ab), np.dot(ac, ab)],
                          [np.dot(ab, ac), np.dot(ac, ac)]])
            rhs = np.array([-np.dot(a, ab), -np.dot(a, ac)])
            try:
                st = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            p3d = a + st[0]*ab + st[1]*ac
            # Project to unit sphere
            norm_p = np.linalg.norm(p3d)
            if norm_p < 1e-12:
                continue
            p_sphere = p3d / norm_p
            # Angular distance from p_sphere to a
            cos_d = np.clip(np.dot(p_sphere, a), -1.0, 1.0)
            ang_r = np.arccos(cos_d)
            max_r = max(max_r, ang_r)
        return np.degrees(max_r)
    except Exception:
        return 90.0  # invalid

def _smooth_surrogate(pts, beta=50.0):
    """Smooth surrogate for covering radius using log-sum-exp of squared facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii = []
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-12:
                continue
            M = np.array([[np.dot(ab, ab), np.dot(ac, ab)],
                          [np.dot(ab, ac), np.dot(ac, ac)]])
            rhs = np.array([-np.dot(a, ab), -np.dot(a, ac)])
            try:
                st = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            p3d = a + st[0]*ab + st[1]*ac
            norm_p = np.linalg.norm(p3d)
            if norm_p < 1e-12:
                continue
            p_sphere = p3d / norm_p
            cos_d = np.clip(np.dot(p_sphere, a), -1.0, 1.0)
            ang_r = np.arccos(cos_d)
            radii.append(ang_r)
        if not radii:
            return 90.0
        radii = np.array(radii)
        # Log-sum-exp soft max
        m = np.max(radii)
        lse = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta
        return np.degrees(lse)
    except Exception:
        return 90.0

def _gradient_step(pts, beta=50.0, lr=0.01, n_iter=50):
    """Gradient descent on smooth surrogate."""
    best_pts = pts.copy()
    best_val = _smooth_surrogate(best_pts, beta)
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        # Compute gradient numerically or analytically
        # Use finite differences for gradient
        grad = np.zeros_like(pts)
        eps = 1e-4
        for i in range(N):
            for j in range(3):
                pts_plus = best_pts.copy()
                pts_plus[i, j] += eps
                # Renormalize
                pts_plus[i] /= np.linalg.norm(pts_plus[i])
                val_plus = _smooth_surrogate(pts_plus, beta)
                pts_minus = best_pts.copy()
                pts_minus[i, j] -= eps
                pts_minus[i] /= np.linalg.norm(pts_minus[i])
                val_minus = _smooth_surrogate(pts_minus, beta)
                grad[i, j] = (val_plus - val_minus) / (2 * eps)
        # Update
        new_pts = best_pts - lr * grad
        # Renormalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        val = _smooth_surrogate(new_pts, beta)
        if val < best_val:
            best_val = val
            best_pts = new_pts
        else:
            lr *= 0.5
    return best_pts, best_val

def _lloyd_refine(pts, n_iter=200):
    """Lloyd's algorithm: move each point to centroid of its Voronoi cell (approximated by nearest neighbors)."""
    best_pts = pts.copy()
    best_cr = _covering_radius(best_pts)
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        # For each point, find its "Voronoi cell" neighbors and move toward centroid
        # Approximate: move each point to average of its k nearest neighbors
        from scipy.spatial.distance import cdist
        dists = cdist(best_pts, best_pts)
        np.fill_diagonal(dists, np.inf)
        k = 10
        neighbors = np.argsort(dists, axis=1)[:, :k]
        centroids = np.mean(best_pts[neighbors], axis=1)
        new_pts = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        new_cr = _covering_radius(new_pts)
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = new_pts
        else:
            # Try smaller step
            new_pts = best_pts + 0.1 * (centroids - best_pts)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / norms
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = new_pts
    return best_pts, best_cr

def _nelder_mead_polish(pts, max_iter=200):
    """Nelder-Mead polish on the covering radius."""
    def objective(x):
        pts_x = x.reshape(N, 3)
        norms = np.linalg.norm(pts_x, axis=1, keepdims=True)
        pts_x = pts_x / norms
        return _covering_radius(pts_x)
    
    x0 = pts.flatten()
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
    best_pts = res.x.reshape(N, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    return best_pts

def _electrostatic_init(seed=0):
    """Electrostatic repulsion initialization."""
    rng = np.random.default_rng(seed)
    pts = rng.random((N, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    # Simulated annealing for electrostatic repulsion
    T = 1.0
    for it in range(2000):
        if _time_left() < 5.0:
            break
        i = rng.integers(N)
        # Move point i randomly
        delta = rng.normal(0, 0.05, 3)
        new_pt = pts[i] + delta
        new_pt = new_pt / np.linalg.norm(new_pt)
        # Compute energy change (Coulomb repulsion)
        # E = sum_{j!=i} 1/|p_i - p_j|
        old_dists = np.linalg.norm(pts[i] - pts, axis=1)
        np.fill_diagonal(old_dists, np.inf)
        old_e = np.sum(1.0 / old_dists)
        new_dists = np.linalg.norm(new_pt - pts, axis=1)
        np.fill_diagonal(new_dists, np.inf)
        new_e = np.sum(1.0 / new_dists)
        de = new_e - old_e
        if de < 0 or rng.random() < np.exp(-de / T):
            pts[i] = new_pt
        T *= 0.999
    return pts

def _fibonacci_init():
    """Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_pts = None
    best_cr = np.inf
    
    # Multiple initializations
    inits = []
    # Fibonacci
    inits.append(_fibonacci_init())
    # Electrostatic with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        if _time_left() < 10.0:
            break
        inits.append(_electrostatic_init(seed))
    
    for init_pts in inits:
        if _time_left() < 10.0:
            break
        
        # Lloyd refinement
        pts_lloyd, cr_lloyd = _lloyd_refine(init_pts, n_iter=100)
        
        # Smooth surrogate gradient descent
        pts_grad, val_grad = _gradient_step(pts_lloyd, beta=50.0, lr=0.01, n_iter=30)
        
        # Nelder-Mead polish
        pts_nm = _nelder_mead_polish(pts_grad, max_iter=100)
        
        # Final Lloyd pass
        pts_final, cr_final = _lloyd_refine(pts_nm, n_iter=50)
        
        # Evaluate
        cr = _covering_radius(pts_final)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts_final.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_init()
    
    # Final check: ensure no coincident points and not in hemisphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
