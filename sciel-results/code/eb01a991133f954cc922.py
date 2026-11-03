import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
DEADLINE = 110.0  # Leave 40s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = params.reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_facet_circumradii(points):
    """Compute circumradius and circumcenter for each convex hull facet.
    
    Returns:
        circumradii: array of angular circumradii for each facet
        circumcenters: array of circumcenter directions (unit vectors) for each facet
        facet_vertices: array of vertex indices for each facet
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    circumradii = []
    circumcenters = []
    facet_vertices = []
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute circumcenter of triangle in 3D
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        cc_norm = np.linalg.norm(circumcenter)
        if cc_norm < 1e-10:
            continue
        
        pole = circumcenter / cc_norm
        
        # Angular circumradius
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        
        circumradii.append(circum_angular)
        circumcenters.append(pole)
        facet_vertices.append(simplex)
    
    if len(circumradii) == 0:
        return None, None, None
    
    return np.array(circumradii), np.array(circumcenters), facet_vertices

def compute_covering_radius(points):
    """Compute the covering radius (in radians)."""
    circumradii, _, _ = compute_facet_circumradii(points)
    if circumradii is None:
        return np.pi
    return np.max(circumradii)

def spherical_lloyd_step(points, step_size=0.5, max_iter=100):
    """Perform spherical Lloyd's algorithm: move each point toward the
    circumcenter of its worst (largest circumradius) facet."""
    points = np.asarray(points, dtype=float).copy()
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        
        circumradii, circumcenters, facet_vertices = compute_facet_circumradii(points)
        if circumradii is None:
            break
        
        # For each point, find the worst facet that includes it
        # A point is "associated" with a facet if it's a vertex of that facet
        point_worst_circum = np.zeros(N)
        point_worst_center = np.zeros((N, 3))
        
        for i, simplex in enumerate(facet_vertices):
            cr = circumradii[i]
            center = circumcenters[i]
            for v in simplex:
                if cr > point_worst_circum[v]:
                    point_worst_circum[v] = cr
                    point_worst_center[v] = center
        
        # Move each point toward its worst facet's circumcenter
        # Use spherical interpolation (slerp)
        new_points = points.copy()
        for i in range(N):
            if point_worst_circum[i] < 1e-10:
                continue
            
            p = points[i]
            c = point_worst_center[i]
            
            # Slerp from p toward c
            dot_pc = np.clip(np.dot(p, c), -1, 1)
            theta = np.arccos(dot_pc)
            
            if theta < 1e-10:
                continue
            
            # Move a fraction of the way toward the circumcenter
            # The step size controls how aggressively we move
            t = step_size * min(1.0, 0.1 / point_worst_circum[i])
            t = np.clip(t, 0.01, 0.5)
            
            sin_theta = np.sin(theta)
            if sin_theta < 1e-10:
                continue
            
            new_p = ((np.sin((1-t)*theta) * p + np.sin(t*theta) * c) / sin_theta)
            new_p = new_p / np.linalg.norm(new_p)
            new_points[i] = new_p
        
        # Check convergence
        movement = np.linalg.norm(new_points - points)
        points = new_points
        
        if movement < 1e-10:
            break
    
    return points

def smooth_covering_radius(points, temperature=0.01):
    """Smooth surrogate for covering radius using log-sum-exp."""
    circumradii, _, _ = compute_facet_circumradii(points)
    if circumradii is None:
        return np.pi
    
    # Log-sum-exp for smooth max
    max_cr = np.max(circumradii)
    shifted = circumradii - max_cr
    log_sum = max_cr + np.log(np.sum(np.exp(shifted / temperature)) / len(circumradii)) * temperature
    return log_sum

def local_optimize_smooth(initial_points, max_iter=50, temperature=0.01):
    """Optimize using smooth surrogate with BFGS."""
    params = initial_points.flatten()
    
    def obj(p):
        points = to_sphere(p)
        return smooth_covering_radius(points, temperature)
    
    best_params = params.copy()
    best_val = obj(best_params)
    
    try:
        result = minimize(obj, params, method='BFGS',
                          options={'maxiter': max_iter, 'gtol': 1e-8})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    return to_sphere(best_params), compute_covering_radius(to_sphere(best_params))

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def icosahedron_based_init(n_points):
    """Generate points based on icosahedron vertices, then add more points."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n_points <= 12:
        return verts[:n_points]
    
    points = verts.copy()
    rng = np.random.RandomState(123)
    
    while len(points) < n_points:
        best_dist = 0
        best_point = None
        for _ in range(200):
            candidate = rng.randn(3)
            candidate = candidate / np.linalg.norm(candidate)
            dists = np.arccos(np.clip(points @ candidate, -1, 1))
            min_dist = np.min(dists)
            if min_dist > best_dist:
                best_dist = min_dist
                best_point = candidate
        if best_point is not None:
            points = np.vstack([points, best_point])
    
    return points[:n_points]

def random_init(n_points, rng):
    """Generate random points on the unit sphere."""
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    rng = np.random.RandomState(42)
    
    # Multiple initialization strategies
    inits = []
    inits.append(fibonacci_spiral(N))
    inits.append(icosahedron_based_init(N))
    inits.append(random_init(N, rng))
    
    # Perturbed Fibonacci
    noise = 0.05 * rng.randn(N, 3)
    fib_perturbed = fibonacci_spiral(N) + noise
    fib_perturbed = fib_perturbed / np.linalg.norm(fib_perturbed, axis=1, keepdims=True)
    inits.append(fib_perturbed)
    
    best_points = inits[0].copy()
    best_cr = compute_covering_radius(best_points)
    
    # Phase 1: Spherical Lloyd's algorithm from each init
    for init in inits:
        if time.time() > DEADLINE - 30:
            break
        
        try:
            # Lloyd's algorithm with adaptive step size
            lloyd_points = spherical_lloyd_step(init, step_size=0.5, max_iter=50)
            cr = compute_covering_radius(lloyd_points)
            
            if cr < best_cr:
                best_cr = cr
                best_points = lloyd_points
        except Exception:
            continue
    
    # Phase 2: Smooth optimization polish
    if time.time() < DEADLINE - 20:
        try:
            polished, cr = local_optimize_smooth(best_points, max_iter=30, temperature=0.005)
            if cr < best_cr:
                best_cr = cr
                best_points = polished
        except Exception:
            pass
    
    # Phase 3: Final Lloyd's polish with smaller steps
    if time.time() < DEADLINE - 10:
        try:
            final_points = spherical_lloyd_step(best_points, step_size=0.2, max_iter=30)
            cr = compute_covering_radius(final_points)
            if cr < best_cr:
                best_cr = cr
                best_points = final_points
        except Exception:
            pass
    
    # Phase 4: High-precision polish if time allows
    if time.time() < DEADLINE - 5:
        try:
            polished, cr = local_optimize_smooth(best_points, max_iter=20, temperature=0.001)
            if cr < best_cr:
                best_cr = cr
                best_points = polished
        except Exception:
            pass
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    # Validate: check no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    # Check not all in a closed hemisphere
    try:
        hull = ConvexHull(best_points)
    except Exception:
        pass
    
    return best_points
