import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50
START_TIME = time.time()
DEADLINE = 140.0  # Leave 10s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = params.reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere.
    
    Uses convex hull approach: for each triangular face, compute the angular
    circumradius. The covering radius is the maximum of these.
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # worst case
    
    max_circum = 0.0
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
        
        # Angular distances from pole to each vertex
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        max_circum = max(max_circum, circum_angular)
    
    return max_circum

def smooth_covering_radius(points, temperature=0.01):
    """Smooth surrogate for covering radius using log-sum-exp.
    
    This provides a differentiable approximation that helps gradient-based
    optimization escape local minima.
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points.flatten())
    
    # Compute angular circumradii for all faces
    circumradii = []
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
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
        
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        circumradii.append(circum_angular)
    
    if not circumradii:
        return np.pi, np.zeros_like(points.flatten())
    
    circumradii = np.array(circumradii)
    
    # Log-sum-exp for smooth max
    max_val = np.max(circumradii)
    if temperature < 1e-10:
        return max_val, np.zeros_like(points.flatten())
    
    # Smooth max approximation
    smooth_max = max_val + temperature * np.log(np.sum(np.exp((circumradii - max_val) / temperature)))
    
    # Approximate gradient: weight each face's contribution
    weights = np.exp((circumradii - max_val) / temperature)
    weights /= np.sum(weights)
    
    # Gradient with respect to points (approximate)
    grad = np.zeros_like(points)
    for i, simplex in enumerate(hull.simplices):
        w = weights[i]
        if w < 1e-8:
            continue
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
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
        
        # Gradient of angular distance w.r.t. each vertex
        # d/dx arccos(p·x) = -p / sqrt(1-(p·x)^2)
        for j, v in enumerate([a, b, c]):
            dot_val = np.dot(pole, v)
            dot_val = np.clip(dot_val, -0.999999, 0.999999)
            denom_g = np.sqrt(1.0 - dot_val * dot_val)
            if denom_g < 1e-8:
                continue
            grad[simplex[j]] -= w * pole / denom_g
    
    return smooth_max, grad.flatten()

def local_optimize_smooth(initial_points, max_iter=200, tol=1e-10, temperature_schedule=None):
    """Local optimization using smooth surrogate with adaptive temperature."""
    params = initial_points.flatten()
    
    def obj_and_grad(p):
        points = to_sphere(p)
        val, grad = smooth_covering_radius(points, temperature=0.005)
        return val, grad
    
    best_params = params.copy()
    best_val, _ = obj_and_grad(best_params)
    
    # Use L-BFGS-B with the smooth surrogate
    try:
        result = minimize(obj_and_grad, params, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': max_iter, 'gtol': tol, 'ftol': tol})
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Refine with exact covering radius using Nelder-Mead
    def obj_exact(p):
        points = to_sphere(p)
        return compute_covering_radius(points)
    
    try:
        result2 = minimize(obj_exact, best_params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12,
                                   'adaptive': True})
        if result2.fun < best_val:
            best_params = result2.x
            best_val = result2.fun
    except Exception:
        pass
    
    return to_sphere(best_params), best_val

def simulated_annealing(initial_points, max_iter=500, t0=0.1, t1=0.001, cooling=0.99):
    """Simulated annealing on the sphere."""
    params = initial_points.flatten()
    current_val = compute_covering_radius(to_sphere(params))
    best_params = params.copy()
    best_val = current_val
    
    rng = np.random.RandomState(42)
    temperature = t0
    
    for i in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Generate perturbation
        step_size = 0.05 * (1.0 + 0.5 * np.sin(i * 0.1))
        noise = step_size * rng.randn(N, 3)
        new_params = params + noise
        new_points = to_sphere(new_params)
        new_val = compute_covering_radius(new_points)
        
        # Acceptance criterion
        delta = new_val - current_val
        if delta < 0 or rng.rand() < np.exp(-delta / max(temperature, 1e-10)):
            params = new_params
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_params = params.copy()
        
        # Cool down
        temperature = max(t1, temperature * cooling)
    
    return to_sphere(best_params), best_val

def random_init(n_points):
    """Generate random points on the unit sphere."""
    v = np.random.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def icosahedron_based_init(n_points):
    """Generate points based on icosahedron vertices with subdivision."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide and project to sphere
    points = [verts[i] for i in range(12)]
    
    # Add midpoints of edges
    edges = [(0,1),(1,3),(3,11),(11,0),(0,5),(5,1),(1,6),(6,3),(3,7),
             (7,11),(11,10),(10,0),(0,10),(5,4),(4,1),(1,2),(2,6),(6,8),
             (8,3),(3,9),(9,7),(7,10),(10,4),(4,11),(11,5),(5,2),(2,8),
             (8,9),(9,10)]
    
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid = mid / np.linalg.norm(mid)
        points.append(mid)
    
    # Add more points by subdividing faces
    faces = [(0,1,5),(1,6,5),(1,2,6),(3,6,2),(3,7,9),(3,8,7),(11,7,10),
             (11,9,7),(11,10,4),(11,4,5),(0,10,11),(0,5,10),(2,4,8),(2,8,9),
             (4,2,5),(4,5,9),(9,5,4),(8,9,2),(6,2,8),(7,8,9)]
    
    for f in faces:
        centroid = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3
        centroid = centroid / np.linalg.norm(centroid)
        points.append(centroid)
    
    points = np.array(points)
    
    # If we have more than needed, randomly sample; if fewer, add random
    if len(points) >= n_points:
        idx = np.random.choice(len(points), n_points, replace=False)
        return points[idx]
    else:
        extra = random_init(n_points - len(points))
        return np.vstack([points, extra])

def high_precision_polish(points, max_iter=1000):
    """Final high-precision polish with very tight tolerances."""
    params = points.flatten()
    
    def obj(p):
        pts = to_sphere(p)
        return compute_covering_radius(pts)
    
    best_params = params.copy()
    best_val = obj(best_params)
    
    # Multiple passes with decreasing step sizes
    for scale in [1e-4, 1e-5, 1e-6]:
        try:
            result = minimize(obj, best_params, method='Nelder-Mead',
                              options={'maxiter': max_iter, 'xatol': 1e-14, 'fatol': 1e-14,
                                       'adaptive': True, 'maxfev': max_iter * 10})
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
    
    return to_sphere(best_params), best_val

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # Multiple initializations
    inits = [
        ("fibonacci", fibonacci_spiral(N)),
        ("icosahedron", icosahedron_based_init(N)),
    ]
    
    # Add random inits
    rng = np.random.RandomState(42)
    for _ in range(3):
        inits.append(("random", random_init(N)))
    
    best_points = None
    best_cr = np.pi
    
    # Phase 1: Local optimization from diverse starts
    for name, init in inits:
        if time.time() > DEADLINE - 30:
            break
        
        optimized, cr = local_optimize_smooth(init, max_iter=150)
        if best_points is None or cr < best_cr:
            best_cr = cr
            best_points = optimized
    
    # Phase 2: Simulated annealing from best
    if time.time() < DEADLINE - 20 and best_points is not None:
        sa_points, sa_cr = simulated_annealing(best_points, max_iter=300)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points
    
    # Phase 3: More local optimization from current best
    if time.time() < DEADLINE - 15 and best_points is not None:
        optimized, cr = local_optimize_smooth(best_points, max_iter=200)
        if cr < best_cr:
            best_cr = cr
            best_points = optimized
    
    # Phase 4: High-precision polish
    if time.time() < DEADLINE - 10 and best_points is not None:
        polished, pol_cr = high_precision_polish(best_points, max_iter=500)
        if pol_cr < best_cr:
            best_cr = pol_cr
            best_points = polished
    
    # Ensure all points are on the unit sphere
    if best_points is None:
        best_points = fibonacci_spiral(N)
    
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    # Validate: check no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
