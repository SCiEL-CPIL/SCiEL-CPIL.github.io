"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return v / norms

def _covering_radius_from_hull(points):
    """Compute covering radius using convex hull circumradii."""
    try:
        hull = ConvexHull(points)
        # For each facet, compute circumradius
        max_radius = 0.0
        for simplex in hull.simplices:
            tri = points[simplex]
            # Circumradius of triangle on sphere: use formula
            a = np.linalg.norm(tri[1] - tri[2])
            b = np.linalg.norm(tri[0] - tri[2])
            c = np.linalg.norm(tri[0] - tri[1])
            # Area of triangle
            cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            area = 0.5 * np.linalg.norm(cross)
            if area < 1e-15:
                continue
            # Circumradius R = abc / (4A) for planar triangle
            # But we need spherical circumradius (angular distance)
            # For points on unit sphere, the circumradius in angle is:
            # sin(R) = abc / (4 * area_3d) where area_3d is 3D triangle area
            # Actually, the circumradius of the spherical triangle is the angular distance
            # from circumcenter to vertices. The circumcenter is along the normal.
            # For a triangle on unit sphere with side lengths a,b,c (chord lengths),
            # the spherical circumradius R satisfies: sin(R) = abc/(4*V) where V is tetrahedron volume
            # Simpler: compute circumcenter of 3 points in 3D, then angle from origin
            # Circumcenter of triangle: intersection of perpendicular bisectors
            # For 3 points, circumcenter = (a^2(b+c) + b^2(a+c) + c^2(a+b)) / (2*(a*b + b*c + c*a)) ... no
            # Let's just compute it properly
            A, B, C = tri[0], tri[1], tri[2]
            # Circumcenter formula
            d = 2 * (A[0]*(B[1]*C[2]-C[1]*B[2]) - B[0]*(A[1]*C[2]-C[1]*A[2]) + C[0]*(A[1]*B[2]-B[1]*A[2]))
            if abs(d) < 1e-15:
                continue
            ux = ((A[0]**2)*(B[1]*C[2]-C[1]*B[2]) - (B[0]**2)*(A[1]*C[2]-C[1]*A[2]) + (C[0]**2)*(A[1]*B[2]-B[1]*A[2])) / d
            uy = ((A[1]**2)*(B[2]*C[0]-C[2]*B[0]) - (B[1]**2)*(A[2]*C[0]-C[2]*A[0]) + (C[1]**2)*(A[2]*B[0]-B[2]*A[0])) / d
            uz = ((A[2]**2)*(B[0]*C[1]-C[0]*B[1]) - (B[2]**2)*(A[0]*C[1]-C[0]*A[1]) + (C[2]**2)*(A[0]*B[1]-B[0]*A[1])) / d
            circum = np.array([ux, uy, uz])
            norm_circum = np.linalg.norm(circum)
            if norm_circum < 1e-15:
                continue
            circum = circum / norm_circum
            # Angular distance from circumcenter to any vertex
            dot = np.dot(circum, A)
            dot = np.clip(dot, -1.0, 1.0)
            radius = np.arccos(dot)
            max_radius = max(max_radius, radius)
        return np.degrees(max_radius)
    except Exception:
        return 180.0

def _smooth_covering_radius(points, test_points, temperature=0.05):
    """Compute a smooth approximation of the covering radius using soft-max."""
    # points: (N, 3)
    # test_points: (M, 3)
    # For each test point, find distance to nearest config point
    # Then take soft-max over all test points
    
    # Compute all pairwise dot products: (M, N)
    dots = test_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Distance (angle) = arccos(dot)
    # But arccos is expensive. Use 1 - dot as proxy for small angles, or use log-sum-exp on exp(-dot/temperature)
    # For soft-min of distances: we want min_i arccos(dot_mi)
    # soft_min = -temperature * log(sum_i exp(-arccos(dot_mi)/temperature))
    # But arccos is non-differentiable at endpoints. Use 1-dot as smooth proxy for small angles.
    # Better: use the fact that for angles < ~30 deg, 1-cos(theta) ~ theta^2/2
    # Let's use: distance proxy = 1 - dot, and soft-max of these
    
    # Actually, let's just compute the actual angles for accuracy
    angles = np.arccos(dots)  # (M, N)
    
    # For each test point, find min angle
    min_angles = np.min(angles, axis=1)  # (M,)
    
    # Soft-max of min_angles: approximate max with log-sum-exp
    # soft_max(x) = temperature * log(sum(exp(x_i/temperature)))
    scaled = min_angles / temperature
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    soft_max = temperature * (max_scaled + np.log(sum_exp))
    
    return np.degrees(soft_max)

def _objective_and_grad(flat_params, test_points, temperature=0.05):
    """Objective function and gradient for L-BFGS-B."""
    points = flat_params.reshape(-1, 3)
    points = _normalize(points)
    
    # Compute objective
    dots = test_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    min_angles = np.min(angles, axis=1)
    
    # Soft-max
    scaled = min_angles / temperature
    max_scaled = np.max(scaled)
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    obj = np.degrees(temperature * (max_scaled + np.log(sum_exp)))
    
    # Gradient: d(obj)/d(points)
    # obj = degrees * temperature * (max_scaled + log(sum_exp))
    # Let's compute gradient numerically or analytically
    # For now, use numerical gradient via finite differences if needed
    # But L-BFGS-B can use finite differences if we don't provide jac
    # Let's provide an analytical gradient
    
    # d(soft_max)/d(min_angles) = exp(scaled) / sum_exp
    weights = np.exp(scaled - max_scaled) / sum_exp  # (M,)
    
    # d(min_angles)/d(points): for each test point m, min over n of arccos(dot_mn)
    # The min is achieved at some n*, and d(arccos(dot))/d(dot) = -1/sqrt(1-dot^2)
    # d(dot_mn)/d(points[n]) = test_points[m]
    
    # For each test point, find argmin
    argmin_n = np.argmin(angles, axis=1)  # (M,)
    
    # Gradient accumulation
    grad_points = np.zeros_like(points)
    for m in range(len(test_points)):
        n = argmin_n[m]
        dot_val = dots[m, n]
        # d(arccos(dot))/d(dot) = -1/sqrt(1-dot^2)
        if abs(1.0 - dot_val * dot_val) < 1e-15:
            continue
        dadot = -1.0 / np.sqrt(1.0 - dot_val * dot_val)
        # d(dot)/d(points[n]) = test_points[m]
        # Chain rule: d(obj)/d(points[n]) += degrees * temperature * weights[m] * dadot * test_points[m]
        coeff = np.degrees(1.0) * temperature * weights[m] * dadot
        grad_points[n] += coeff * test_points[m]
    
    # Normalize gradient: since we normalize points, we need to project gradient
    # The constraint is ||p_i|| = 1, so gradient should be orthogonal to p_i
    for i in range(len(points)):
        proj = np.dot(grad_points[i], points[i])
        grad_points[i] -= proj * points[i]
    
    return obj, grad_points.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    
    # Generate test points on sphere for smooth surrogate
    # Use a denser Fibonacci spiral for test points
    n_test = 2000
    test_points = _fibonacci_spiral(n_test)
    
    # Initial evaluation
    init_radius = _smooth_covering_radius(points, test_points, temperature=0.05)
    
    # Optimize using L-BFGS-B
    x0 = points.flatten()
    
    def obj_wrapper(x):
        obj, _ = _objective_and_grad(x, test_points, temperature=0.05)
        return obj
    
    def grad_wrapper(x):
        _, grad = _objective_and_grad(x, test_points, temperature=0.05)
        return grad
    
    best_points = points.copy()
    best_radius = init_radius
    
    try:
        result = minimize(
            obj_wrapper,
            x0,
            jac=grad_wrapper,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        
        if result.success or result.fun < best_radius:
            best_points = _normalize(result.x.reshape(-1, 3))
            best_radius = result.fun
        
        # Check time
        if time.time() > deadline - 20.0:
            return best_points
        
        # Try with lower temperature for more accurate max
        for temp in [0.02, 0.01]:
            if time.time() > deadline - 15.0:
                break
            def obj_wrapper2(x):
                obj, _ = _objective_and_grad(x, test_points, temperature=temp)
                return obj
            def grad_wrapper2(x):
                _, grad = _objective_and_grad(x, test_points, temperature=temp)
                return grad
            
            result2 = minimize(
                obj_wrapper2,
                best_points.flatten(),
                jac=grad_wrapper2,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if result2.success or result2.fun < best_radius:
                best_points = _normalize(result2.x.reshape(-1, 3))
                best_radius = result2.fun
        
        # Final polish: use exact hull-based radius for evaluation
        # Do a few more iterations with exact evaluation
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            # Use exact covering radius as objective (non-smooth, so use Nelder-Mead or just keep)
            # Actually, let's just return the best we have
            break
        
    except Exception:
        pass
    
    # Ensure points are on unit sphere
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
