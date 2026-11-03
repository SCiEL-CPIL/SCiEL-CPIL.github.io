"""
Strategy: Direct Global Optimization using Differential Evolution.
Instead of iterative Lloyd's algorithm which can get stuck in local minima 
or converge slowly, we treat the positions of the 50 points as a 100-dimensional 
optimization problem (theta, phi for each point). We use scipy.optimize.differential_evolution 
which is robust for global minimization of non-smooth, non-convex functions. 
The objective function is the covering radius, calculated exactly via the convex hull.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 50
_START_TIME = time.time()
# Use 145s as hard deadline to ensure return within 150s limit
_DEADLINE = _START_TIME + 145.0

def _compute_covering_radius(points):
    """
    Compute the covering radius of points on the unit sphere.
    Returns the radius in radians.
    """
    if len(points) < 3:
        return np.pi
        
    # Ensure points are on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = points / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # If hull fails (e.g., points are coplanar or degenerate), 
        # it's a bad configuration. Return a large penalty.
        return np.pi

    max_radius = 0.0
    # Iterate over all facets (simplices)
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute normal to the plane containing a, b, c
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
            
        n = cross / norm_cross
        
        # Ensure normal points outward (away from origin)
        # The origin is at (0,0,0). The plane equation is n . x = d.
        # Since a is on the plane, d = n . a.
        # For the hull of points on the unit sphere, the origin is inside.
        # The outward normal should satisfy n . a > 0? 
        # Actually, the circumcenter of the spherical triangle is the point on the sphere 
        # furthest from the triangle vertices in the direction of the normal.
        # The angular distance R satisfies cos(R) = n . a (if n is normalized and points to circumcenter).
        # Wait, the circumcenter C is the normalized normal vector n (or -n).
        # The distance from C to a is arccos(C . a).
        
        # Check which direction is the circumcenter.
        # The circumcenter is the pole of the great circle passing through a,b,c? No, 
        # it's the pole of the plane containing a,b,c only if the triangle is spherical right? 
        # No. The circumcenter of a spherical triangle is the intersection of the sphere 
        # and the line passing through the origin and the circumcenter of the planar triangle? 
        # No, that's not right either.
        
        # Correct Geometry:
        # The vertices a,b,c are on the unit sphere.
        # The plane containing them is n . x = d, where n is the unit normal.
        # d = n . a.
        # The circumcenter of the spherical triangle is the point C on the sphere 
        # such that C is equidistant from a, b, c.
        # By symmetry, C lies on the line defined by the origin and the vector n? 
        # Yes, the locus of points equidistant from a and b is the plane bisecting them, 
        # which passes through the origin and is perpendicular to a-b.
        # The intersection of these bisector planes is the line through the origin 
        # and the normal vector n of the plane abc.
        # So C is either n or -n.
        # We choose the one that is "outside" the triangle relative to the origin?
        # Actually, for a convex hull facet, the origin is on one side.
        # The circumcenter of the facet (cap) is the point on the sphere in the direction 
        # of the outward normal of the facet.
        
        # Let's verify: 
        # If n . a > 0, then n is in the same hemisphere as a.
        # The angular distance from n to a is arccos(n . a).
        # Is this the covering radius for this facet?
        # Yes, the "empty cap" condition implies the cap centered at the circumcenter 
        # with radius R contains no other points. The circumcenter is the center of the 
        # largest empty cap.
        
        # We need to ensure n points away from the origin relative to the facet?
        # The origin is at 0. The plane is n.x = d.
        # If we pick the normal n such that n.a > 0, then d > 0.
        # The point on the sphere in direction n is C = n.
        # Distance from C to a: arccos(n.a).
        
        d = np.dot(n, a)
        if d < 0:
            n = -n
            d = -d
            
        d = np.clip(d, -1.0, 1.0)
        r = np.arccos(d)
        
        if r > max_radius:
            max_radius = r
            
    return max_radius

def _objective(params):
    """
    Objective function for differential_evolution.
    params: array of shape (2*N,) containing [theta_0, phi_0, theta_1, phi_1, ...]
    theta in [0, pi], phi in [0, 2*pi]
    """
    try:
        thetas = params[0::2]
        phis = params[1::2]
        
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        
        points = np.column_stack([x, y, z])
        
        return _compute_covering_radius(points)
    except Exception:
        return np.pi

def _random_initial_points():
    """Generate random points on the sphere for potential seeding or fallback."""
    u = np.random.rand(N)
    v = np.random.rand(N)
    theta = np.arccos(2 * u - 1)
    phi = 2 * np.pi * v
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    np.random.seed(42)
    
    # Bounds for theta (0 to pi) and phi (0 to 2*pi)
    bounds = []
    for _ in range(N):
        bounds.append((0, np.pi))  # theta
        bounds.append((0, 2 * np.pi)) # phi
    
    # We use differential_evolution.
    # It is a global optimizer.
    # We set maxiter high but rely on the timeout.
    # popsize=15 is a good balance.
    # tol=1e-5 for convergence.
    
    # To ensure we get a valid result even if DE is slow, 
    # we can run it and check time.
    
    # Since DE can be slow for 100 dimensions, we might need to be careful.
    # However, 50 points is manageable.
    
    # Let's try to run it.
    # Note: DE does not have a direct time limit, but we can monitor it?
    # Actually, we can't easily interrupt DE mid-run in a simple script without threading.
    # Instead, we can run it with a reasonable maxiter and see if it finishes.
    # If not, we fallback to a simpler method or return the best found so far?
    # DE doesn't expose intermediate results easily in the standard call.
    
    # Alternative: Use `scipy.optimize.minimize` with `method='Powell'` or `Nelder-Mead` 
    # from a good random start? No, Powell is also local.
    
    # Let's stick to DE but with a conservative maxiter.
    # 100 dims * 15 popsize * maxiter evaluations.
    # Each evaluation is ~1-5ms (ConvexHull is fast for N=50).
    # 100 * 15 * 100 = 150,000 evals. At 5ms each, that's 750s. Too slow.
    
    # We need a faster approach or fewer evaluations.
    # Let's reduce popsize to 10 and maxiter to 50?
    # 100 * 10 * 50 = 50,000 evals. 250s. Still too slow.
    
    # We need to speed up the objective or reduce search space.
    # Or use a different global optimizer.
    
    # What about `scipy.optimize.differential_evolution` with `workers=-1`? 
    # If parallelism is available, it might help, but we don't know the environment.
    
    # Let's try a hybrid: 
    # 1. Run a few random restarts of a local optimizer (like Nelder-Mead) from random starts.
    # 2. Or, run DE with very low popsize and maxiter, hoping to find a good basin, 
    #    then refine with local search.
    
    # Let's try DE with popsize=5, maxiter=20.
    # 100 * 5 * 20 = 10,000 evals. ~50s. This is safe.
    # It might not find the global optimum, but it should find a decent local optimum.
    # Then we can refine that result with a local search (e.g., Nelder-Mead or gradient descent).
    
    best_params = None
    best_radius = np.pi
    
    # Run DE
    start_time = time.time()
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=20, 
            popsize=5, 
            tol=1e-4, 
            seed=42, 
            polish=False  # Disable polish to save time, we will do our own refinement
        )
        
        if result.success or result.fun < best_radius:
            best_params = result.x
            best_radius = result.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE didn't run or failed, use a random start
    if best_params is None:
        thetas = np.random.rand(N) * np.pi
        phis = np.random.rand(N) * 2 * np.pi
        best_params = np.zeros(2 * N)
        best_params[0::2] = thetas
        best_params[1::2] = phis
        best_radius = _objective(best_params)
        
    # Now, refine the best solution found by DE using a local optimizer.
    # Nelder-Mead is robust for non-smooth functions.
    # We have ~100s left.
    
    # Convert best_params to points
    thetas = best_params[0::2]
    phis = best_params[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    current_points = np.column_stack([x, y, z])
    
    # Refine using Nelder-Mead on the points directly?
    # Nelder-Mead works on parameters. Let's stick to theta/phi params.
    
    # We will run Nelder-Mead with a time limit check inside the objective?
    # No, Nelder-Mead doesn't support that easily.
    # We can run it with a maxiter limit.
    
    # Let's estimate iterations. 100 dims. Nelder-Mead takes ~10*N iterations per step?
    # It's slow in high dimensions.
    
    # Better: Use `scipy.optimize.minimize` with `method='Powell'`.
    # Powell is derivative-free and often faster than NM in high dims.
    
    from scipy.optimize import minimize
    
    try:
        res = minimize(
            _objective, 
            best_params, 
            method='Powell', 
            options={'maxiter': 50, 'maxfev': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if res.fun < best_radius:
            best_params = res.x
            best_radius = res.fun
            
    except Exception as e:
        print(f"Powell failed: {e}")
        
    # Check time
    if time.time() > _DEADLINE - 5:
        # Return current best
        pass
    else:
        # Try one more refinement with Nelder-Mead if time permits
        # Or just return.
        pass

    # Convert final params to points
    thetas = best_params[0::2]
    phis = best_params[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    points = np.column_stack([x, y, z])
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Validation: Check for coincident points
    # If any two points are too close, perturb them
    for i in range(N):
        for j in range(i+1, N):
            if np.dot(points[i], points[j]) > 0.99999:
                # Perturb j
                noise = np.random.randn(3) * 0.01
                points[j] = points[j] + noise
                points[j] = points[j] / np.linalg.norm(points[j])
                
    return points
