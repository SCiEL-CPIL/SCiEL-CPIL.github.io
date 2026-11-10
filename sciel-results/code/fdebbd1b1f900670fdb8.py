import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50

def _circumradius(u, v, w):
    """
    Compute the circumradius of the spherical triangle with vertices u, v, w (unit vectors).
    """
    # Normal to the plane
    n = np.cross(u, v)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        return 0.0 # Degenerate
    n = n / n_norm
    
    # Circumcenter direction Y
    # Y is parallel to (u-v) x (v-w)
    Y = np.cross(u - v, v - w)
    Y_norm = np.linalg.norm(Y)
    if Y_norm < 1e-10:
        return 0.0
    c = Y / Y_norm
    
    # Ensure c is on the same side of the plane as the triangle "center"
    # The triangle center is roughly (u+v+w)/|u+v+w|
    # Or simply check if c is in the hemisphere defined by n
    # The normal n points to one side. The circumcenter should be in the direction of the triangle from the origin?
    # Actually, the circumcenter is the point on the sphere.
    # There are two points equidistant from u,v,w on the sphere: c and -c.
    # The one that is the circumcenter of the "small" triangle is the one closer to the centroid.
    centroid = u + v + w
    if np.dot(c, centroid) < 0:
        c = -c
        
    cos_R = np.dot(c, u)
    # Clamp for numerical stability
    cos_R = np.clip(cos_R, -1.0, 1.0)
    return np.arccos(cos_R)

def _compute_circumradii(points):
    """
    Compute circumradii for all facets of the convex hull of points.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.array([])
    
    radii = []
    # hull.simplices contains the indices of the points forming each facet
    for simplex in hull.simplices:
        # For N=50 on a sphere, facets are almost always triangles
        if len(simplex) == 3:
            u, v, w = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            R = _circumradius(u, v, w)
            radii.append(R)
        else:
            # If a facet is not a triangle (e.g., coplanar points), we might need to triangulate
            # For now, skip or approximate. On a sphere, generic positions yield triangles.
            pass
            
    return np.array(radii)

def _objective_and_grad(x):
    """
    Objective: LSE of circumradii.
    x: array of shape (N, 3)
    """
    points = x.reshape(N, 3)
    # Normalize points to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points = points / norms
    
    radii = _compute_circumradii(points)
    if len(radii) == 0:
        return 1e6, np.zeros_like(x)
    
    # Annealing parameter beta
    # We can't easily anneal inside a black-box optimizer unless we use a callback or restart
    # Let's use a fixed beta that is large enough to approximate max but smooth enough for gradient
    beta = 50.0
    
    # LSE
    max_R = np.max(radii)
    shifted = radii - max_R
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-10:
        lse_val = max_R
    else:
        lse_val = max_R + np.log(sum_exp) / beta
        
    # Gradient
    # grad_LSE = (1/beta) * sum( exp(beta * R_i) * grad_R_i )
    # We need grad_R_i with respect to x.
    # This is complex. Let's use finite differences for the gradient?
    # Or just return the value and let scipy do numerical gradients?
    # scipy.optimize.minimize with method 'BFGS' uses numerical gradients if jac is None.
    # This will be slow but N=50 is small.
    
    return lse_val, None

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = np.inf
    
    # Evaluate initial score
    radii = _compute_circumradii(best_points)
    if len(radii) > 0:
        best_score = np.max(radii)
    
    # Optimization
    # We will run a few iterations of minimize with increasing beta
    # Or just one run with a moderate beta.
    
    x0 = points.flatten()
    
    def obj_wrapper(x):
        val, _ = _objective_and_grad(x)
        return val
        
    # Use L-BFGS-B with bounds? No, constraints are better.
    # But L-BFGS-B doesn't handle equality constraints well.
    # SLSQP does.
    
    # Let's try SLSQP
    constraints = []
    for i in range(N):
        constraints.append({
            'type': 'eq',
            'fun': lambda x, i=i: x[3*i]**2 + x[3*i+1]**2 + x[3*i+2]**2 - 1.0,
            'jac': lambda x, i=i: np.array([
                2*x[3*i], 2*x[3*i+1], 2*x[3*i+2]
            ])
        })
        
    # This creates 50 constraints. SLSQP might struggle.
    # Alternative: Parameterize by (theta, phi) and minimize on that space, then project.
    # But theta/phi has singularities.
    
    # Let's stick to Cartesian and use a custom gradient descent if SLSQP is too slow.
    # Given 150s, SLSQP on 150 vars with 50 constraints might be okay for a few hundred iterations.
    
    try:
        res = scipy.optimize.minimize(
            obj_wrapper, 
            x0, 
            method='SLSQP', 
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-4} # Limit iterations to stay within time
        )
        
        if res.success or res.fun < best_score:
            x_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            x_opt = x_opt / norms
            radii_opt = _compute_circumradii(x_opt)
            if len(radii_opt) > 0:
                score_opt = np.max(radii_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_points = x_opt
                    
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline:
        return best_points
        
    # If we have time, try a second pass with a different start or higher beta
    # For now, return the best found.
    
    return best_points
