import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for validity: finite, non-collinear
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    # Vectorized area calculation for all triangles
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the Heilbronn ratio.
    Strategy: Direct L-BFGS-B Optimization with Smoothed Objective.
    
    1. Parameterize the 13 points as a 26-dimensional vector.
    2. Define a smooth objective function:
       f(x) = - (SoftMin_Area / Hull_Area)
       where SoftMin_Area is a log-sum-exp approximation of the minimum triangle area.
       The log-sum-exp function is smooth and differentiable, allowing gradient-based methods.
       As the temperature parameter 'k' increases, the soft-min approaches the true min.
    3. Use L-BFGS-B optimizer which is efficient for smooth functions with many parameters.
    4. Initialize with a regular 13-gon.
    5. Run optimization with time guards.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Pre-compute triangle indices for speed
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i1, i2, i3 = idx[:, 0], idx[:, 1], idx[:, 2]
    n_tri = len(idx)
    
    def get_areas_vectorized(pts_flat):
        """
        Compute areas of all triangles from a flat vector of points.
        pts_flat: shape (2*n,)
        Returns: array of shape (n_tri,)
        """
        pts = pts_flat.reshape(n, 2)
        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]
        
        v1 = p2 - p1
        v2 = p3 - p1
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(det)
    
    def get_hull_area(pts_flat):
        """Calculate convex hull area from flat vector."""
        pts = pts_flat.reshape(n, 2)
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def smooth_objective(pts_flat, k=100.0):
        """
        Objective function to minimize: - (SoftMinArea / HullArea)
        SoftMinArea = -1/k * log(sum(exp(-k * area)))
        This approximates min(area) as k -> infinity.
        """
        areas = get_areas_vectorized(pts_flat)
        
        # Numerical stability for log-sum-exp
        # We want min(area). Let m = min(areas).
        # log(sum(exp(-k*areas))) = -k*m + log(sum(exp(-k*(areas-m))))
        # SoftMin = m - (1/k) * log(sum(exp(-k*(areas-m))))
        
        m = np.min(areas)
        if m < 1e-12:
            # If min area is tiny, penalty is huge
            return 1e10 
            
        # Subtract m to prevent underflow/overflow in exp
        shifted = areas - m
        # exp(-k * shifted) is stable since shifted >= 0
        exp_terms = np.exp(-k * shifted)
        sum_exp = np.sum(exp_terms)
        
        # SoftMin = m - (1/k) * log(sum_exp)
        soft_min_area = m - (1.0 / k) * np.log(sum_exp)
        
        hull_area = get_hull_area(pts_flat)
        if hull_area < 1e-12:
            return 1e10
        
        ratio = soft_min_area / hull_area
        # We want to maximize ratio, so minimize -ratio
        return -ratio
    
    def smooth_gradient(pts_flat, k=100.0):
        """
        Analytical gradient of smooth_objective.
        f = - (S / H)
        df/dx = - ( (dS/dx) * H - S * (dH/dx) ) / H^2
        = (S * dH/dx - H * dS/dx) / H^2
        
        S = m - (1/k) * log(sum(exp(-k*(areas-m))))
        Let L = log(sum(exp(-k*(areas-m))))
        S = m - L/k
        
        dS/dx = dm/dx - (1/k) * dL/dx
        
        dL/dx = (1/sum_exp) * d(sum_exp)/dx
        sum_exp = sum_j exp(-k*(A_j - m))
        d(sum_exp)/dx = sum_j exp(-k*(A_j - m)) * (-k) * dA_j/dx
        
        Note: dm/dx is tricky because m is the min. However, in the log-sum-exp formulation,
        the gradient of the soft-min is a weighted average of the gradients of the areas,
        where weights are proportional to exp(-k*(A_j - m)).
        
        Specifically, gradient of SoftMin = sum_j w_j * grad(A_j), where w_j = exp(-k*(A_j-m)) / sum_exp.
        Note that sum(w_j) = 1.
        
        So dS/dx = sum_j w_j * dA_j/dx.
        
        Now we need dA_j/dx and dH/dx.
        
        Area A_j = 0.5 * |det(v1, v2)|.
        Let det_j = v1_x * v2_y - v1_y * v2_x.
        A_j = 0.5 * |det_j|.
        dA_j/dx = 0.5 * sign(det_j) * d(det_j)/dx.
        
        det_j depends on points p1, p2, p3.
        det = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
        = x2y3 - x2y1 - x1y3 + x1y1 - y2x3 + y2x1 + y1x3 - y1x1
        = x2y3 - x2y1 - x1y3 - y2x3 + y2x1 + y1x3
        
        Partial derivatives:
        d(det)/dx1 = -y3 + y2
        d(det)/dy1 = x3 - x2
        d(det)/dx2 = y3 - y1
        d(det)/dy2 = -(x3 - x1) = x1 - x3
        d(det)/dx3 = y1 - y2
        d(det)/dy3 = x2 - x1
        
        dH/dx is the gradient of the convex hull area.
        The area of a convex polygon with vertices (x_i, y_i) in CCW order is:
        H = 0.5 * sum_i (x_i y_{i+1} - x_{i+1} y_i)
        dH/dx_i = 0.5 * (y_{i-1} - y_{i+1})
        dH/dy_i = 0.5 * (x_{i+1} - x_{i-1})
        
        This requires knowing the hull vertices and their order.
        Calculating dH/dx analytically is complex because the set of hull vertices and their order
        can change discontinuously.
        
        Alternative: Use numerical gradient for the hull part? Or approximate?
        Actually, L-BFGS-B can use numerical gradients if we don't provide them, but it's slower.
        However, for 26 dimensions, numerical gradient (26 * 2 evaluations) per step might be okay
        if the number of steps is small. But L-BFGS-B with numerical gradient is essentially
        a quasi-Newton method with finite differences.
        
        Let's try providing only the smooth part gradient and letting the optimizer handle the rest?
        No, the objective is the ratio.
        
        Let's simplify: Since the hull area is scale-invariant in the ratio, and we are optimizing
        the shape, maybe we can fix the hull area?
        No, the hull shape matters.
        
        Given the complexity of the hull gradient, let's use **numerical gradients** for the full objective.
        L-BFGS-B with numerical gradients in 26D is feasible within the time limit if we limit iterations.
        
        Wait, the previous "inspiration 2" used Nelder-Mead. L-BFGS-B with numerical gradients
        requires O(N) function evaluations per step. N=26.
        If we do 100 steps, that's 2600 evaluations.
        Each evaluation involves:
        1. Reshape (fast)
        2. Compute 286 areas (fast vectorized)
        3. Compute Hull (scipy ConvexHull is the bottleneck).
        
        ConvexHull on 13 points is fast (microseconds).
        So 2600 * (286 ops + hull) is very fast.
        
        Let's implement the objective without analytical gradient and let scipy use finite differences.
        This is robust and avoids the hull gradient complexity.
        """
        pass # Not implementing analytical gradient to keep code simple and robust against hull topology changes.
        # scipy.optimize.minimize with method='L-BFGS-B' will use numerical gradients.
    
    # Initialize with a regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    init_flat = init_points.flatten()
    
    best_points = init_points.copy()
    best_score = compute_score(best_points)
    
    # Optimization parameters
    # k is the temperature for the soft-min. Higher k -> closer to true min.
    # Start with moderate k and increase? Or just use a high k?
    # If k is too high, the function is non-smooth (like the true min).
    # If k is too low, it optimizes the average area, not the min.
    # A value around 50-100 is often good for 13 points.
    
    k = 80.0
    
    # Run optimization
    # We might need to restart if it gets stuck.
    # L-BFGS-B is local.
    
    current_flat = init_flat.copy()
    
    # We will run L-BFGS-B.
    # To handle the non-convexity and potential local minima, we can do a few restarts
    # with slightly perturbed initial conditions if time permits.
    
    n_restarts = 3
    max_iter_per_restart = 200
    
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        # Perturb initial points for restarts > 0
        if r == 0:
            start_flat = init_flat.copy()
        else:
            rng = np.random.default_rng(42 + r)
            noise = rng.normal(0, 0.1, size=2*n)
            start_flat = init_flat + noise
            # Re-center
            pts = start_flat.reshape(n, 2)
            pts -= np.mean(pts, axis=0)
            start_flat = pts.flatten()
        
        # Define objective for this restart
        def obj(x):
            return smooth_objective(x, k=k)
        
        try:
            res = minimize(obj, start_flat, method='L-BFGS-B', 
                           options={'maxiter': max_iter_per_restart, 'ftol': 1e-9, 'gtol': 1e-6, 'maxfun': 5000})
            
            if res.success or res.fun < 0: # If it found a solution
                opt_flat = res.x
                opt_points = opt_flat.reshape(n, 2)
                
                # Check validity
                if np.all(np.isfinite(opt_points)):
                    score = compute_score(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
        except Exception as e:
            # If optimization fails, continue
            pass
            
    return best_points

# EVOLVE-BLOCK-END
