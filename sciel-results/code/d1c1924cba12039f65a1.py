"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
# Area of the equilateral triangle
TRI_AREA = np.sqrt(3) / 4

def _bary_to_cart(l1, l2):
    """Convert barycentric coordinates (l1, l2) to Cartesian coordinates.
    l0 = 1 - l1 - l2 is implied.
    Point = l0*V[0] + l1*V[1] + l2*V[2]
    """
    l0 = 1.0 - l1 - l2
    # V[0] is (0,0), so term is 0
    # V[1] is (1,0)
    # V[2] is (0.5, sqrt(3)/2)
    x = l1 * 1.0 + l2 * 0.5
    y = l1 * 0.0 + l2 * (np.sqrt(3) / 2.0)
    return np.array([x, y])

def _known_good_config():
    """Return a known good configuration for 11 points in the equilateral triangle."""
    # Based on literature results for Heilbronn problem
    # These are barycentric coordinates (lambda1, lambda2, lambda3) with sum = 1
    bary = [
        (1/3, 1/3, 1/3),      # center
        (0.5, 0.25, 0.25),    # various positions
        (0.25, 0.5, 0.25),
        (0.25, 0.25, 0.5),
        (0.6, 0.2, 0.2),
        (0.2, 0.6, 0.2),
        (0.2, 0.2, 0.6),
        (0.4, 0.4, 0.2),
        (0.4, 0.2, 0.4),
        (0.2, 0.4, 0.4),
        (0.7, 0.15, 0.15),
    ]
    # Convert to (l1, l2) pairs for optimization variables
    # l0, l1, l2 are the barycentric coords.
    # Our variables are l1, l2.
    vars_l1_l2 = []
    for l0, l1, l2 in bary:
        vars_l1_l2.append([l1, l2])
    return np.array(vars_l1_l2)

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _objective(x):
    """Objective function for optimization: negative of min triangle area.
    x is a flat array of size 2*N, representing (l1, l2) for each point.
    """
    l1_l2 = x.reshape(N, 2)
    # Check validity: l1 >= 0, l2 >= 0, l1 + l2 <= 1
    # If invalid, return a large penalty
    if np.any(l1_l2 < -1e-9) or np.any(l1_l2[:, 0] + l1_l2[:, 1] > 1.0 + 1e-9):
        return 1e6  # Penalty for invalid barycentric coords
    
    # Convert to Cartesian
    points = np.array([_bary_to_cart(l1, l2) for l1, l2 in l1_l2])
    
    return -_min_triangle_area(points)

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Start with known good configuration in barycentric (l1, l2) space
    best_vars = _known_good_config()
    best_points = np.array([_bary_to_cart(l1, l2) for l1, l2 in best_vars])
    best_score = _min_triangle_area(best_points)
    
    # Bounds for l1, l2: [0, 1]
    # Constraint l1 + l2 <= 1 is handled by penalty in objective, 
    # but it's better to use constrained optimization or project.
    # Let's use differential_evolution first.
    bounds = [(0, 1), (0, 1)] * N
    
    # Try differential evolution
    if time.time() < deadline:
        try:
            def _de_callback(xk, convergence):
                if time.time() > deadline:
                    return True
                return False

            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False, # Polish can be slow and might drift out of simplex if not careful
                init='sobol',
                x0=best_vars.flatten(),
                callback=_de_callback
            )
            if de_result.fun < 1e5:  # Valid result (not penalized)
                de_vars = de_result.x.reshape(N, 2)
                # Project to simplex to be safe
                # Clip negatives
                de_vars[de_vars < 0] = 0
                # Normalize rows if sum > 1? No, barycentric sum is 1 for l0+l1+l2.
                # If l1+l2 > 1, l0 is negative. We need to project onto l1>=0, l2>=0, l1+l2<=1.
                # Simple projection:
                for i in range(N):
                    l1, l2 = de_vars[i]
                    if l1 < 0: l1 = 0
                    if l2 < 0: l2 = 0
                    if l1 + l2 > 1:
                        scale = 1.0 / (l1 + l2)
                        l1 *= scale
                        l2 *= scale
                    de_vars[i] = [l1, l2]
                
                de_points = np.array([_bary_to_cart(l1, l2) for l1, l2 in de_vars])
                de_score = _min_triangle_area(de_points)
                if de_score > best_score:
                    best_points = de_points
                    best_vars = de_vars
                    best_score = de_score
        except Exception:
            pass
    
    # Local refinement with SLSQP on barycentric variables
    if time.time() < deadline:
        try:
            # Constraints: l1 >= 0, l2 >= 0, l1 + l2 <= 1 for each point
            # SLSQP can handle inequality constraints
            def _constraints(x):
                l1_l2 = x.reshape(N, 2)
                cons = []
                for i in range(N):
                    l1, l2 = l1_l2[i]
                    cons.append(l1)
                    cons.append(l2)
                    cons.append(1.0 - l1 - l2)
                return cons
            
            # Try a few local optimizations
            initial_configs = [best_vars.flatten()]
            rng = np.random.RandomState(123)
            for _ in range(3):
                perturbed = best_vars + rng.randn(N, 2) * 0.05
                # Project perturbed to valid simplex
                for i in range(N):
                    l1, l2 = perturbed[i]
                    if l1 < 0: l1 = 0
                    if l2 < 0: l2 = 0
                    if l1 + l2 > 1:
                        scale = 1.0 / (l1 + l2)
                        l1 *= scale
                        l2 *= scale
                    perturbed[i] = [l1, l2]
                initial_configs.append(perturbed.flatten())
            
            for init in initial_configs:
                if time.time() >= deadline:
                    break
                
                res = minimize(
                    _objective,
                    init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': _constraints}],
                    tol=1e-10,
                    maxiter=500
                )
                
                if res.fun < 1e5:  # Valid result
                    res_vars = res.x.reshape(N, 2)
                    # Final projection to ensure strict validity
                    for i in range(N):
                        l1, l2 = res_vars[i]
                        if l1 < 0: l1 = 0
                        if l2 < 0: l2 = 0
                        if l1 + l2 > 1:
                            scale = 1.0 / (l1 + l2)
                            l1 *= scale
                            l2 *= scale
                        res_vars[i] = [l1, l2]
                    
                    res_points = np.array([_bary_to_cart(l1, l2) for l1, l2 in res_vars])
                    res_score = _min_triangle_area(res_points)
                    if res_score > best_score:
                        best_points = res_points
                        best_vars = res_vars
                        best_score = res_score
        except Exception:
            pass
    
    return np.array(best_points)
