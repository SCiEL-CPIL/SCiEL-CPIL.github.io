"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize the smallest triangle area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Vertices of the equilateral triangle with area 1
# Actually, the task says "unit-area triangle" but the interface specifies (0,0), (1,0), (1/2, sqrt(3)/2)
# Let's check the area of that triangle: base=1, height=sqrt(3)/2, area = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4
# The task says "divides by its area", so the evaluator likely normalizes.
# But the interface says "n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
# So we work in this coordinate system. The area of this triangle is sqrt(3)/4.
# The "smallest area of a triangle formed by three points" will be in these coordinates.
# The evaluator measures the raw value. Let's assume it computes the min area directly.
# Or maybe it normalizes by the triangle area? "divides by its area" suggests the score is (min_area / triangle_area).
# Let's assume the evaluator computes min_area / (sqrt(3)/4).
# To be safe, let's maximize the min_area. The relative ordering is the same.

# Precompute combinations of indices
idx_combos = list(combinations(range(N), 3))

def triangle_area(p1, p2, p3):
    # p1, p2, p3 are 2D arrays
    # Area = 0.5 * |det(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])

def min_triangle_area(points):
    # points is (N, 2)
    min_a = float('inf')
    for i, j, k in idx_combos:
        a = triangle_area(points[i], points[j], points[k])
        if a < min_a:
            min_a = a
    return min_a

def is_inside_triangle(p, margin=0.0):
    # Triangle vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    
    # Check if point is inside using barycentric coordinates or cross products
    # Let's use the method of checking signs of cross products
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v1, v2)
    d2 = sign(p, v2, v3)
    d3 = sign(p, v3, v1)
    
    has_neg = (d1 < -margin) or (d2 < -margin) or (d3 < -margin)
    has_pos = (d1 > margin) or (d2 > margin) or (d3 > margin)
    
    return not (has_neg and has_pos)

def objective_and_grad(x):
    # x is (2*N,)
    points = x.reshape(N, 2)
    
    # Compute all triangle areas
    areas = np.zeros(len(idx_combos))
    for idx, (i, j, k) in enumerate(idx_combos):
        areas[idx] = triangle_area(points[i], points[j], points[k])
    
    min_a = np.min(areas)
    
    # Gradient with respect to min_a
    # This is tricky because min is not differentiable.
    # We can use a soft-min or just perturb.
    # For now, let's use a simple heuristic or finite differences.
    # Actually, for optimization, it's better to use a smooth surrogate like -log-sum-exp(-areas/T)
    # or simply maximize the minimum area using a subgradient method.
    
    # Let's return the negative min area for minimization
    return -min_a

def soft_min_objective(x, T=0.01):
    points = x.reshape(N, 2)
    areas = np.zeros(len(idx_combos))
    for idx, (i, j, k) in enumerate(idx_combos):
        areas[idx] = triangle_area(points[i], points[j], points[k])
    
    # Soft min: -log-sum-exp(-areas/T) / T
    # This is smooth and approximates min(areas) as T -> 0
    max_a = np.max(areas)
    exp_vals = np.exp(-(areas - max_a) / T)
    sum_exp = np.sum(exp_vals)
    soft_min = max_a - T * np.log(sum_exp)
    
    return -soft_min

def grad_soft_min(x, T=0.01):
    points = x.reshape(N, 2)
    areas = np.zeros(len(idx_combos))
    for idx, (i, j, k) in enumerate(idx_combos):
        areas[idx] = triangle_area(points[i], points[j], points[k])
    
    max_a = np.max(areas)
    exp_vals = np.exp(-(areas - max_a) / T)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad = np.zeros(2*N)
    
    for idx, (i, j, k) in enumerate(idx_combos):
        w = weights[idx]
        # Gradient of area with respect to points
        # area = 0.5 * |det(pj-pi, pk-pi)|
        # Let's assume the sign is positive (points in CCW order) for gradient purposes
        # If not, we need to handle the absolute value.
        # For a smooth approximation, we can just use the signed area and hope the points are well-ordered
        # Or use the gradient of the absolute value.
        
        # Let's compute the gradient of the signed area
        pi = points[i]
        pj = points[j]
        pk = points[k]
        
        v1 = pj - pi
        v2 = pk - pi
        det = v1[0]*v2[1] - v1[1]*v2[0]
        
        if det < 0:
            # Swap j and k to make it CCW? Or just flip the gradient sign
            # The gradient of |det| is sign(det) * grad(det)
            # So if det < 0, the gradient is -grad(det)
            sign_det = -1
        else:
            sign_det = 1
            
        # grad(area) w.r.t. pi, pj, pk
        # area = 0.5 * sign * ( (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x) )
        
        # d(area)/d(pi_x) = 0.5 * sign * ( -(pk_y - pi_y) + (pj_y - pi_y) ) = 0.5 * sign * (pj_y - pk_y)
        # d(area)/d(pi_y) = 0.5 * sign * ( -(pj_x - pi_x) + (pk_x - pi_x) ) = 0.5 * sign * (pk_x - pj_x)
        
        # d(area)/d(pj_x) = 0.5 * sign * (pk_y - pi_y)
        # d(area)/d(pj_y) = 0.5 * sign * ( -(pk_x - pi_x) )
        
        # d(area)/d(pk_x) = 0.5 * sign * ( -(pj_y - pi_y) )
        # d(area)/d(pk_y) = 0.5 * sign * (pj_x - pi_x)
        
        factor = 0.5 * w * sign_det
        
        grad[2*i] += factor * (points[j,1] - points[k,1])
        grad[2*i+1] += factor * (points[k,0] - points[j,0])
        
        grad[2*j] += factor * (points[k,1] - points[i,1])
        grad[2*j+1] += factor * (-(points[k,0] - points[i,0]))
        
        grad[2*k] += factor * (-(points[j,1] - points[i,1]))
        grad[2*k+1] += factor * (points[j,0] - points[i,0])
        
    return -grad  # Because we are minimizing -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: grid-like distribution
    # Let's try a few different initializations
    
    best_points = None
    best_score = -float('inf')
    
    # Triangle vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    
    # Helper to generate random points inside triangle
    def random_points(n, rng):
        pts = []
        for _ in range(n):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            # Barycentric coordinates
            p = u * v1 + v * v2 + (1 - u - v) * v3
            pts.append(p)
        return np.array(pts)
    
    import random
    rng = random.Random(42)
    
    # Try multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate initial points
        if restart == 0:
            # Symmetric initial guess
            # Place points in a hexagonal-like pattern
            # Center
            center = (v1 + v2 + v3) / 3
            pts = [center]
            
            # Add points around
            # This is heuristic, let's just use random for now
            pass
            
        pts = random_points(N, rng)
        
        # Optimize
        x0 = pts.flatten()
        
        # Use L-BFGS-B with bounds
        # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
        # But we need to ensure points are inside the triangle.
        # L-BFGS-B doesn't support arbitrary constraints easily.
        # We can use penalty method or project onto the triangle.
        
        # Let's use a penalty function for the triangle constraint
        def penalty(x):
            points = x.reshape(N, 2)
            pen = 0
            for p in points:
                if not is_inside_triangle(p, margin=0.01):
                    # Compute distance to boundary
                    # Simplified: just add a large penalty
                    pen += 100.0
            return pen
        
        def objective_with_penalty(x, T=0.01):
            return soft_min_objective(x, T) + penalty(x)
        
        # This might be slow. Let's try to use a simpler approach.
        # Instead of penalty, let's parameterize points in barycentric coordinates.
        # Each point p = u*v1 + v*v2 + w*v3, where u+v+w=1, u,v,w >= 0.
        # We can use u, v as variables, and w = 1-u-v.
        # Constraint: u >= 0, v >= 0, u+v <= 1.
        # This is a simple box constraint on (u, v) in [0, 1] x [0, 1] with u+v <= 1.
        # We can map (u, v) to the triangle.
        
        # Let's redefine the optimization in terms of (u, v) for each point.
        # x = [u1, v1, u2, v2, ..., uN, vN]
        # p_i = u_i * v1 + v_i * v2 + (1-u_i-v_i) * v3
        
        def uv_to_points(x):
            uvs = x.reshape(N, 2)
            u = uvs[:, 0]
            v = uvs[:, 1]
            w = 1 - u - v
            # p = u*v1 + v*v2 + w*v3
            # v1 = (0,0), v2 = (1,0), v3 = (0.5, sqrt(3)/2)
            px = u*0 + v*1 + w*0.5
            py = u*0 + v*0 + w*(np.sqrt(3)/2)
            return np.column_stack([px, py])
        
        def soft_min_objective_uv(x, T=0.01):
            points = uv_to_points(x)
            areas = np.zeros(len(idx_combos))
            for idx, (i, j, k) in enumerate(idx_combos):
                areas[idx] = triangle_area(points[i], points[j], points[k])
            max_a = np.max(areas)
            exp_vals = np.exp(-(areas - max_a) / T)
            sum_exp = np.sum(exp_vals)
            soft_min = max_a - T * np.log(sum_exp)
            return -soft_min
        
        # Bounds for u, v: 0 <= u, 0 <= v, u+v <= 1
        # L-BFGS-B supports box constraints. We can use u in [0, 1], v in [0, 1-u]
        # But the upper bound for v depends on u.
        # We can use a transformation: let v = (1-u) * s, where s in [0, 1].
        # Then u in [0, 1], s in [0, 1].
        # p = u*v1 + (1-u)*s*v2 + (1-u)*(1-s)*v3
        
        def us_to_points(x):
            us = x.reshape(N, 2)
            u = us[:, 0]
            s = us[:, 1]
            v = (1 - u) * s
            w = (1 - u) * (1 - s)
            px = u*0 + v*1 + w*0.5
            py = u*0 + v*0 + w*(np.sqrt(3)/2)
            return np.column_stack([px, py])
        
        def soft_min_objective_us(x, T=0.01):
            points = us_to_points(x)
            areas = np.zeros(len(idx_combos))
            for idx, (i, j, k) in enumerate(idx_combos):
                areas[idx] = triangle_area(points[i], points[j], points[k])
            max_a = np.max(areas)
            exp_vals = np.exp(-(areas - max_a) / T)
            sum_exp = np.sum(exp_vals)
            soft_min = max_a - T * np.log(sum_exp)
            return -soft_min
        
        # Initial x in (u, s) space
        # u in [0, 1], s in [0, 1]
        x0_us = np.random.uniform(0, 1, (2*N,))
        
        # Optimize
        try:
            res = minimize(soft_min_objective_us, x0_us, method='L-BFGS-B', 
                           bounds=[(0, 1)] * (2*N), options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success:
                best_points_candidate = us_to_points(res.x)
                score = -soft_min_objective_us(res.x, T=0.001) # Use smaller T for evaluation
                # Actually, we want the min area, not the soft min.
                min_a = min_triangle_area(best_points_candidate)
                if min_a > best_score:
                    best_score = min_a
                    best_points = best_points_candidate
                    
        except Exception as e:
            pass
            
        # Also try a few more random restarts with different T
        if restart % 5 == 0:
            T = 0.005
            x0_us = np.random.uniform(0, 1, (2*N,))
            try:
                res = minimize(soft_min_objective_us, x0_us, method='L-BFGS-B', 
                               bounds=[(0, 1)] * (2*N), options={'maxiter': 500, 'ftol': 1e-10},
                               args=(T,))
                if res.success:
                    best_points_candidate = us_to_points(res.x)
                    min_a = min_triangle_area(best_points_candidate)
                    if min_a > best_score:
                        best_score = min_a
                        best_points = best_points_candidate
            except Exception as e:
                pass

    if best_points is None:
        # Fallback: return random points
        pts = random_points(N, rng)
        return pts
    
    return best_points
