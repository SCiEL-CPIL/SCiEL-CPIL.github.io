# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def _all_triangle_areas(points):
    """
    Compute areas of all C(n,3) triangles formed by points.
    Returns array of shape (C(n,3),)
    """
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

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
        
    areas = _all_triangle_areas(points)
    A_min = np.min(areas)
    
    return A_min / A_hull

def _smooth_objective(x):
    """
    Smooth surrogate objective for L-BFGS-B.
    Minimizes -log(A_min) + log(A_hull).
    Equivalent to maximizing A_min / A_hull.
    """
    n = 13
    points = x.reshape((n, 2))
    
    if not np.all(np.isfinite(points)):
        return 1e6, np.zeros_like(x)
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e6, np.zeros_like(x)
        
    if A_hull <= 1e-12:
        return 1e6, np.zeros_like(x)
        
    # Compute areas and identify the minimum
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    abs_det = np.abs(det)
    
    min_idx = np.argmin(abs_det)
    min_abs_det = abs_det[min_idx]
    
    # Avoid log(0)
    eps = 1e-12
    min_abs_det = max(min_abs_det, eps)
    
    val = -np.log(min_abs_det) + np.log(A_hull)
    
    # Gradient
    # d/dx ( -log(min_abs_det) + log(A_hull) )
    # = - (1/min_abs_det) * d(min_abs_det)/dx + (1/A_hull) * d(A_hull)/dx
    
    # Gradient of min_abs_det w.r.t. points
    # min_abs_det is determined by triangle i, j, k.
    # det = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # sign = sign(det)
    # d(abs_det)/dp_i = -sign * (p_j - p_k)  (rotated 90 deg? Let's derive carefully)
    # Let's use the cross product form. In 2D, cross(u, v) = u_x v_y - u_y v_x.
    # d/dp1 ( |cross(p2-p1, p3-p1)| )
    # Let u = p2-p1, v = p3-p1.
    # d/dp1 cross(u, v) = -cross(p2, v) - cross(u, p3) ? No.
    # cross(p2-p1, p3-p1) = cross(p2, p3) - cross(p2, p1) - cross(p1, p3) + cross(p1, p1)
    # = cross(p2, p3) - cross(p2, p1) - cross(p1, p3)
    # d/dp1: -(-cross(1, p2)) - (-cross(1, p3)) ?
    # Let's use the vector derivative:
    # d/dp ( |cross(p2-p1, p3-p1)| )
    # If det > 0:
    # d/dp1 (det) = - (p2 - p3)_y, (p2 - p3)_x  -> -R90(p2-p3) ?
    # Let's check: det = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # d/dx1: -(y3-y1) - (y2-y1)(-1) = -(y3-y1) + (y2-y1) = y2 - y3
    # d/dy1: -(x2-x1) - (x3-x1)(-1) = -(x2-x1) + (x3-x1) = x3 - x2
    # So d/dp1 (det) = [y2-y3, x3-x2]
    # Note that cross(p2-p3, [1,0]) = (y2-y3)*0 - (x2-x3)*1 = x3-x2.
    # cross([0,1], p2-p3) = 0*(y2-y3) - 1*(x2-x3) = x3-x2.
    # Actually, standard rotation: R90(v) = [-v_y, v_x].
    # R90(p2-p3) = [-(y2-y3), x2-x3].
    # My derivative is [y2-y3, x3-x2] = -R90(p2-p3).
    
    # d/dp2 (det) = (y3-y1) - 0 = y3-y1?
    # d/dx2: (y3-y1). d/dy2: -(x3-x1).
    # So d/dp2 (det) = [y3-y1, -(x3-x1)] = R90(p3-p1)?
    # R90(p3-p1) = [-(y3-y1), x3-x1]. No, it's -R90(p3-p1).
    # Wait. R90(v) = [-v_y, v_x].
    # d/dp2 = [y3-y1, x1-x3].
    # R90(p1-p3) = [-(y1-y3), x1-x3] = [y3-y1, x1-x3]. Yes.
    # So d/dp2 (det) = R90(p1-p3).
    
    # d/dp3 (det) = 0 - (-(y2-y1)) = y2-y1?
    # d/dx3: -(y2-y1). d/dy3: (x2-x1).
    # d/dp3 = [-(y2-y1), x2-x1] = R90(p2-p1).
    
    # So for det > 0:
    # grad_p1 = -R90(p2-p3)
    # grad_p2 = R90(p1-p3)
    # grad_p3 = R90(p2-p1)
    # Sum is 0? -R90(p2-p3) + R90(p1-p3) + R90(p2-p1) = R90(-p2+p3 + p1-p3 + p2-p1) = 0. Correct.
    
    # For det < 0, abs_det = -det, so gradients are negated.
    
    sign = 1.0 if det[min_idx] > 0 else -1.0
    
    i, j, k = idx[min_idx]
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    # R90(v) = [-v[1], v[0]]
    def r90(v):
        return np.array([-v[1], v[0]])
    
    grad_abs_det = np.zeros_like(points)
    
    if sign > 0:
        grad_abs_det[i] -= r90(pj - pk)
        grad_abs_det[j] += r90(pi - pk)
        grad_abs_det[k] += r90(pj - pi)
    else:
        # d/dp (-det) = - d/dp (det)
        grad_abs_det[i] += r90(pj - pk)
        grad_abs_det[j] -= r90(pi - pk)
        grad_abs_det[k] -= r90(pj - pi)
        
    grad_val = - (1.0 / min_abs_det) * grad_abs_det
    
    # Gradient of A_hull
    # A_hull is piecewise linear. The gradient is the sum of gradients of the active facets.
    # Facet (a, b) contributes 0.5 * cross(p_b - p_a, p) for points p inside?
    # Actually, Area = 0.5 * sum_{facets (a,b)} cross(p_a, p_b)
    # This is constant! The area of the convex hull is constant with respect to interior points.
    # Wait. If a point moves and changes the hull, the area changes.
    # If the point is interior, A_hull is constant locally, so grad A_hull = 0.
    # If the point is on the hull, A_hull is not differentiable (it's the max of linear functions).
    # For L-BFGS, we can approximate the gradient of A_hull.
    # Or, simpler: if we assume the hull topology doesn't change often, we can use the subgradient.
    # The subgradient of A_hull w.r.t. p_i (if i is on hull) involves the adjacent hull vertices.
    # However, a simpler heuristic: if the point is strictly interior, grad is 0.
    # If it's on the hull, we can use the normal to the hull edge.
    # Let's try to compute the subgradient for points on the hull.
    
    # Get hull vertices
    hull_vertices = hull.vertices
    # Map point index to hull vertex index?
    # ConvexHull.vertices gives indices into the input points.
    
    # For each point i:
    # If i is not in hull_vertices, grad is 0.
    # If i is in hull_vertices, it is part of the boundary.
    # The area is 0.5 * sum_{hull edges (u,v)} cross(p_u, p_v).
    # The term involving p_i appears in edges (prev, i) and (i, next).
    # Term 1: 0.5 * cross(p_prev, p_i). d/dp_i = 0.5 * R90(p_prev) ?
    # cross(u, v) = u_x v_y - u_y v_x.
    # d/dv (u_x v_y - u_y v_x) = [ -u_y, u_x ] = R90(u).
    # So d/dp_i (0.5 cross(p_prev, p_i)) = 0.5 * R90(p_prev).
    # Term 2: 0.5 * cross(p_i, p_next).
    # d/du (u_x v_y - u_y v_x) = [ v_y, -v_x ] = -R90(v).
    # So d/dp_i (0.5 cross(p_i, p_next)) = -0.5 * R90(p_next).
    
    # So grad A_hull at p_i = 0.5 * (R90(p_prev) - R90(p_next)).
    
    # We need to find prev and next for each hull vertex.
    # hull.vertices is in CCW order.
    
    hull_grad = np.zeros_like(points)
    for idx_h, v in enumerate(hull_vertices):
        prev_v = hull_vertices[(idx_h - 1) % len(hull_vertices)]
        next_v = hull_vertices[(idx_h + 1) % len(hull_vertices)]
        
        # Check if point v is strictly on the hull (it is, by definition)
        # But we should check if it's a "corner" or if the gradient is well-defined.
        # The subgradient is the average of the gradients of the active facets.
        # Here we are using the gradient of the area function assuming the hull topology is fixed.
        # This is valid if the point stays on the hull.
        
        hull_grad[v] = 0.5 * (r90(points[prev_v]) - r90(points[next_v]))
        
    grad_val += (1.0 / A_hull) * hull_grad
    
    return val, grad_val

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 580
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init_regular)
    
    # 2. Concentric polygons
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    init_concentric = np.vstack([center, inner_pts, outer_pts])
    inits.append(init_concentric)
    
    # 3. Random
    rng = np.random.default_rng(seed=42)
    init_random = rng.random((n, 2)) * 2 - 1
    inits.append(init_random)
    
    # 4. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # Optimize each
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init_pts.flatten()
        
        try:
            # L-BFGS-B with analytic gradient
            res = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            final_pts = res.x.reshape((n, 2))
            final_score = compute_score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
        except Exception:
            pass
        
        # If we have time, do a second pass with Nelder-Mead on the best found so far
        # to polish the non-smooth minimum
        if time.time() < deadline - 100 and best_points is not None:
            x0_nm = best_points.flatten()
            def objective_nm(x):
                pts = x.reshape((n, 2))
                s = compute_score(pts)
                if s == 0.0:
                    return 1.0
                return -s
            try:
                res_nm = minimize(objective_nm, x0_nm, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
                final_pts_nm = res_nm.x.reshape((n, 2))
                final_score_nm = compute_score(final_pts_nm)
                if final_score_nm > best_score:
                    best_score = final_score_nm
                    best_points = final_pts_nm.copy()
            except Exception:
                pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
