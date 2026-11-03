import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 10s margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _compute_all_vertices(inner_hexes):
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    return all_vertices

def _support_function(vertices, angle_deg):
    """Compute the maximum and minimum projections of vertices onto the normal at angle_deg.
    Returns (max_proj, min_proj, max_idx, min_idx)."""
    a = math.radians(angle_deg)
    nx = math.cos(a)
    ny = math.sin(a)
    
    max_proj = float('-inf')
    min_proj = float('inf')
    max_idx = 0
    min_idx = 0
    
    for i, (vx, vy) in enumerate(vertices):
        proj = vx * nx + vy * ny
        if proj > max_proj:
            max_proj = proj
            max_idx = i
        if proj < min_proj:
            min_proj = proj
            min_idx = i
    
    return max_proj, min_proj, max_idx, min_idx

def _smooth_max_support(vertices, angle_deg, beta=50.0):
    """Smooth approximation of max support using log-sum-exp.
    Returns (smooth_max, gradient_wrt_angle)."""
    a = math.radians(angle_deg)
    nx = math.cos(a)
    ny = math.sin(a)
    
    projections = np.array([vx * nx + vy * ny for vx, vy in vertices])
    
    # Log-sum-exp for smooth max
    m = np.max(projections)
    if m > 1e10:
        return m, 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (projections - m)))) / beta
    
    # Gradient of smooth max w.r.t. angle
    # d/dtheta of (v . n(theta)) = v . n'(theta) where n'(theta) = (-sin, cos)
    n_prime_x = -ny
    n_prime_y = nx
    
    # Weights for log-sum-exp
    exp_vals = np.exp(beta * (projections - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of each projection w.r.t. angle
    dprojs = np.array([vx * n_prime_x + vy * n_prime_y for vx, vy in vertices])
    
    grad_angle = np.sum(weights * dprojs)
    
    return smooth_max, grad_angle

def _smooth_min_support(vertices, angle_deg, beta=50.0):
    """Smooth approximation of min support using log-sum-exp (negated).
    Returns (smooth_min, gradient_wrt_angle)."""
    a = math.radians(angle_deg)
    nx = math.cos(a)
    ny = math.sin(a)
    
    projections = np.array([vx * nx + vy * ny for vx, vy in vertices])
    
    # For min, use log-sum-exp of negated projections
    neg_projections = -projections
    m = np.max(neg_projections)
    if m > 1e10:
        return -m, 0.0
    smooth_neg_max = m + math.log(np.sum(np.exp(beta * (neg_projections - m)))) / beta
    smooth_min = -smooth_neg_max
    
    # Gradient
    n_prime_x = -ny
    n_prime_y = nx
    
    exp_vals = np.exp(beta * (neg_projections - m))
    weights = exp_vals / np.sum(exp_vals)
    
    dprojs = np.array([vx * n_prime_x + vy * n_prime_y for vx, vy in vertices])
    # Gradient of -proj w.r.t. angle is -dproj/dangle
    grad_angle = -np.sum(weights * dprojs)
    
    return smooth_min, grad_angle

def _objective_smooth(params, vertices, beta=50.0):
    """Objective: minimize the inradius t of the enclosing hexagon.
    params = [cx, cy, angle_deg]
    
    The inradius t must satisfy:
    For each of 6 normals at angles (angle + 30 + 60k):
      max_proj - cx*nx - cy*ny <= t
      -(min_proj - cx*nx - cy*ny) <= t  =>  cx*nx + cy*ny - min_proj <= t
    
    So t >= max over all 6 directions of:
      max_proj_k - cx*nx_k - cy*ny_k
      cx*nx_k + cy*ny_k - min_proj_k
    
    We use a smooth max over all 12 quantities.
    """
    cx, cy, angle_deg = params
    
    # All 12 constraints (6 directions, 2 sides each)
    values = []
    grads_cx = []
    grads_cy = []
    grads_angle = []
    
    for k in range(6):
        a = angle_deg + 30 + 60 * k
        nx = math.cos(math.radians(a))
        ny = math.sin(math.radians(a))
        
        # Max side: max_proj - cx*nx - cy*ny
        max_proj, _, _, _ = _support_function(vertices, a)
        smooth_max, grad_max = _smooth_max_support(vertices, a, beta)
        val1 = smooth_max - cx * nx - cy * ny
        values.append(val1)
        grads_cx.append(-nx)
        grads_cy.append(-ny)
        grads_angle.append(grad_max)
        
        # Min side: cx*nx + cy*ny - min_proj
        min_proj, _, _, _ = _support_function(vertices, a)
        smooth_min, grad_min = _smooth_min_support(vertices, a, beta)
        val2 = cx * nx + cy * ny - smooth_min
        values.append(val2)
        grads_cx.append(nx)
        grads_cy.append(ny)
        grads_angle.append(-grad_min)  # derivative of -smooth_min
    
    values = np.array(values)
    grads_cx = np.array(grads_cx)
    grads_cy = np.array(grads_cy)
    grads_angle = np.array(grads_angle)
    
    # Smooth max of all 12 values
    m = np.max(values)
    if m > 1e10:
        return m, [0, 0, 0]
    
    exp_vals = np.exp(beta * (values - m))
    weights = exp_vals / np.sum(exp_vals)
    
    smooth_t = m + math.log(np.sum(exp_vals)) / beta
    
    grad_cx = np.sum(weights * grads_cx)
    grad_cy = np.sum(weights * grads_cy)
    grad_angle = np.sum(weights * grads_angle)
    
    return smooth_t, [grad_cx, grad_cy, grad_angle]

def _exact_inradius(cx, cy, angle_deg, vertices):
    """Compute exact inradius (t) for given center and angle."""
    t = 0.0
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        
        for vx, vy in vertices:
            proj = (vx - cx) * nx + (vy - cy) * ny
            t = max(t, abs(proj))
    
    return t

def _exact_side(cx, cy, angle_deg, vertices):
    """Compute exact side length for given center and angle."""
    t = _exact_inradius(cx, cy, angle_deg, vertices)
    return t / (math.sqrt(3) / 2)

def _check_disjoint(inner_hexes, eps=1e-6):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _hexagons_overlap(h1, h2, eps=1e-6):
    verts1 = _get_hex_vertices(h1[0], h1[1], h1[2])
    verts2 = _get_hex_vertices(h2[0], h2[1], h2[2])
    
    axes = []
    for verts in [verts1, verts2]:
        for k in range(6):
            x1, y1 = verts[k]
            x2, y2 = verts[(k + 1) % 6]
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-12:
                continue
            nx = -dy / length
            ny = dx / length
            axes.append((nx, ny))
            
    for nx, ny in axes:
        min1, max1 = float('inf'), float('-inf')
        min2, max2 = float('inf'), float('-inf')
        
        for vx, vy in verts1:
            proj = vx * nx + vy * ny
            min1 = min(min1, proj)
            max1 = max(max1, proj)
            
        for vx, vy in verts2:
            proj = vx * nx + vy * ny
            min2 = min(min2, proj)
            max2 = max(max2, proj)
            
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
            
    return True

def _optimize_outer_for_lattice(inner_hexes, beta=50.0):
    """Optimize outer hexagon (center, angle) for fixed inner lattice using smooth objective."""
    vertices = _compute_all_vertices(inner_hexes)
    
    # Initial guess: centroid of vertices
    cx0 = sum(v[0] for v in vertices) / len(vertices)
    cy0 = sum(v[1] for v in vertices) / len(vertices)
    
    best_side = float('inf')
    best_result = (cx0, cy0, 0.0)
    
    # Try multiple starting angles
    for angle0 in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
        if time.time() > _DEADLINE - 10.0:
            break
        
        x0 = np.array([cx0, cy0, angle0])
        
        def obj_and_grad(x):
            if time.time() > _DEADLINE - 10.0:
                return 1e10, [0, 0, 0]
            val, grad = _objective_smooth(x, vertices, beta)
            return val, grad
        
        try:
            result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            if result.fun < 1e9:
                cx, cy, angle = result.x
                angle = angle % 30  # Normalize to [0, 30)
                side = _exact_side(cx, cy, angle, vertices)
                if side < best_side:
                    best_side = side
                    best_result = (cx, cy, angle)
        except Exception:
            pass
    
    # Fine-grained search around best angle
    if best_result[2] < 30:
        best_angle = best_result[2]
        for da in np.arange(-2.0, 2.01, 0.05):
            if time.time() > _DEADLINE - 5.0:
                break
            test_angle = (best_angle + da) % 30
            cx, cy = best_result[0], best_result[1]
            side = _exact_side(cx, cy, test_angle, vertices)
            if side < best_side:
                best_side = side
                best_result = (cx, cy, test_angle)
    
    # Very fine refinement
    if best_result[2] < 30:
        best_angle = best_result[2]
        for da in np.arange(-0.1, 0.11, 0.005):
            if time.time() > _DEADLINE - 3.0:
                break
            test_angle = (best_angle + da) % 30
            cx, cy = best_result[0], best_result[1]
            side = _exact_side(cx, cy, test_angle, vertices)
            if side < best_side:
                best_side = side
                best_result = (cx, cy, test_angle)
    
    return best_result, best_side

def _coarse_angle_search(inner_hexes, vertices):
    """Coarse search over outer angles."""
    cx0 = sum(v[0] for v in vertices) / len(vertices)
    cy0 = sum(v[1] for v in vertices) / len(vertices)
    
    best_side = float('inf')
    best_angle = 0.0
    best_center = (cx0, cy0)
    
    for angle in np.arange(0, 30.1, 0.5):
        if time.time() > _DEADLINE - 15.0:
            break
        side = _exact_side(cx0, cy0, angle, vertices)
        if side < best_side:
            best_side = side
            best_angle = angle
    
    # Fine search around best
    for angle in np.arange(best_angle - 0.5, best_angle + 0.51, 0.01):
        if angle < 0 or angle > 30:
            continue
        if time.time() > _DEADLINE - 10.0:
            break
        side = _exact_side(cx0, cy0, angle, vertices)
        if side < best_side:
            best_side = side
            best_angle = angle
    
    return best_center, best_angle, best_side

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate 11 points on triangular lattice
    sqrt3 = math.sqrt(3)
    scale = 1.000001  # Slightly larger than touching
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y, i, j))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y, _, _ in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Try different inner hexagon angles
    best_overall_side = float('inf')
    best_overall_inner = None
    best_overall_center = (0.0, 0.0)
    best_overall_angle = 0.0
    
    for inner_angle in [0.0, 15.0, 30.0]:
        if time.time() > _DEADLINE - 20.0:
            break
        
        inner = [[pos[0], pos[1], inner_angle] for pos in positions]
        vertices = _compute_all_vertices(inner)
        
        # First do coarse search to get good initial center/angle
        center, angle, side = _coarse_angle_search(inner, vertices)
        
        if side < best_overall_side:
            best_overall_side = side
            best_overall_inner = inner
            best_overall_center = center
            best_overall_angle = angle
        
        # Then do smooth optimization starting from this point
        if time.time() < _DEADLINE - 10.0:
            # Override initial guess with coarse result
            cx0, cy0 = center
            angle0 = angle
            
            x0 = np.array([cx0, cy0, angle0])
            
            def obj_and_grad(x):
                if time.time() > _DEADLINE - 10.0:
                    return 1e10, [0, 0, 0]
                val, grad = _objective_smooth(x, vertices, beta=100.0)
                return val, grad
            
            try:
                result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                                 options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
                if result.fun < 1e9:
                    cx, cy, angle = result.x
                    angle = angle % 30
                    side = _exact_side(cx, cy, angle, vertices)
                    if side < best_overall_side:
                        best_overall_side = side
                        best_overall_inner = inner
                        best_overall_center = (cx, cy)
                        best_overall_angle = angle
            except Exception:
                pass
        
        # Final fine refinement on the best so far
        if time.time() < _DEADLINE - 5.0 and best_overall_inner == inner:
            vertices_best = _compute_all_vertices(best_overall_inner)
            cx_best, cy_best = best_overall_center
            angle_best = best_overall_angle
            
            for da in np.arange(-0.5, 0.51, 0.01):
                if time.time() > _DEADLINE - 3.0:
                    break
                test_angle = (angle_best + da) % 30
                side = _exact_side(cx_best, cy_best, test_angle, vertices_best)
                if side < best_overall_side:
                    best_overall_side = side
                    best_overall_angle = test_angle
            
            # Even finer
            for da in np.arange(-0.05, 0.051, 0.001):
                if time.time() > _DEADLINE - 2.0:
                    break
                test_angle = (best_overall_angle + da) % 30
                side = _exact_side(cx_best, cy_best, test_angle, vertices_best)
                if side < best_overall_side:
                    best_overall_side = side
                    best_overall_angle = test_angle
    
    # Verify disjointness
    if best_overall_inner is None or not _check_disjoint(best_overall_inner, eps=-1e-5):
        # Revert to basic lattice with angle 0
        inner = [[pos[0], pos[1], 0.0] for pos in positions]
        vertices = _compute_all_vertices(inner)
        cx0 = sum(v[0] for v in vertices) / len(vertices)
        cy0 = sum(v[1] for v in vertices) / len(vertices)
        best_overall_center = (cx0, cy0)
        best_overall_angle = 0.0
        best_overall_side = _exact_side(cx0, cy0, 0.0, vertices)
        best_overall_inner = inner
    
    return best_overall_inner, (best_overall_center[0], best_overall_center[1]), best_overall_side, best_overall_angle

