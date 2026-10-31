"""Heilbronn triangles: maximize min triangle area for 11 points in unit equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = 3 ** 0.5
H = SQRT3 / 2  # height of unit triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u+v<=1 to cartesian.
    Point = u*V0 + v*V1 + (1-u-v)*V2
    Actually let's use: point = V0 + x*(V1-V0) + y*(V2-V0) where x,y >= 0, x+y <= 1
    """
    # bary = (x, y) for each point, point = V0 + x*(V1-V0) + y*(V2-V0)
    pts = np.zeros((bary.shape[0], 2))
    pts[:, 0] = bary[:, 0] * 1.0 + bary[:, 1] * 0.5
    pts[:, 1] = bary[:, 1] * H
    return pts


def cart_to_bary(pts):
    """Inverse of bary_to_cart."""
    bary = np.zeros((pts.shape[0], 2))
    bary[:, 1] = pts[:, 1] / H
    bary[:, 0] = pts[:, 0] - 0.5 * bary[:, 1]
    return bary


def triangle_areas_from_bary(bary):
    """Compute all triangle areas from N points in barycentric coords.
    Returns array of areas (N choose 3 values).
    """
    n = bary.shape[0]
    # Convert to cartesian for area computation
    pts = bary_to_cart(bary)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle formed by pts[i], pts[j], pts[k]
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                areas.append(area)
    return np.array(areas)


def min_area_from_bary(bary):
    """Minimum triangle area."""
    return np.min(triangle_areas_from_bary(bary))


def smooth_min_area(bary, beta=100.0):
    """Smooth approximation of min area using log-sum-exp (soft-min).
    -log(sum(exp(-beta * a_i))) / beta approximates min(a_i)
    """
    areas = triangle_areas_from_bary(bary)
    # Soft min: -(1/beta) * log(sum(exp(-beta * a)))
    # For numerical stability, subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    return -log_sum / beta


def objective_and_grad(bary, beta=100.0):
    """Objective = -smooth_min_area (we maximize min area, so minimize negative).
    Returns value and gradient.
    """
    pts = bary_to_cart(bary)
    n = bary.shape[0]
    
    # Compute all triangle areas and their gradients
    areas = np.zeros((n*(n-1)*(n-2)//6,))
    # Gradient of each area w.r.t. bary coords
    # Area of triangle (i,j,k) in cartesian:
    # A = 0.5 * |cross(pj-pi, pk-pi)|
    # But we want signed area for differentiability. Let's use the fact that
    # for points in convex position, we can track signs.
    # Actually, let's just compute the min area directly and use finite differences
    # for gradient if needed, or use the smooth surrogate.
    
    # For the smooth surrogate, gradient of soft_min w.r.t. areas is:
    # d(soft_min)/d(a_i) = exp(-beta*a_i) / sum(exp(-beta*a_j))
    # Then chain rule: d(soft_min)/d(bary) = sum_i (d(soft_min)/d(a_i)) * (d(a_i)/d(bary))
    
    # This is complex. Let's use a simpler approach: just compute the value
    # and let scipy use numerical gradients, or compute gradient analytically.
    
    # Let me compute areas and their analytical gradients
    tri_idx = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_idx.append((i, j, k))
    
    m = len(tri_idx)
    areas = np.zeros(m)
    # Gradient of area w.r.t. bary coords: shape (m, n, 2)
    # For triangle (i,j,k) with cartesian points pi, pj, pk:
    # A = 0.5 * ((pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x))
    # dA/dpi = 0.5 * (-(pk_y - pi_y), (pk_x - pi_x))  ... wait let me redo
    # A = 0.5 * cross(pj-pi, pk-pi)
    # dA/dpi = -0.5 * cross(e1, pk-pi) - 0.5 * cross(pj-pi, e2) ... 
    # Let's use: A = 0.5 * (x_j - x_i)(y_k - y_i) - 0.5 * (y_j - y_i)(x_k - x_i)
    # dA/dx_i = -0.5*(y_k - y_i) + 0.5*(y_j - y_i) = 0.5*(y_j - y_k)
    # dA/dy_i = -0.5*(x_k - x_i) + 0.5*(x_j - x_i) = 0.5*(x_j - x_k)
    # dA/dx_j = 0.5*(y_k - y_i)
    # dA/dy_j = -0.5*(x_k - x_i)
    # dA/dx_k = -0.5*(y_j - y_i)
    # dA/dy_k = 0.5*(x_j - x_i)
    
    # Now convert from cartesian gradient to barycentric gradient
    # pt = V0 + u*(V1-V0) + v*(V2-V0)
    # d(pt_x)/du = 1, d(pt_x)/dv = 0.5
    # d(pt_y)/du = 0, d(pt_y)/dv = H
    # So dA/du = dA/dx * 1 + dA/dy * 0 = dA/dx
    #    dA/dv = dA/dx * 0.5 + dA/dy * H
    
    grad_bary = np.zeros((m, n, 2))
    
    for idx, (i, j, k) in enumerate(tri_idx):
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # Signed area (no abs for differentiability)
        A = 0.5 * ((pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0]))
        areas[idx] = A  # Use signed area; the min will handle signs
        
        # Gradients in cartesian
        dA_dx_i = 0.5 * (pj[1] - pk[1])
        dA_dy_i = 0.5 * (pj[0] - pk[0])
        dA_dx_j = 0.5 * (pk[1] - pi[1])
        dA_dy_j = -0.5 * (pk[0] - pi[0])
        dA_dx_k = -0.5 * (pj[1] - pi[1])
        dA_dy_k = 0.5 * (pj[0] - pi[0])
        
        # Convert to barycentric
        grad_bary[idx, i, 0] = dA_dx_i  # dA/du_i
        grad_bary[idx, i, 1] = 0.5 * dA_dx_i + H * dA_dy_i  # dA/dv_i
        grad_bary[idx, j, 0] = dA_dx_j
        grad_bary[idx, j, 1] = 0.5 * dA_dx_j + H * dA_dy_j
        grad_bary[idx, k, 0] = dA_dx_k
        grad_bary[idx, k, 1] = 0.5 * dA_dx_k + H * dA_dy_k
    
    # Soft min weights
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # d(soft_min)/d(a_i)
    
    # Gradient of soft_min w.r.t. bary
    # d(soft_min)/d(bary[p]) = sum_i weights[i] * d(a_i)/d(bary[p])
    # But we're minimizing -soft_min, so gradient is negated
    grad = np.zeros((n, 2))
    for idx in range(m):
        w = weights[idx]
        if w > 1e-10:
            grad += w * grad_bary[idx]
    
    value = -np.min(areas)  # Use true min for the objective value
    
    return value, grad


def objective_only(bary, beta=100.0):
    """Just the objective value for optimization."""
    pts = bary_to_cart(bary)
    n = bary.shape[0]
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                areas.append(area)
    return -np.min(areas)


def is_valid_bary(bary, margin=0.005):
    """Check if barycentric coords are valid (inside triangle with margin)."""
    if np.any(bary < -margin) or np.any(bary > 1 + margin):
        return False
    if np.any(bary[:, 0] + bary[:, 1] > 1 + margin):
        return False
    return True


def clip_bary(bary, margin=0.001):
    """Clip barycentric coords to be inside the triangle."""
    bary = bary.copy()
    bary[:, 0] = np.clip(bary[:, 0], margin, 1 - margin)
    bary[:, 1] = np.clip(bary[:, 1], margin, 1 - margin)
    # Ensure u + v <= 1
    s = bary[:, 0] + bary[:, 1]
    mask = s > 1 - margin
    if np.any(mask):
        # Scale down proportionally
        scale = (1 - margin) / s[mask]
        bary[mask, 0] *= scale
        bary[mask, 1] *= scale
    return bary


def random_bary_init(rng, n=N, margin=0.01):
    """Generate random valid barycentric coordinates."""
    bary = np.zeros((n, 2))
    for _ in range(n):
        while True:
            u = rng.random() * (1 - 2*margin) + margin
            v = rng.random() * (1 - u - margin) + margin
            if u + v <= 1 - margin:
                bary[_] = [u, v]
                break
    return bary


def optimize_from_start(bary_init, beta=50.0, max_iter=500, time_limit=None, start_time=None):
    """Optimize from a given initial point using L-BFGS-B."""
    if start_time is not None and time_limit is not None:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return None, -np.inf
    
    # Use numerical gradient for robustness, or analytical
    try:
        result = minimize(
            objective_only,
            bary_init,
            method='L-BFGS-B',
            bounds=[(0.001, 0.999)] * (2 * N),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
    except Exception:
        return None, -np.inf
    
    if start_time is not None and time_limit is not None:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return None, -np.inf
    
    if not result.success and result.fun > 0:
        # Still might be a valid result
        pass
    
    bary_opt = clip_bary(result.x.reshape(N, 2))
    val = -objective_only(bary_opt)
    return bary_opt, val


def polish(bary, beta=500.0, max_iter=1000):
    """High-precision polish pass with tighter tolerances."""
    try:
        result = minimize(
            objective_only,
            bary,
            method='L-BFGS-B',
            bounds=[(0.0001, 0.9999)] * (2 * N),
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
    except Exception:
        return bary
    
    bary_polished = clip_bary(result.x.reshape(N, 2), margin=0.0001)
    return bary_polished


def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_val = -np.inf
    
    # Multi-start with different seeds
    n_starts = 8
    for s in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 20:
            break
        
        # Different initialization strategies
        if s == 0:
            # Near-uniform grid
            bary_init = np.array([
                [0.5, 0.0],      # midpoint of bottom
                [0.0, 0.0],      # vertex 0
                [1.0, 0.0],      # vertex 1
                [0.0, 1.0],      # vertex 2
                [0.25, 0.25],
                [0.5, 0.25],
                [0.25, 0.5],
                [0.5, 0.5],
                [0.75, 0.25],
                [0.75, 0.5],
                [0.5, 0.75],
            ])
            bary_init = clip_bary(bary_init, margin=0.01)
        elif s == 1:
            # Fibonacci-like distribution
            golden = (1 + 5**0.5) / 2
            bary_init = np.zeros((N, 2))
            for i in range(N):
                t = (i + 0.5) / N
                u = t * (1 - t) * 4  # parabola in [0,1]
                v = ((i * golden) % 1.0) * (1 - u) * 0.5
                bary_init[i] = [u, v]
            bary_init = clip_bary(bary_init, margin=0.01)
        else:
            # Random with varying seed
            rng_s = np.random.default_rng(s * 1000 + 7)
            bary_init = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u = rng_s.random() * 0.95 + 0.025
                    v = rng_s.random() * (0.95 - u) + 0.025
                    if u + v <= 0.95:
                        bary_init[i] = [u, v]
                        break
        
        bary_opt, val = optimize_from_start(
            bary_init, beta=50.0, max_iter=300,
            time_limit=time_limit - 10, start_time=start_time
        )
        
        if bary_opt is not None and val > best_val:
            best_val = val
            best_bary = bary_opt
        
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15:
            break
    
    # Polish the best result
    if best_bary is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5:
            best_bary = polish(best_bary, beta=500.0, max_iter=500)
            best_val = -objective_only(best_bary)
    
    # Convert to cartesian
    if best_bary is not None:
        points = bary_to_cart(best_bary)
    else:
        # Fallback: simple random
        points = np.random.default_rng(7).random((N, 2))
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * H * 0.9 + 0.05 * H
    
    return points
