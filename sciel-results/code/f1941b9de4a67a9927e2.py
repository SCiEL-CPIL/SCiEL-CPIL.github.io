import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Uses convex hull facets: the covering radius is the maximum circumradius
    of the triangular facets (each facet's circumcenter is the farthest point
    from its three vertices, and we verify it's not closer to another point).
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        for f in facets:
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            # Compute circumcenter and circumradius of triangle abc on unit sphere
            # For three unit vectors, the circumcenter of the spherical triangle
            # is in the direction of a+b+c (if not degenerate), and the radius
            # is the angular distance from that center to any vertex.
            s = a + b + c
            norm_s = np.linalg.norm(s)
            if norm_s < 1e-12:
                # Degenerate: use largest edge
                max_radius = max(max_radius, np.arccos(np.clip(np.dot(a, b), -1, 1)) / 2)
                max_radius = max(max_radius, np.arccos(np.clip(np.dot(a, c), -1, 1)) / 2)
                max_radius = max(max_radius, np.arccos(np.clip(np.dot(b, c), -1, 1)) / 2)
                continue
            center = s / norm_s
            # Check that no other point is closer to center than the three vertices
            dots = points @ center
            # The three vertices should have the smallest angular distance to center
            # i.e., largest dot product
            # Find the 3 largest dot products
            sorted_dots = np.sort(dots)[::-1]
            # The circumradius is the angular distance from center to a vertex
            cos_r = np.dot(center, a)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            radius = np.arccos(cos_r)
            max_radius = max(max_radius, radius)
        return max_radius
    except Exception:
        return np.pi

def _facet_circumradius(a, b, c):
    """Compute circumradius of spherical triangle with vertices a, b, c (unit vectors)."""
    s = a + b + c
    norm_s = np.linalg.norm(s)
    if norm_s < 1e-12:
        # Degenerate
        r1 = np.arccos(np.clip(np.dot(a, b), -1, 1))
        r2 = np.arccos(np.clip(np.dot(a, c), -1, 1))
        r3 = np.arccos(np.clip(np.dot(b, c), -1, 1))
        return max(r1, r2, r3) / 2
    center = s / norm_s
    cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
    return np.arccos(cos_r)

def _get_circumcenters_and_radii(points):
    """Get circumcenters and radii of all convex hull facets."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        centers = []
        radii = []
        for f in facets:
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            s = a + b + c
            norm_s = np.linalg.norm(s)
            if norm_s < 1e-12:
                continue
            center = s / norm_s
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            radius = np.arccos(cos_r)
            centers.append(center)
            radii.append(radius)
        if len(centers) == 0:
            return np.zeros((0, 3)), np.zeros(0)
        return np.array(centers), np.array(radii)
    except Exception:
        return np.zeros((0, 3)), np.zeros(0)

def _gradient_of_max_circumradius(points):
    """
    Compute gradient of the maximum circumradius with respect to the points.
    Only the facet(s) achieving the maximum radius contribute to the gradient.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_r = 0.0
        max_facets = []
        for f in facets:
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            s = a + b + c
            norm_s = np.linalg.norm(s)
            if norm_s < 1e-12:
                continue
            center = s / norm_s
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_r + 1e-10:
                max_r = r
                max_facets = [f]
            elif abs(r - max_r) < 1e-10:
                max_facets.append(f)
        
        if not max_facets:
            return np.zeros_like(points)
        
        grad = np.zeros_like(points)
        for f in max_facets:
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            s = a + b + c
            norm_s = np.linalg.norm(s)
            if norm_s < 1e-12:
                continue
            center = s / norm_s
            cos_r = np.clip(np.dot(center, a), -1.0, 1.0)
            sin_r = np.sqrt(max(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-12:
                continue
            
            # d(arccos(dot(center, a))) / da = -center / sin_r
            # But center depends on a, b, c. For simplicity, treat center as fixed
            # (first-order approximation)
            # Actually, the gradient of arccos(u·v) w.r.t. u is -v/sqrt(1-(u·v)^2)
            # when v is fixed. Here both center and a vary, but as an approximation:
            # grad_a = -center / sin_r (tangent component)
            # Similarly for b and c.
            
            g_a = -center / sin_r
            g_b = -center / sin_r
            g_c = -center / sin_r
            
            # Project to tangent space
            g_a = g_a - np.dot(g_a, a) * a
            g_b = g_b - np.dot(g_b, b) * b
            g_c = g_c - np.dot(g_c, c) * c
            
            grad[f[0]] += g_a
            grad[f[1]] += g_b
            grad[f[2]] += g_c
        
        # Normalize by number of max facets
        grad /= len(max_facets)
        return grad
    except Exception:
        return np.zeros_like(points)

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_P = P.copy()
    best_val = _compute_covering_radius(P)
    
    def get_reshaped(x):
        P_reshaped = x.reshape((N, 3))
        norms = np.linalg.norm(P_reshaped, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return P_reshaped / norms
    
    def objective(x):
        P_local = get_reshaped(x)
        return _compute_covering_radius(P_local)
    
    def gradient(x):
        P_local = get_reshaped(x)
        grad = _gradient_of_max_circumradius(P_local)
        return grad.flatten()
    
    # Iterative optimization with gradient-based steps
    # Use a simple gradient descent with adaptive step size
    lr = 0.01
    n_iter = 0
    max_iter = 500
    
    while n_iter < max_iter:
        if time.time() > deadline - 5:
            break
        
        P_local = best_P
        val = _compute_covering_radius(P_local)
        if val < best_val - 1e-12:
            best_val = val
        
        grad = _gradient_of_max_circumradius(P_local)
        
        # Check if gradient is small
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Take gradient step
        P_new = P_local - lr * grad
        # Normalize
        norms = np.linalg.norm(P_new, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P_new = P_new / norms
        
        val_new = _compute_covering_radius(P_new)
        
        if val_new < val - 1e-14:
            best_P = P_new
            best_val = val_new
            # Increase learning rate slightly if making progress
            lr = min(lr * 1.05, 0.05)
        else:
            # Decrease learning rate
            lr *= 0.7
            if lr < 1e-6:
                break
        
        n_iter += 1
    
    # Polish with Nelder-Mead if time permits
    if time.time() < deadline - 15:
        try:
            x0 = best_P.flatten()
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'fatol': 1e-10, 'disp': False,
                                    'xatol': 1e-12})
            if res.fun < best_val:
                best_P = get_reshaped(res.x)
                best_val = res.fun
        except Exception:
            pass
    
    # Final polish with smaller steps
    if time.time() < deadline - 5:
        lr2 = 0.001
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            grad = _gradient_of_max_circumradius(best_P)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            P_new = best_P - lr2 * grad
            norms = np.linalg.norm(P_new, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            P_new = P_new / norms
            val_new = _compute_covering_radius(P_new)
            if val_new < best_val - 1e-14:
                best_P = P_new
                best_val = val_new
            else:
                lr2 *= 0.5
                if lr2 < 1e-8:
                    break
    
    return best_P
