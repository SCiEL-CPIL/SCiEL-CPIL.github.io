"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds; kill margin before 150s


def _covering_radius_degrees(pts):
    """Compute the covering radius (degrees) as the largest circumradius of hull facets."""
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return 90.0
    facets = hull.simplices
    best = 0.0
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Circumcenter of triangle on unit sphere (in 3D, the circumcenter of the
        # great-circle triangle is the normalized cross-product formula).
        # For points on unit sphere, the circumcenter of the spherical triangle
        # with vertices a,b,c (unit vectors) is proportional to a x b + b x c + c x a
        # (the pole of the great circle through them is a x b, but the circumcenter
        # of the spherical triangle is more carefully computed).
        # Actually, for three unit vectors, the center of the minimal enclosing
        # cap (spherical circumcenter) can be computed as follows:
        # The circumcenter of a spherical triangle with vertices a, b, c is the
        # point x (unit vector) maximizing the minimum angular distance to a,b,c.
        # Equivalently, x is the unit vector that makes x·a = x·b = x·c.
        # This gives x ∝ a(b·c - a·b) + b(a·c - b·a) + c(a·b - c·a) ... 
        # Let's use a more robust formula.
        
        # The spherical circumcenter of three unit vectors a, b, c:
        # Let n = (a × b) · c / |a × b|  ... this is the pole.
        # Actually the circumcenter x satisfies x·a = x·b = x·c = cos(R)
        # where R is the circumradius. We can find x as the unit vector in the
        # direction of (a × (b × c)) + (b × (c × a)) + (c × (a × b))
        # which simplifies to:
        # x ∝ a(b·c - a·b) + b(a·c - b·a) + c(a·b - c·a)  -- no, let me derive.
        
        # Use the formula: for three unit vectors, the circumcenter is
        # x = normalize( a*(b·c) + b*(c·a) + c*(a·b) - (a·b)(b·c)(c·a) * something )
        # This is getting complex. Let me use a numerical approach for safety.
        
        # Simpler: the circumradius of a facet is the max angular distance from
        # the circumcenter to any vertex. The circumcenter of the spherical triangle
        # formed by three unit vectors a, b, c is the unit vector x that is
        # equidistant from all three. We can solve this with a 3x3 system.
        
        # Actually, let's use the known formula. For three unit vectors a, b, c,
        # the spherical circumcenter is:
        # x ∝ sin(A) a + sin(B) b + sin(C) c  where A,B,C are spherical angles
        # This is complex. Let me just compute it numerically.
        
        # Use the fact that x·a = x·b = x·c. Let x = α a + β b + γ c.
        # This gives a 3x3 linear system. Let M be the matrix with columns a,b,c.
        # x = M @ w, and x·a = x·b = x·c means (M^T M) w = (M^T M) @ [1,1,1] / 3? 
        # No, x·a = x·b = x·c means M^T @ x = [1,1,1]^T * cos(R).
        # So M^T @ M @ w = [1,1,1]^T * cos(R).
        # This is underdetermined. Let's use a different approach.
        
        # The circumcenter is the point on the sphere equidistant from a, b, c.
        # It lies in the plane perpendicular to the great-circle arc midpoints...
        # Let me just use a simple iterative approach or the formula:
        
        # For three points on unit sphere, the circumcenter x satisfies:
        # x = normalize( a * (b·c - (a·b)(a·c)) + b * (c·a - (b·c)(b·a)) + c * (a·b - (c·a)(c·b)) )
        # This is the vector formula for the circumcenter of a spherical triangle.
        
        ab = np.dot(a, b)
        bc = np.dot(b, c)
        ca = np.dot(c, a)
        ac = ca
        ba = ab
        cb = bc
        
        x_vec = a * (bc - ab * ac) + b * (ca - bc * ba) + c * (ab - ca * cb)
        norm = np.linalg.norm(x_vec)
        if norm < 1e-10:
            # Degenerate; use the midpoint of the longest edge as approximation
            continue
        x = x_vec / norm
        # Compute angular distance from x to each vertex
        d1 = np.arccos(np.clip(np.dot(x, a), -1, 1))
        d2 = np.arccos(np.clip(np.dot(x, b), -1, 1))
        d3 = np.arccos(np.clip(np.dot(x, c), -1, 1))
        r_facet = max(d1, d2, d3)
        if r_facet > best:
            best = r_facet
    return np.degrees(best)


def _objective_and_grad(pts_flat):
    """Objective: covering radius (radians). Returns value and gradient."""
    pts = pts_flat.reshape(-1, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts, qhull_options='QJ')
    except Exception:
        return 90.0, np.zeros_like(pts_flat)
    
    facets = hull.simplices
    # Compute circumradius for each facet and find the max
    active = []  # list of (radius, facet_indices)
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        ab = np.dot(a, b)
        bc = np.dot(b, c)
        ca = np.dot(c, a)
        
        x_vec = a * (bc - ab * ca) + b * (ca - bc * ab) + c * (ab - ca * bc)
        norm = np.linalg.norm(x_vec)
        if norm < 1e-10:
            continue
        x = x_vec / norm
        d1 = np.arccos(np.clip(np.dot(x, a), -1, 1))
        d2 = np.arccos(np.clip(np.dot(x, b), -1, 1))
        d3 = np.arccos(np.clip(np.dot(x, c), -1, 1))
        r_facet = max(d1, d2, d3)
        active.append((r_facet, tri, x, (a, b, c)))
    
    if not active:
        return 90.0, np.zeros_like(pts_flat)
    
    max_r = max(r for r, _, _, _ in active)
    
    # Aggregate gradients from facets near the maximum
    thresh = max_r - 1e-4
    grad = np.zeros_like(pts)
    for r_facet, tri, x, (a, b, c) in active:
        if r_facet < thresh:
            continue
        weight = 1.0  # equal weight for active facets
        # Gradient of angular distance from x to point p on sphere:
        # d/dp arccos(p·x) = -x / sqrt(1 - (p·x)^2) projected onto tangent space
        # But we want to move p to reduce the max distance, so gradient direction is
        # toward reducing distance, i.e., move p toward x in tangent space.
        # The gradient of distance w.r.t. p (on sphere) is -x/sqrt(1-(p·x)^2) projected.
        # For minimization, we want to move in the negative gradient direction.
        # Let's compute: if p moves toward x, distance decreases.
        # grad_p = -x / max(sqrt(1 - (p·x)^2), 1e-10)
        # Project onto tangent space: grad_t = grad_p - (grad_p · p) p
        
        for p in (a, b, c):
            dot_px = np.dot(p, x)
            denom = max(np.sqrt(max(1.0 - dot_px * dot_px, 0.0)), 1e-10)
            gp = -x / denom
            # Project to tangent space
            gp = gp - np.dot(gp, p) * p
            grad[np.where(tri[0] == tri[0])[0] if False else 0]  # placeholder
            
    # The above gradient aggregation is tricky. Let me use a simpler approach:
    # For each active facet, for each vertex p in the facet, the gradient of the
    # circumradius w.r.t. p is in the direction that moves p toward the circumcenter x.
    # On the sphere, this is the tangent vector: x - (x·p) p, normalized appropriately.
    
    grad = np.zeros_like(pts)
    for r_facet, tri, x, (a, b, c) in active:
        if r_facet < thresh:
            continue
        w = 1.0
        for idx, p in zip(tri, (a, b, c)):
            # Move p toward x: direction is x projected to tangent at p
            # tan_dir = x - (x·p) p
            tan_dir = x - np.dot(x, p) * p
            norm_td = np.linalg.norm(tan_dir)
            if norm_td > 1e-12:
                grad[idx] += w * tan_dir / norm_td
    
    # Normalize gradient
    gn = np.linalg.norm(grad)
    if gn > 1e-12:
        grad = grad / gn
    
    return max_r, grad.flatten()


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def _optimize_from_start(pts0, time_budget):
    """Run local optimization from pts0 within time_budget seconds."""
    t0 = time.time()
    best_pts = pts0.copy()
    best_r = _covering_radius_degrees(best_pts)
    
    # Use L-BFGS with line search, multiple restarts with small perturbations
    for restart in range(10):
        if time.time() - t0 > time_budget:
            break
        if restart == 0:
            pts = pts0.copy()
        else:
            # Small random perturbation
            noise = np.random.randn(N, 3) * 0.05
            pts = pts0 + noise
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-12)
        
        def obj_and_grad(flat):
            r, g = _objective_and_grad(flat)
            return r, g
        
        res = minimize(obj_and_grad, pts.flatten(), method='L-BFGS-B',
                       jac=True, options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50})
        
        if res.success or res.fun < best_r:
            new_pts = res.x.reshape(-1, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.maximum(norms, 1e-12)
            new_r = _covering_radius_degrees(new_pts)
            if new_r < best_r:
                best_r = new_r
                best_pts = new_pts.copy()
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    
    # Start with Fibonacci sphere
    pts0 = _fibonacci_sphere(N)
    
    # Run optimization with time budget
    time_budget = DEADLINE - (time.time() - start_time) - 5.0
    if time_budget < 5.0:
        return pts0
    
    best_pts, best_r = _optimize_from_start(pts0, time_budget)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 2.0:
        return best_pts
    
    # Final polish: run a few more iterations on the best solution
    remaining = DEADLINE - (time.time() - start_time) - 3.0
    if remaining > 5.0:
        best_pts, best_r = _optimize_from_start(best_pts, remaining)
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-12)
    
    return best_pts
# EVOLVE-BLOCK-END
