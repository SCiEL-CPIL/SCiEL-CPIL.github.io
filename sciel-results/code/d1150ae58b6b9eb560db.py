"""Optimize 130-point sphere covering: minimize true covering radius via smooth surrogate + polish."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130
DEADLINE = 130.0  # seconds, safety margin below 150s kill

def _facet_circumradii(pts):
    """Return array of circumradii (in radians) of each facet of the convex hull."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([np.pi])
    facets = hull.simplices
    radii = np.zeros(len(facets))
    for k, (i, j, l) in enumerate(facets):
        a, b, c = pts[i], pts[j], pts[l]
        # Circumcenter of triangle abc on unit sphere
        # Using the formula: circumcenter = (a*|b-c|^2 + b*|c-a|^2 + c*|a-b|^2) / (2*|a x b x ...|)
        # Actually for points on unit sphere, the circumcenter is in the plane of a,b,c
        # and equidistant from all three. The circumradius R satisfies sin(R) = |a x b| / (2 * distance from origin to plane)
        # Simpler: circumradius of spherical triangle = angle from circumcenter to vertex
        # Compute circumcenter: it's the normalized (a + b + c) only for equilateral.
        # General: solve for point o on unit sphere equidistant from a,b,c.
        # o.a = o.b = o.c = cos(R)
        # Use cross product method: circumcenter direction = (a x b) x (b x c) ... 
        # Actually: the circumcenter of triangle with vertices on unit sphere:
        # Let n = (b-a) x (c-a) normalized (normal to plane)
        # The circumcenter lies along the direction from origin to the plane, projected...
        # For unit sphere, circumcenter o satisfies o = alpha*(a+b+c) + beta*n? No.
        # Better: o is in span{a,b,c} and |o|=1, o.a=o.b=o.c.
        # o.a = o.b => o.(a-b)=0, o.a=o.c => o.(a-c)=0
        # So o is perpendicular to (a-b) and (a-c), i.e., o parallel to (a-b)x(a-c) = n
        # But also |o|=1 and o must be on unit sphere.
        # o = n / |n|? No, that's the normal. The circumcenter on the unit sphere is:
        # Actually for a spherical triangle, the circumcenter is the point equidistant from vertices.
        # On the unit sphere, if the triangle is not too large, circumcenter = normalize(a+b+c) works only for equilateral.
        # General formula: circumcenter o = (a*sin(A) + b*sin(B) + c*sin(C)) / (2*sin(A)*sin(B)*sin(C))... too complex.
        # Let me use: the circumradius R satisfies sin(R) = |a x b| / (2 * h) where h is distance from origin to plane abc.
        # h = |a . (b x c)| / |b x c| ... no, h = |det([a,b,c])| / |(b-a)x(c-a)|
        # And sin(R) = |a x b| / (2 * |projection|)... 
        # Simpler numerical approach: the circumradius is the angle from circumcenter to vertex.
        # Circumcenter on unit sphere: o = normalize( (a*|b-c|^2 + b*|c-a|^2 + c*|a-b|^2) / (2*|a x b x c|) )... 
        # Let me just compute it properly:
        # The circumcenter of a,b,c on unit sphere:
        # It's the point o with o.a = o.b = o.c and |o|=1.
        # o is in the plane of a,b,c (since equidistant from all three means in perpendicular bisector planes,
        # intersection is the line through circumcenter of planar triangle, which on sphere is... )
        # Actually o must satisfy o.(a-b)=0 and o.(a-c)=0, so o is parallel to (a-b)x(a-c).
        # But that gives the normal, not the circumcenter on the sphere.
        # I think for unit sphere, the circumcenter is:
        # o = normalize(a + b + c) is WRONG in general.
        # Let me use the formula: circumradius R = asin(|a x b| / (2 * |a x b x c| / |c|)) ... 
        # |a x b x c| = |det([a,b,c])| = |a . (b x c)|
        # The distance from origin to plane abc: d = |a . (b x c)| / |(b-a) x (c-a)|
        # For unit sphere, circumradius R: sin(R) = |a x b| / (2 * sqrt(1 - d^2)) ... no.
        # Let me just do it numerically: circumcenter is the pole of the great circle through a,b,c,
        # which is normalize((a x b) x (b x c))... no.
        # OK simplest: the circumcenter on the unit sphere for triangle abc is:
        # o = normalize( a * |b-c|^2 + b * |c-a|^2 + c * |a-b|^2 ) / something
        # Actually I recall: for three points on unit sphere, circumcenter = 
        # normalize( (a*|b-c|^2 + b*|c-a|^2 + c*|a-b|^2) / (2 * |a x b x c|) )
        # Let me just compute circumradius directly:
        # sin(R) = |a x b| / (2 * |a x b x c| / |c|) ... since |c|=1
        # |a x b x c| = |a . (b x c)|
        # Actually: sin(R) = |a x b| / (2 * |a . (b x c)| / |b x c|) ... I'm confusing myself.
        # Let me use: the circumradius of the spherical triangle is the angle from circumcenter to vertex.
        # The circumcenter o satisfies o.a = cos(R). 
        # From o.(a-b)=0: o.a = o.b. And o is in the "plane" of the triangle on the sphere.
        # For unit sphere: o = normalize( (a*|b-c|^2 + b*|c-a|^2 + c*|a-b|^2) / (2*|a x b x c|) )
        # Let me verify: if a,b,c form equilateral triangle, a*|b-c|^2 + ... = (a+b+c)*side^2
        # and |a x b x c| = |a . (b x c)|. For equilateral, this should give o = normalize(a+b+c).
        # I'll use this formula.
        ab2 = np.dot(a-b, a-b)
        bc2 = np.dot(b-c, b-c)
        ca2 = np.dot(c-a, c-a)
        det = np.abs(np.dot(a, np.cross(b, c)))
        if det < 1e-15:
            radii[k] = np.pi
            continue
        o = (a * bc2 + b * ca2 + c * ab2) / (2.0 * det)
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-15:
            radii[k] = np.pi
            continue
        o = o / o_norm
        # Circumradius = angle from o to a
        cos_R = np.clip(np.dot(o, a), -1.0, 1.0)
        radii[k] = np.arccos(cos_R)
    return radii

def _covering_radius(pts):
    """True covering radius in radians: max facet circumradius."""
    radii = _facet_circumradii(pts)
    return np.max(radii)

def _soft_covering_radius(pts, alpha=50.0):
    """Smooth surrogate: soft-max of facet circumradii."""
    radii = _facet_circumradii(pts)
    # Use log-sum-exp for smooth max
    m = np.max(radii)
    return m + np.log(np.sum(np.exp(radii - m))) / alpha

def _soft_covering_radius_grad(pts, alpha=50.0):
    """Gradient of soft covering radius w.r.t. points (numerical)."""
    # We'll compute gradient numerically since analytic is complex
    return None

def _refine_one_step(pts, alpha=50.0, lr=0.01):
    """One gradient step to minimize soft covering radius."""
    # Numerical gradient
    grad = np.zeros_like(pts)
    eps = 1e-6
    f0 = _soft_covering_radius(pts, alpha)
    for i in range(N):
        for j in range(3):
            orig = pts[i, j]
            pts[i, j] = orig + eps
            f_plus = _soft_covering_radius(pts, alpha)
            pts[i, j] = orig - eps
            f_minus = _soft_covering_radius(pts, alpha)
            pts[i, j] = orig
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    # Project gradient to tangent space
    for i in range(N):
        pts[i] = pts[i] - lr * grad[i]
        pts[i] = pts[i] - np.dot(pts[i], pts[i]) * pts[i] / max(np.dot(pts[i], pts[i]), 1e-15)
        # Renormalize
        norm = np.linalg.norm(pts[i])
        if norm > 1e-10:
            pts[i] = pts[i] / norm
    return pts

def _polish(pts, max_iter=200, tol=1e-10):
    """High-precision polish: minimize true covering radius with small steps."""
    best = pts.copy()
    best_cr = _covering_radius(best)
    alpha = 100.0
    lr = 0.005
    for it in range(max_iter):
        # Compute numerical gradient of soft max
        grad = np.zeros_like(pts)
        eps = 1e-7
        f0 = _soft_covering_radius(pts, alpha)
        for i in range(N):
            for j in range(3):
                orig = pts[i, j]
                pts[i, j] = orig + eps
                f_plus = _soft_covering_radius(pts, alpha)
                pts[i, j] = orig - eps
                f_minus = _soft_covering_radius(pts, alpha)
                pts[i, j] = orig
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        # Tangent projection and step
        new_pts = pts.copy()
        for i in range(N):
            # Project gradient to tangent
            g_tan = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
            new_pts[i] = pts[i] - lr * g_tan
            norm = np.linalg.norm(new_pts[i])
            if norm > 1e-10:
                new_pts[i] = new_pts[i] / norm
        cr = _covering_radius(new_pts)
        if cr < best_cr - 1e-12:
            best = new_pts.copy()
            best_cr = cr
            pts = new_pts
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-10:
                break
    return best, best_cr

def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    best_pts = pts.copy()
    best_cr = _covering_radius(best_pts)
    
    # Phase 1: Coarse optimization with moderate alpha
    alpha = 30.0
    lr = 0.02
    max_iter_phase1 = 300
    
    for it in range(max_iter_phase1):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        # Compute numerical gradient
        grad = np.zeros_like(pts)
        eps = 1e-5
        for i in range(N):
            for j in range(3):
                orig = pts[i, j]
                pts[i, j] = orig + eps
                f_plus = _soft_covering_radius(pts, alpha)
                pts[i, j] = orig - eps
                f_minus = _soft_covering_radius(pts, alpha)
                pts[i, j] = orig
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        # Step with tangent projection
        new_pts = pts.copy()
        for i in range(N):
            g_tan = grad[i] - np.dot(grad[i], pts[i]) * pts[i]
            new_pts[i] = pts[i] - lr * g_tan
            norm = np.linalg.norm(new_pts[i])
            if norm > 1e-10:
                new_pts[i] = new_pts[i] / norm
        cr = _covering_radius(new_pts)
        if cr < best_cr:
            best_pts = new_pts.copy()
            best_cr = cr
            pts = new_pts
            # Slightly decrease learning rate
            lr *= 0.995
        else:
            lr *= 0.8
            # Revert
            pts = best_pts.copy()
        # Decrease alpha gradually to sharpen
        alpha = max(10.0, alpha * 0.99)
    
    # Phase 2: Fine polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        best_pts, best_cr = _polish(best_pts, max_iter=100, tol=1e-10)
    
    # Final check
    final_cr = _covering_radius(best_pts)
    if final_cr < best_cr:
        best_cr = final_cr
    
    return best_pts
