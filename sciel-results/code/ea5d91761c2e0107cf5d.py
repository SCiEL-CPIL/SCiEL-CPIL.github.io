"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _facet_circumcenter(p1, p2, p3):
    """Compute circumcenter of three unit vectors on sphere (plane through origin)."""
    # The circumcenter of a spherical triangle lies in the plane of the three points
    # Solve for c = a*p1 + b*p2 + c3*p3 such that |c - p1|^2 = |c - p2|^2 = |c - p3|^2
    # and c is in span(p1,p2,p3)
    # Actually, for points on unit sphere, the circumcenter in 3D satisfies:
    # c . p1 = c . p2 = c . p3 = d (same dot product)
    # and c is in the plane spanned by p1,p2,p3
    M = np.array([p1, p2, p3])  # 3x3
    # We want c = M @ x such that M^T c = d * ones
    # c . p1 = x^T M^T M x1... let's use linear system
    # c . pi = d for i=1,2,3
    # Let c = a*p1 + b*p2 + c3*p3
    # c . p1 = a + b*(p2.p1) + c3*(p3.p1) = d
    # c . p2 = a*(p1.p2) + b + c3*(p3.p2) = d
    # c . p3 = a*(p1.p3) + b*(p2.p3) + c3 = d
    G = np.array([
        [1.0, p2 @ p1, p3 @ p1],
        [p1 @ p2, 1.0, p3 @ p2],
        [p1 @ p3, p2 @ p3, 1.0]
    ])
    rhs = np.ones(3)
    try:
        x = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        return None, None
    c = x[0]*p1 + x[1]*p2 + x[2]*p3
    d = c @ p1
    # Circumradius: |c - p1|
    r = np.linalg.norm(c - p1)
    if r < 1e-12:
        return None, None
    # The angular covering radius contribution: arcsin(r) approximately, but exactly:
    # The cap radius is the angle from circumcenter direction to p1
    # cos(angle) = (c - p1).norm... actually the angular distance from c/|c| to p1
    cn = c / np.linalg.norm(c)
    cos_ang = np.clip(cn @ p1, -1.0, 1.0)
    ang = np.arccos(cos_ang)
    return cn, ang

def _compute_covering_radius(pts):
    """Compute covering radius via convex hull facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback
    max_ang = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Check they're not collinear
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) < 1e-10:
            continue
        cn, ang = _facet_circumcenter(p1, p2, p3)
        if cn is not None:
            # Verify the cap is empty: all other points should be on the same side
            # as the circumcenter (i.e., the circumcenter is "outside" relative to the sphere interior)
            # Actually for covering, we need the circumcenter to be in the "empty cap" direction
            # The facet normal should point away from origin
            n = np.cross(p2 - p1, p3 - p1)
            n = n / np.linalg.norm(n)
            # Ensure normal points outward
            if n @ p1 < 0:
                n = -n
            # Check all points are on the same side as origin (inside the hull)
            # The empty cap condition: circumcenter direction should have all points
            # with dot product <= cn @ p1 (i.e., p1 is the farthest in that direction)
            dots = pts @ cn
            if np.max(dots) > cn @ p1 + 1e-9:
                # Cap is not empty, this facet's circumradius doesn't count
                continue
            max_ang = max(max_ang, ang)
    return max_ang

def _soft_max_covering_radius(pts, beta=100.0):
    """Smooth surrogate for max circumradius using log-sum-exp."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros(N)
    angs = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        if np.linalg.norm(cross) < 1e-10:
            continue
        cn, ang = _facet_circumcenter(p1, p2, p3)
        if cn is not None:
            dots = pts @ cn
            if np.max(dots) <= cn @ p1 + 1e-9:
                angs.append(ang)
    if not angs:
        return np.pi, np.zeros(N)
    angs = np.array(angs)
    # Log-sum-exp for soft max
    m = np.max(angs)
    s = np.sum(np.exp(beta * (angs - m)))
    soft_max = m + np.log(s) / beta
    return soft_max, angs

def _objective(x, beta=100.0):
    pts = _normalize(x)
    val, _ = _soft_max_covering_radius(pts, beta)
    return val

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_val = _compute_covering_radius(best_pts)
    
    # Multi-stage optimization with increasing beta
    betas = [10, 50, 200, 500]
    for beta in betas:
        if time.time() > deadline - 15:
            break
        # Try a few random perturbations as restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            if restart == 0:
                x0 = best_pts.copy()
            else:
                x0 = best_pts + np.random.normal(0, 0.1, (N, 3))
                x0 = _normalize(x0)
            
            res = minimize(
                _objective, x0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if time.time() > deadline - 5:
                break
            new_pts = _normalize(res.x)
            new_val = _compute_covering_radius(new_pts)
            if new_val < best_val - 1e-10:
                best_val = new_val
                best_pts = new_pts
    
    # Final polish with tight tolerance
    if time.time() < deadline - 5:
        res = minimize(
            _objective, best_pts, args=(500,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        new_pts = _normalize(res.x)
        new_val = _compute_covering_radius(new_pts)
        if new_val < best_val - 1e-10:
            best_val = new_val
            best_pts = new_pts
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END

