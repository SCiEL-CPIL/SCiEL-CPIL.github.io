"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _cover_radius(pts):
    """Compute the covering radius in degrees for points on the unit sphere.
    Uses the convex hull facets: for each facet, the circumradius of the
    spherical triangle is the angular distance from the facet center (pole)
    to the farthest vertex. The covering radius is the max over all facets.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    max_r = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]  # 3x3
        # Normalize
        tri = tri / np.linalg.norm(tri, axis=1, keepdims=True)
        a, b, c = tri[0], tri[1], tri[2]
        # Spherical circumradius: the pole of the triangle
        # Normal to the plane
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        n = n / nn
        # The pole is the unit normal. The angular distance from pole to each vertex:
        # cos(angle) = n . vertex
        cos_a = np.clip(np.dot(n, a), -1.0, 1.0)
        cos_b = np.clip(np.dot(n, b), -1.0, 1.0)
        cos_c = np.clip(np.dot(n, c), -1.0, 1.0)
        ang_a = np.arccos(cos_a)
        ang_b = np.arccos(cos_b)
        ang_c = np.arccos(cos_c)
        # The covering radius for this facet is the max angular distance from pole to vertices
        # But we need to check that the pole is on the correct side (outward)
        # For a convex hull, the outward normal should give positive dot with vertices
        # Actually, the circumradius is the angular distance from the centroid of the
        # spherical triangle to the farthest vertex. Let's use the pole approach.
        # The pole (unit normal) is the center of the smallest enclosing circle of the
        # three points on the sphere (if the triangle is small enough).
        r = np.max([ang_a, ang_b, ang_c])
        if r > max_r:
            max_r = r
    return np.degrees(max_r)


def _objective(x):
    """Objective: covering radius. x is a flat array of 3*N coordinates."""
    pts = x.reshape(N, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts / norms
    return _cover_radius(pts)


def _soft_max_objective(x, beta):
    """Smooth surrogate: soft-max of squared angular distances from facet poles to vertices."""
    from scipy.spatial import ConvexHull
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts / norms
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1e10
    vals = []
    for simplex in hull.simplices:
        tri = pts[simplex]
        tri = tri / np.linalg.norm(tri, axis=1, keepdims=True)
        a, b, c = tri[0], tri[1], tri[2]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        n = n / nn
        cos_a = np.clip(np.dot(n, a), -1.0, 1.0)
        cos_b = np.clip(np.dot(n, b), -1.0, 1.0)
        cos_c = np.clip(np.dot(n, c), -1.0, 1.0)
        ang_a = np.arccos(cos_a)
        ang_b = np.arccos(cos_b)
        ang_c = np.arccos(cos_c)
        # Use squared distances for soft-max
        sq = np.array([ang_a**2, ang_b**2, ang_c**2])
        vals.append(sq)
    if not vals:
        return 1e10
    vals = np.array(vals).flatten()
    # Soft max
    m = np.max(vals)
    if beta < 1e10:
        sm = m + np.log(np.sum(np.exp((vals - m) / beta))) * beta
    else:
        sm = m
    return np.degrees(np.sqrt(sm))


def _gradient_step(pts, lr, beta, n_inner=50):
    """Perform gradient descent on the soft-max objective using finite differences."""
    x = pts.flatten()
    for step in range(n_inner):
        # Compute gradient via central differences
        grad = np.zeros_like(x)
        h = 1e-5
        for j in range(len(x)):
            xp = x.copy(); xp[j] += h
            xm = x.copy(); xm[j] -= h
            fp = _soft_max_objective(xp, beta)
            fm = _soft_max_objective(xm, beta)
            grad[j] = (fp - fm) / (2 * h)
        # Normalize gradient
        gn = np.linalg.norm(grad)
        if gn < 1e-15:
            break
        grad = grad / gn
        x = x - lr * grad
    return x.reshape(N, 3)


def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    start_time = time.time()
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_score = _cover_radius(pts)
    
    # Phase 1: Soft-max gradient descent with annealing
    beta = 0.5
    lr = 0.02
    n_inner = 30
    while time.time() < deadline - 5.0:
        # Anneal beta
        beta = max(0.01, beta * 0.9)
        
        # Gradient step
        x = pts.flatten()
        for step in range(n_inner):
            if time.time() > deadline - 5.0:
                break
            grad = np.zeros_like(x)
            h = 1e-5
            for j in range(0, len(x), 3):  # Step by 3 for speed
                for k in range(3):
                    idx = j + k
                    xp = x.copy(); xp[idx] += h
                    xm = x.copy(); xm[idx] -= h
                    fp = _soft_max_objective(xp, beta)
                    fm = _soft_max_objective(xm, beta)
                    grad[idx] = (fp - fm) / (2 * h)
            gn = np.linalg.norm(grad)
            if gn < 1e-15:
                break
            grad = grad / gn
            x = x - lr * grad
        
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        pts = pts / norms
        
        score = _cover_radius(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
        
        # If we've converged, try a different beta or restart
        if score < best_score + 0.01:
            # Perturb to escape local minimum
            if beta > 0.1:
                beta *= 0.8
            # Small random perturbation
            pts = pts + 0.01 * np.random.randn(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
    
    # Phase 2: Local refinement with smaller steps
    if time.time() < deadline - 2.0:
        beta = 0.01
        lr = 0.005
        n_inner = 20
        x = best_pts.flatten()
        for step in range(n_inner):
            if time.time() > deadline - 2.0:
                break
            grad = np.zeros_like(x)
            h = 1e-6
            for j in range(len(x)):
                xp = x.copy(); xp[j] += h
                xm = x.copy(); xm[j] -= h
                fp = _soft_max_objective(xp, beta)
                fm = _soft_max_objective(xm, beta)
                grad[j] = (fp - fm) / (2 * h)
            gn = np.linalg.norm(grad)
            if gn < 1e-15:
                break
            grad = grad / gn
            x = x - lr * grad
        
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        pts = pts / norms
        score = _cover_radius(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
