import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0


def _raw_score(points):
    """
    Calculate the score: (d_min / d_max)^2.
    Returns 0 if invalid.
    """
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0

    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    dmin = np.min(dists)
    dmax = np.max(dists)
    if not (np.isfinite(dmin) and np.isfinite(dmax)):
        return 0.0
    if dmax <= 1e-15 or dmin <= 1e-15:
        return 0.0
    r = dmin / dmax
    return float(r * r)


def _tetrahedron_orbit_template():
    """
    Returns base directions for the 14 points using tetrahedral symmetry.
    Orbit 0 (4 points): Vertices of regular tetrahedron.
    Orbit 1 (10 points): Midpoints of edges of regular tetrahedron.
    """
    # Regular tetrahedron vertices (normalized to unit sphere)
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, -1, -1])
    v3 = np.array([-1, 1, -1])
    v4 = np.array([-1, -1, 1])
    
    verts = np.array([v1, v2, v3, v4])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edge midpoints (normalized)
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3)
    ]
    # Wait, a tetrahedron has 6 edges.
    # Let's list them properly.
    # 0-1, 0-2, 0-3
    # 1-2, 1-3
    # 2-3
    
    mids = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (verts[i] + verts[j]) / 2.0
            norm = np.linalg.norm(mid)
            if norm > 1e-10:
                mids.append(mid / norm)
    
    mids = np.array(mids)
    
    # Total 4 + 6 = 10 points.
    # We need 14 points.
    # The inspiration suggested "4 vertices + 10 edge midpoints" but a tetrahedron only has 6 edges.
    # Perhaps the inspiration meant something else or I am misinterpreting.
    # Let's re-read: "Adding the midpoints of the 6 edges gives 4+6=10 po..."
    # It cuts off. Maybe it meant something else to get to 14?
    # Or maybe it's a different polyhedron?
    # 14 points in 3D... 
    # Maybe it's not just tetrahedron.
    # Let's stick to a general high-symmetry approach or just use the best known 14-point configuration if I can recall it.
    # Actually, let's use a different symmetry group or just optimize the 42 coordinates directly but with a better initialization and method.
    # The current program uses spherical coordinates and SA.
    # The instruction says: "Represent the 14 points as vertices of a regular tetrahedron subdivided into 14 points using a high-symmetry combinatorial template... optimize only the single radial scaling factor of each of the 5 symmetry orbits via 1D coordinate descent"
    # This implies there are 5 orbits. 4+6=10. Where do the other 4 come from?
    # Maybe face centers? 4 faces. 10+4=14.
    # Yes! 4 vertices, 6 edge midpoints, 4 face centers. Total 14.
    
    # Face centers (normalized)
    faces = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3)
    ]
    centers = []
    for face in faces:
        c = np.mean(verts[list(face)], axis=0)
        norm = np.linalg.norm(c)
        if norm > 1e-10:
            centers.append(c / norm)
            
    centers = np.array(centers)
    
    # Combine
    # Orbit 0: Vertices (4)
    # Orbit 1: Edge Midpoints (6)
    # Orbit 2: Face Centers (4)
    # Total 14.
    # This has 3 orbits, not 5. But it's a very strong symmetric candidate.
    # Let's try optimizing the radii of these 3 orbits.
    
    return verts, mids, centers

def _symmetric_points(radii):
    """
    Construct 14 points from 3 orbital radii.
    radii: [r_v, r_m, r_c]
    """
    verts, mids, centers = _tetrahedron_orbit_template()
    
    pts = np.vstack([
        verts * radii[0],
        mids * radii[1],
        centers * radii[2]
    ])
    
    return pts

def _objective_symmetric(radii):
    """
    Objective function to minimize: -log(d_min/d_max)
    """
    pts = _symmetric_points(radii)
    dists = pdist(pts)
    dmin = np.min(dists)
    dmax = np.max(dists)
    
    if dmin <= 1e-15 or dmax <= 1e-15:
        return 1e10
        
    return np.log(dmax) - np.log(dmin)

def _find_best_symmetric():
    """
    Optimize the 3 radii.
    """
    # Initial guess: all 1.0
    x0 = np.array([1.0, 1.0, 1.0])
    
    # Try a few different initial guesses
    best_x = x0
    best_f = _objective_symmetric(x0)
    
    # Grid search for initial guess
    for rv in np.linspace(0.5, 1.5, 11):
        for rm in np.linspace(0.5, 1.5, 11):
            for rc in np.linspace(0.5, 1.5, 11):
                x = np.array([rv, rm, rc])
                f = _objective_symmetric(x)
                if f < best_f:
                    best_f = f
                    best_x = x
    
    # Polish with L-BFGS-B
    try:
        res = minimize(
            _objective_symmetric,
            best_x,
            method="L-BFGS-B",
            options={
                "maxiter": 1000,
                "maxfun": 50000,
                "ftol": 1e-15,
                "gtol": 1e-12,
            },
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception:
        pass
        
    pts = _symmetric_points(best_x)
    return pts

def _fibonacci_sphere(n):
    """
    Generate n points on a sphere using Fibonacci lattice.
    """
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - (i / (n - 1)) * 2
        radius = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / golden_ratio
        points[i] = [radius * np.cos(theta), radius * np.sin(theta), z]
    return points

def _refine_general(points, deadline):
    """
    Refine a general set of 14 points using L-BFGS-B on Cartesian coords.
    """
    if time.time() > deadline:
        return points
        
    def obj(x):
        pts = x.reshape(14, 3)
        dists = pdist(pts)
        dmin = np.min(dists)
        dmax = np.max(dists)
        if dmin <= 1e-15 or dmax <= 1e-15:
            return 1e10
        return np.log(dmax) - np.log(dmin)
        
    x0 = points.flatten()
    try:
        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            options={
                "maxiter": 500,
                "maxfun": 20000,
                "ftol": 1e-12,
                "gtol": 1e-10,
            },
        )
        return res.x.reshape(14, 3)
    except Exception:
        return points

def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN
    
    best_pts = None
    best_score = 0.0
    
    # 1. Try the high-symmetry tetrahedral subdivision
    pts_sym = _find_best_symmetric()
    score_sym = _raw_score(pts_sym)
    if score_sym > best_score:
        best_score = score_sym
        best_pts = pts_sym.copy()
        
    # 2. Try Fibonacci sphere initialization + refinement
    pts_fib = _fibonacci_sphere(14)
    # Scale to unit sphere? _fibonacci_sphere already returns unit sphere points.
    # Let's try refining this.
    if time.time() < deadline - 10:
        pts_ref = _refine_general(pts_fib, deadline - 5)
        score_ref = _raw_score(pts_ref)
        if score_ref > best_score:
            best_score = score_ref
            best_pts = pts_ref.copy()
            
    # 3. Multi-start random spherical + SA + Polish (from original, but faster)
    # We have plenty of time left if the above was fast.
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 15:
                break
            
            # Random start
            x_current = np.zeros(28)
            x_current[0::2] = rng.uniform(0, np.pi, 14)
            x_current[1::2] = rng.uniform(0, 2 * np.pi, 14)
            
            # Simple SA
            T = 1.0
            for _ in range(500):
                if time.time() > deadline - 10:
                    break
                x_prop = x_current.copy()
                idx = rng.integers(0, 28)
                x_prop[idx] += rng.normal(0, 0.5 * T)
                
                # Convert and score
                theta = x_prop[0::2]
                phi = x_prop[1::2]
                pts_prop = np.zeros((14, 3))
                pts_prop[:, 0] = np.sin(theta) * np.cos(phi)
                pts_prop[:, 1] = np.sin(theta) * np.sin(phi)
                pts_prop[:, 2] = np.cos(theta)
                
                dmin_p = np.min(pdist(pts_prop))
                dmax_p = np.max(pdist(pts_prop))
                if dmin_p > 1e-15 and dmax_p > 1e-15:
                    f_prop = np.log(dmax_p) - np.log(dmin_p)
                else:
                    f_prop = 1e10
                    
                theta_c = x_current[0::2]
                phi_c = x_current[1::2]
                pts_cur = np.zeros((14, 3))
                pts_cur[:, 0] = np.sin(theta_c) * np.cos(phi_c)
                pts_cur[:, 1] = np.sin(theta_c) * np.sin(phi_c)
                pts_cur[:, 2] = np.cos(theta_c)
                
                dmin_c = np.min(pdist(pts_cur))
                dmax_c = np.max(pdist(pts_cur))
                if dmin_c > 1e-15 and dmax_c > 1e-15:
                    f_cur = np.log(dmax_c) - np.log(dmin_c)
                else:
                    f_cur = 1e10
                    
                if f_prop < f_cur or rng.random() < np.exp(-(f_prop - f_cur) / T):
                    x_current = x_prop
                T *= 0.995
                
            # Convert final SA result to Cartesian
            theta = x_current[0::2]
            phi = x_current[1::2]
            pts_sa = np.zeros((14, 3))
            pts_sa[:, 0] = np.sin(theta) * np.cos(phi)
            pts_sa[:, 1] = np.sin(theta) * np.sin(phi)
            pts_sa[:, 2] = np.cos(theta)
            
            # Polish
            pts_pol = _refine_general(pts_sa, deadline - 5)
            score_pol = _raw_score(pts_pol)
            if score_pol > best_score:
                best_score = score_pol
                best_pts = pts_pol.copy()

    if best_pts is None:
        best_pts = _fibonacci_sphere(14)
        
    best_pts = np.asarray(best_pts, dtype=np.float64)
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((14, 3))
        
    return best_pts
