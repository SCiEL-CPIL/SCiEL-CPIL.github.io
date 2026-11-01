"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

_START_TIME = time.time()
_DEADLINE = 130.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return pts / n

def _spherical_voronoi_inradii(pts):
    """
    For each site, compute the inradius of its spherical Voronoi cell.
    The Voronoi cell of site i is { y : d(y, p_i) <= d(y, p_j) for all j }.
    On the sphere, the bisector plane between i and j is { y : y . (p_i - p_j) = 0 }.
    The cell is intersection of hemispheres y . (p_i - p_j) >= 0 for all j != i.
    The farthest point from p_i in this cell is a vertex (intersection of 3 planes).
    We compute inradius as max over vertices of angle(p_i, vertex).
    
    For efficiency, we find vertices by checking triples of constraints.
    """
    n = len(pts)
    inradii = np.zeros(n)
    
    # Precompute difference vectors
    # For site i, constraint j: y . (p_i - p_j) >= 0
    # Vertex is intersection of 3 such planes: y . d1 = 0, y . d2 = 0, y . d3 = 0
    # => y is parallel to d1 x d2, and we check it satisfies all other constraints
    
    for i in range(n):
        pi = pts[i]
        # Constraints: y . (pi - pj) >= 0 for j != i
        diffs = pi[None, :] - pts  # shape (n, 3)
        # For j=i, diff is 0, skip
        valid_j = [j for j in range(n) if j != i]
        
        best_angle = 0.0
        # Check all triples of constraints to find vertices
        # This is O(n^3) per site, O(n^4) total - too slow for n=50 in Python
        # Alternative: use the fact that the inradius equals the max over j of 
        # the distance to the midpoint of the edge, or use a different approach.
        
        # Better approach: The inradius of site i's Voronoi cell equals
        # max over vertices. But we can also use: the covering radius contribution
        # from site i is the max distance from any point in its cell to pi.
        # 
        # Actually, a simpler and more robust approach: the inradius equals
        # the maximum, over all j != i, of the distance from pi to the point
        # on the bisector plane between i and j that is farthest from pi
        # while still being in the cell. But this is complex.
        #
        # Let's use a different strategy: for each site i, the inradius is
        # at least the max over j of angle(pi, midpoint_on_bisector).
        # The midpoint on the bisector plane between i and j in the direction
        # away from pi is: normalize(pi + pj) is on the bisector? No.
        # The bisector plane is { y : y.(pi-pj)=0 }. The point on this plane
        # farthest from pi (in angle) within the cell...
        #
        # Let me just do the triple check but vectorize over triples.
        
        # Get all diff vectors for site i
        di = diffs[valid_j]  # shape (n-1, 3)
        m = len(di)
        
        if m < 3:
            inradii[i] = np.pi
            continue
        
        # For each triple (a, b, c), compute vertex = normalize(di[a] x di[b])
        # and check if it satisfies all constraints
        # This is still O(m^3) = O(n^3) per site
        
        # Vectorized triple approach is complex. Let's use a simpler surrogate:
        # The inradius is bounded below by max_j angle(pi, normalize(pi+di_j))
        # where di_j = pi - pj, so pi + di_j = 2*pi - pj
        # Wait, that's not on the bisector.
        
        # Point on bisector plane y.(pi-pj)=0: the point in direction of pi that's
        # on the plane is... actually the closest point on the plane to pi is
        # pi - (pi.(pi-pj))/(pi-pj).(pi-pj) * (pi-pj)
        # But we want the FARTHEST point in the cell from pi.
        
        # Let me just use a grid-based estimate for the inradius as a differentiable surrogate.
        # Actually, for optimization purposes, let's use a differentiable proxy:
        # For each pair (i,j), compute the angle from pi to the point on the bisector
        # that is "in the direction" of the cell. The relevant point is the one where
        # the bisector plane is tangent to the cap centered at pi.
        
        # Simpler differentiable surrogate: for each i, inradius_i = max_j f(pi, pj)
        # where f is a smooth function. Let's use:
        # f(pi, pj) = angle between pi and the point on the bisector plane in the 
        # direction of pj from pi. The bisector plane normal is (pi-pj).
        # The point on the plane in the "pj direction" from pi:
        # Project pj onto the plane: pj - (pj.(pi-pj))/(pi-pj).(pi-pj)*(pi-pj)
        # Then normalize and compute angle from pi.
        
        # Let's compute this for all j
        pi_pj = di  # pi - pj
        pj = pts[valid_j]
        
        # Project pj onto bisector plane
        dot_pj_normal = np.sum(pj * pi_pj, axis=1)  # (m,)
        norm2 = np.sum(pi_pj * pi_pj, axis=1)  # (m,)
        norm2[norm2 < 1e-12] = 1e-12
        proj = pj - (dot_pj_normal / norm2)[:, None] * pi_pj  # (m, 3)
        proj_norm = np.linalg.norm(proj, axis=1, keepdims=True)
        proj_norm[proj_norm < 1e-12] = 1e-12
        proj = proj / proj_norm
        
        # Angle from pi to proj
        cos_angles = np.sum(pi[None, :] * proj, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        
        inradii[i] = np.max(angles)
    
    return inradii

def _covering_radius(pts):
    """Compute the actual covering radius by sampling the sphere."""
    # Use a fine grid
    n_theta = 360
    n_phi = 180
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    samples = _normalize(samples)
    
    # For each sample, find nearest point
    # Compute angles: cos(angle) = sample . point
    # Vectorized: for each sample, max dot with all points
    dots = samples @ pts.T  # (n_samples, N)
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    return np.degrees(np.max(angles))

def _objective(params):
    """Objective: sum of squared inradii (differentiable surrogate)."""
    pts = _normalize(params.reshape(N, 3))
    inradii = _spherical_voronoi_inradii(pts)
    return np.sum(inradii ** 2)

def _objective_and_grad(params):
    """Objective and gradient via finite differences."""
    pts = _normalize(params.reshape(N, 3))
    inradii = _spherical_voronoi_inradii(pts)
    obj = np.sum(inradii ** 2)
    
    # Finite difference gradient
    eps = 1e-5
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        obj_plus = _objective(params_plus)
        params_minus = params.copy()
        params_minus[i] -= eps
        obj_minus = _objective(params_minus)
        grad[i] = (obj_plus - obj_minus) / (2 * eps)
    
    return obj, grad

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    params0 = pts0.ravel()
    
    best_params = params0.copy()
    best_obj = _objective(params0)
    
    # Try L-BFGS-B with finite differences
    try:
        result = scipy.optimize.minimize(
            _objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        if _time_left() > 10:
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
    except Exception:
        pass
    
    # Additional local refinement with smaller steps
    if _time_left() > 20:
        try:
            result2 = scipy.optimize.minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 300}
            )
            if _time_left() > 5:
                if result2.fun < best_obj:
                    best_obj = result2.fun
                    best_params = result2.x.copy()
        except Exception:
            pass
    
    # Final polish: coordinate descent / local moves
    if _time_left() > 15:
        pts = _normalize(best_params.reshape(N, 3))
        for _iter in range(200):
            if _time_left() < 5:
                break
            improved = False
            for i in range(N):
                if _time_left() < 5:
                    break
                # Try small random perturbations
                for _ in range(10):
                    if _time_left() < 5:
                        break
                    perturb = np.random.randn(3) * 0.01
                    new_pts = pts.copy()
                    new_pts[i] = _normalize(new_pts[i] + perturb)[0]
                    new_params = new_pts.ravel()
                    new_obj = _objective(new_params)
                    if new_obj < best_obj - 1e-10:
                        best_obj = new_obj
                        best_params = new_params
                        pts = new_pts
                        improved = True
                        break
            if not improved:
                break
    
    final_pts = _normalize(best_params.reshape(N, 3))
    return final_pts
# EVOLVE-BLOCK-END

