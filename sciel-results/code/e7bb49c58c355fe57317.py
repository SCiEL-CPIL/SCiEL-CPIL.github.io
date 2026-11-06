"""Combined program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
import math

N = 130
DEADLINE = 125.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _facet_circumradius(pts, simplex):
    """Compute circumradius of a facet (triangle) on the unit sphere."""
    verts = pts[simplex]
    if len(verts) < 3:
        return 0.0
    v0, v1, v2 = verts[0], verts[1], verts[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-15:
        return 0.0
    normal = normal / norm_len
    
    # The circumcenter is along ±normal
    # Check which sign gives equidistant point
    for sign in [1.0, -1.0]:
        x = sign * normal
        da = np.dot(x, v0)
        db = np.dot(x, v1)
        dc = np.dot(x, v2)
        if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
            # Circumradius is arccos(da)
            return math.acos(max(-1.0, min(1.0, da)))
    # Fallback: use max angle to any vertex
    max_ang = 0.0
    for v in verts:
        cos_ang = max(-1.0, min(1.0, np.dot(normal, v)))
        ang = math.acos(cos_ang)
        if ang > max_ang:
            max_ang = ang
    return max_ang

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_rad = 0.0
    for simplex in hull.simplices:
        rad = _facet_circumradius(pts, simplex)
        if rad > max_rad:
            max_rad = rad
    
    return math.degrees(max_rad)

def _get_worst_facet(pts):
    """Find the facet with the largest circumradius. Returns (index, circumradius)."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return None, 0.0
    
    max_rad = 0.0
    worst_idx = None
    for i, simplex in enumerate(hull.simplices):
        rad = _facet_circumradius(pts, simplex)
        if rad > max_rad:
            max_rad = rad
            worst_idx = i
    
    return worst_idx, max_rad

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _soft_max_loss(P_flat, samples, lam=10.0):
    """
    Soft-max covering loss: sum over samples of exp(lam * min_dist_to_points).
    Returns (loss, grad_flat).
    P_flat: shape (3N,) raw (unnormalized) vectors.
    samples: (S, 3) unit vectors.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms  # normalized points on sphere

    # Compute angular distances: (S, N)
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)

    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)

    loss = np.sum(np.exp(lam * min_dists))

    # Gradient
    weights = lam * np.exp(lam * min_dists)  # (S,)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom  # (S,)

    grad_P = np.zeros_like(Pn)
    np.add.at(grad_P, nearest, grad_factor[:, None] * samples)

    # Project to tangent
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn

    # Back-prop through normalization
    grad_flat = (grad_tan / norms).flatten()

    return loss, grad_flat

def _local_search_hull(P, max_iters=500, deadline=None):
    """
    Hull-based local search: repeatedly find the worst facet and move its
    vertices toward the facet's circumcenter.
    """
    rng = np.random.default_rng(42)
    
    # Initial worst facet
    worst_idx, current_max = _get_worst_facet(P)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if worst_idx is None:
            break
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(P)
        except Exception:
            break
        
        if worst_idx >= len(hull.simplices):
            worst_idx, current_max = _get_worst_facet(P)
            continue
        
        simplex = hull.simplices[worst_idx]
        verts = P[simplex]
        
        # Compute facet normal
        v0, v1, v2 = verts[0], verts[1], verts[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            break
        normal = normal / norm_len
        
        # Find circumcenter direction
        circum_dir = None
        for sign in [1.0, -1.0]:
            x = sign * normal
            da = np.dot(x, v0)
            db = np.dot(x, v1)
            dc = np.dot(x, v2)
            if abs(da - db) < 1e-6 and abs(da - dc) < 1e-6:
                circum_dir = x
                break
        
        if circum_dir is None:
            circum_dir = normal
        
        # Move each vertex slightly toward the circumcenter
        improved = False
        for idx in simplex:
            p_old = P[idx].copy()
            # Direction from p_old toward circum_dir
            proj = np.dot(p_old, circum_dir)
            grad_dir = circum_dir - proj * p_old
            grad_norm = np.linalg.norm(grad_dir)
            
            if grad_norm < 1e-8:
                continue
            
            grad_dir = grad_dir / grad_norm
            
            # Try several step sizes
            best_step = 0.0
            best_max = current_max
            best_p = None
            
            for step in [0.10, 0.05, 0.02, 0.01]:
                new_p = p_old + step * grad_dir
                new_p = _normalize(new_p.reshape(1, 3))[0]
                P[idx] = new_p
                
                new_worst_idx, new_max = _get_worst_facet(P)
                
                if new_max < best_max:
                    best_max = new_max
                    best_step = step
                    best_p = new_p.copy()
                    best_worst_idx = new_worst_idx
                else:
                    P[idx] = p_old
            
            if best_p is not None:
                P[idx] = best_p
                current_max = best_max
                worst_idx = best_worst_idx
                improved = True
        
        if not improved:
            # Stuck; try a small random perturbation on the worst facet
            for idx in simplex:
                p_old = P[idx].copy()
                rand = rng.standard_normal(3)
                rand = rand - np.dot(rand, p_old) * p_old
                rand_norm = np.linalg.norm(rand)
                if rand_norm > 1e-8:
                    rand /= rand_norm
                P[idx] = _normalize((p_old + 0.03 * rand).reshape(1, 3))[0]
            
            worst_idx, current_max = _get_worst_facet(P)
    
    return P, current_max

def _repulsion_refine(P, iterations=50, deadline=None):
    """
    Repulsion-based refinement: add a repulsive force between nearby points
    to encourage better spreading.
    """
    rng = np.random.default_rng(42)
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute repulsive forces
        dots = P @ P.T
        np.fill_diagonal(dots, 1.0)
        
        # Repulsion strength: higher for closer points
        repulsion = np.zeros_like(dots)
        close_mask = dots < 0.95  # points within ~18 degrees
        repulsion[close_mask] = (0.95 - dots[close_mask]) ** 2
        
        # Net force on each point
        forces = np.zeros_like(P)
        for i in range(N):
            for j in range(N):
                if i != j:
                    forces[i] += repulsion[i, j] * (P[j] - P[i])
        
        # Project forces to tangent plane
        for i in range(N):
            proj = np.dot(forces[i], P[i])
            forces[i] -= proj * P[i]
        
        # Apply small step
        step = 0.005
        for i in range(N):
            new_p = P[i] + step * forces[i]
            P[i] = _normalize(new_p.reshape(1, 3))[0]
        
        # Normalize again
        P = _normalize(P)
    
    return P

def _multi_start_optimize(start_pts, deadline):
    """
    Run L-BFGS optimization from a starting configuration.
    """
    S_opt = 10000
    samples_opt = _fibonacci_sphere(S_opt)
    
    time_check = [False]
    def time_callback(xk):
        if time.time() > deadline - 30.0:
            time_check[0] = True
            raise ValueError("Time limit approaching")
    
    try:
        res = minimize(
            lambda x: _soft_max_loss(x, samples_opt, lam=10.0),
            start_pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
            callback=time_callback
        )
        if res.x is not None:
            return _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass
    
    return start_pts

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin (limit is 150s)
    
    # 1. Initialize with Fibonacci sphere + small noise
    P = _fibonacci_sphere(N, seed_noise=0.02, seed=42)
    
    # 2. L-BFGS optimization with soft-max loss
    P = _multi_start_optimize(P, deadline)
    
    # 3. Second optimization round with more samples
    if time.time() < deadline - 60.0:
        S_opt2 = 20000
        samples_opt2 = _fibonacci_sphere(S_opt2)
        
        time_check2 = [False]
        def time_callback2(xk):
            if time.time() > deadline - 40.0:
                time_check2[0] = True
                raise ValueError("Time limit approaching")
        
        try:
            res2 = minimize(
                lambda x: _soft_max_loss(x, samples_opt2, lam=12.0),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
                callback=time_callback2
            )
            if res2.x is not None:
                P = _normalize(res2.x.reshape(-1, 3))
        except Exception:
            pass
    
    # 4. Repulsion refinement
    if time.time() < deadline - 30.0:
        P = _repulsion_refine(P, iterations=30, deadline=deadline - 25.0)
    
    # 5. Hull-based local search
    if time.time() < deadline - 10.0:
        P, _ = _local_search_hull(P, max_iters=300, deadline=deadline - 5.0)
    
    # 6. Final repulsion refinement
    if time.time() < deadline - 3.0:
        P = _repulsion_refine(P, iterations=20, deadline=deadline - 1.0)
    
    # Final normalization
    P = _normalize(P)
    
    return P

# EVOLVE-BLOCK-END
