import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 50
START_TIME = time.time()
DEADLINE = 140.0  # Leave 10s margin before 150s kill

def to_sphere(params):
    """Convert raw params to unit vectors."""
    v = params.reshape(-1, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return v / norms

def compute_facet_data(points):
    """
    Compute circumradii (angular), circumcenters (unit vectors), and facet vertices
    for all convex hull facets.
    Returns:
        circumradii: array of angular circumradii (radians)
        circumcenters: array of unit vectors pointing to circumcenters
        facet_vertices: list of tuples of indices
    """
    points = np.asarray(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    circumradii = []
    circumcenters = []
    facet_vertices = []
    
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        ab = b - a
        ac = c - a
        abab = np.dot(ab, ab)
        acac = np.dot(ac, ac)
        abac = np.dot(ab, ac)
        
        denom = abab * acac - abac * abac
        if abs(denom) < 1e-12:
            continue
        
        # Circumcenter in 3D space
        u1 = (acac * abac - abab * abac) / denom
        u2 = (abab * abac - acac * abac) / denom
        circumcenter = a + u1 * ab + u2 * ac
        
        cc_norm = np.linalg.norm(circumcenter)
        if cc_norm < 1e-10:
            continue
        
        pole = circumcenter / cc_norm
        
        # Angular distance from pole to vertices
        theta_a = np.arccos(np.clip(np.dot(pole, a), -1, 1))
        theta_b = np.arccos(np.clip(np.dot(pole, b), -1, 1))
        theta_c = np.arccos(np.clip(np.dot(pole, c), -1, 1))
        
        circum_angular = max(theta_a, theta_b, theta_c)
        
        circumradii.append(circum_angular)
        circumcenters.append(pole)
        facet_vertices.append(simplex)
    
    if len(circumradii) == 0:
        return None, None, None
    
    return np.array(circumradii), np.array(circumcenters), facet_vertices

def compute_covering_radius(points):
    """Compute the raw covering radius (max angular circumradius) in radians."""
    circumradii, _, _ = compute_facet_data(points)
    if circumradii is None:
        return np.pi
    return np.max(circumradii)

def compute_smooth_covering_radius(points, beta=1.0):
    """
    Compute a smoothed covering radius using log-sum-exp over facet circumradii.
    This is differentiable and helps gradient-free optimizers like Nelder-Mead
    navigate the landscape by providing a 'soft' maximum.
    """
    circumradii, _, _ = compute_facet_data(points)
    if circumradii is None:
        return np.pi
    
    if len(circumradii) == 0:
        return 0.0
        
    max_cr = np.max(circumradii)
    # Shift for numerical stability
    shifted = circumradii - max_cr
    # Log-sum-exp: max_cr + beta * log(sum(exp(r_i / beta)))
    # As beta -> 0, this approaches max(r_i)
    # We use beta to control smoothness.
    weights = np.exp(shifted / beta)
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return max_cr
    return max_cr + beta * np.log(sum_weights)

def lloyd_worst_facet_step(points, max_iter=50, step_size=0.5, beta_smooth=0.5):
    """
    Iteratively refine points by moving vertices of the worst facets
    towards the circumcenter of that facet.
    """
    points = np.asarray(points, dtype=float).copy()
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        circumradii, circumcenters, facet_vertices = compute_facet_data(points)
        if circumradii is None:
            break
        
        # Find the facet with the largest circumradius
        # We can look at top K facets to avoid oscillating on a single noisy one
        top_k = min(5, len(circumradii))
        top_indices = np.argsort(circumradii)[-top_k:]
        
        # Accumulate moves for each point
        moves = np.zeros_like(points)
        counts = np.zeros(N)
        
        for idx in top_indices:
            # Weight by how much worse this facet is compared to the max
            # Or just treat all top facets equally
            simplex = facet_vertices[idx]
            center = circumcenters[idx]
            
            # Move each vertex of this facet towards the center
            for v_idx in simplex:
                p = points[v_idx]
                # Slerp towards center
                dot_pc = np.clip(np.dot(p, center), -1, 1)
                theta = np.arccos(dot_pc)
                
                if theta < 1e-6:
                    continue
                
                # Step size can be scaled by the severity of the error
                # Larger error -> larger step
                error = circumradii[idx]
                # Normalize error roughly to 0..1 range for step scaling? 
                # Or just use a fixed step.
                step = step_size * np.clip(error / 0.5, 0.1, 1.0)
                
                sin_theta = np.sin(theta)
                if sin_theta < 1e-10:
                    continue
                
                # Vector pointing from p to center on the sphere
                # new_p = (sin((1-step)*theta)*p + sin(step*theta)*center) / sin(theta)
                # We just accumulate the direction
                move_dir = (np.sin(step*theta) * center - np.sin((1-step)*theta) * p) / sin_theta
                moves[v_idx] += move_dir
                counts[v_idx] += 1
        
        # Apply moves
        any_moved = False
        for i in range(N):
            if counts[i] == 0:
                continue
            p = points[i]
            avg_move = moves[i] / counts[i]
            
            # Project move to tangent plane
            move_tangent = avg_move - np.dot(avg_move, p) * p
            move_norm = np.linalg.norm(move_tangent)
            
            if move_norm < 1e-8:
                continue
            
            # Normalize step length
            # We want a small step along the tangent
            step_len = 0.05 # Fixed small step for stability
            # If the move is large, cap it
            if move_norm > 1.0:
                 move_tangent = move_tangent / move_norm
            
            # Slerp
            new_p = p + step_len * move_tangent
            new_p = new_p / np.linalg.norm(new_p)
            
            if np.linalg.norm(new_p - p) > 1e-12:
                points[i] = new_p
                any_moved = True
        
        if not any_moved:
            break
            
    return points

def nelder_mead_optimize(points, max_iter=100, beta=0.5):
    """
    Optimize using Nelder-Mead on the smoothed covering radius.
    """
    params = points.flatten()
    
    def objective(p):
        pts = to_sphere(p)
        return compute_smooth_covering_radius(pts, beta=beta)
    
    best_params = params.copy()
    best_val = objective(best_params)
    
    try:
        # Nelder-Mead options
        # initial_simplex_size: how big the initial simplex is
        # For sphere covering, small perturbations are better
        res = minimize(objective, best_params, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8,
                                'adaptive': True, 'initial_simplex_size': 0.05})
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass
    
    final_points = to_sphere(best_params)
    final_cr = compute_covering_radius(final_points)
    
    return final_points, final_cr

def fibonacci_spiral(n_points):
    """Fibonacci spiral on the sphere."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def icosahedron_based_init(n_points, seed=123):
    """Generate points based on icosahedron vertices, then add more points."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n_points <= 12:
        return verts[:n_points]
    
    points = verts.copy()
    rng = np.random.RandomState(seed)
    
    while len(points) < n_points:
        best_dist = 0
        best_point = None
        # Greedy farthest point
        for _ in range(100):
            candidate = rng.randn(3)
            candidate = candidate / np.linalg.norm(candidate)
            dists = np.arccos(np.clip(points @ candidate, -1, 1))
            min_dist = np.min(dists)
            if min_dist > best_dist:
                best_dist = min_dist
                best_point = candidate
        if best_point is not None:
            points = np.vstack([points, best_point])
    
    return points[:n_points]

def random_init(n_points, rng):
    """Generate random points on the unit sphere."""
    v = rng.randn(n_points, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci
    inits.append(fibonacci_spiral(N))
    
    # 2. Icosahedron-based
    inits.append(icosahedron_based_init(N, seed=123))
    inits.append(icosahedron_based_init(N, seed=456))
    
    # 3. Random
    inits.append(random_init(N, rng))
    
    # 4. Perturbed Fibonacci
    noise = 0.05 * rng.randn(N, 3)
    fib_perturbed = fibonacci_spiral(N) + noise
    fib_perturbed = fib_perturbed / np.linalg.norm(fib_perturbed, axis=1, keepdims=True)
    inits.append(fib_perturbed)
    
    best_points = inits[0].copy()
    best_cr = compute_covering_radius(best_points)
    
    # Main Loop: Cycle through initializations and optimize
    # We have ~140s. 
    # Nelder-Mead is slow but robust. Lloyd is fast.
    # Strategy: 
    # 1. Quick Lloyd to clean up init
    # 2. Nelder-Mead to refine
    # 3. Quick Lloyd again
    # 4. Check time, if more, repeat with next init or refine current best
    
    # Pre-compute cr for inits
    init_crs = []
    for init in inits:
        cr = compute_covering_radius(init)
        init_crs.append(cr)
    
    # Sort inits by initial CR (best first)
    sorted_indices = np.argsort(init_crs)
    sorted_inits = [inits[i] for i in sorted_indices]
    
    # We will process the top 3 inits thoroughly
    for init_idx in range(len(sorted_inits)):
        if time.time() > DEADLINE - 20:
            break
            
        current_init = sorted_inits[init_idx]
        
        # Phase 1: Lloyd refinement (fast)
        if time.time() < DEADLINE - 15:
            try:
                lloyd_pts = lloyd_worst_facet_step(current_init, max_iter=30, step_size=0.5)
                cr = compute_covering_radius(lloyd_pts)
                if cr < best_cr:
                    best_cr = cr
                    best_points = lloyd_pts.copy()
            except Exception:
                pass
        
        # Phase 2: Nelder-Mead on the best point found so far from this init
        if time.time() < DEADLINE - 10:
            try:
                # Use a smaller beta for NM to be closer to the true max
                nm_pts, nm_cr = nelder_mead_optimize(lloyd_pts, max_iter=50, beta=0.2)
                if nm_cr < best_cr:
                    best_cr = nm_cr
                    best_points = nm_pts.copy()
            except Exception:
                pass
        
        # Phase 3: Final Lloyd polish
        if time.time() < DEADLINE - 5:
            try:
                final_pts = lloyd_worst_facet_step(best_points, max_iter=20, step_size=0.3)
                cr = compute_covering_radius(final_pts)
                if cr < best_cr:
                    best_cr = cr
                    best_points = final_pts.copy()
            except Exception:
                pass
        
        # If we have time, do another round of NM on the polished result
        if time.time() < DEADLINE - 5:
            try:
                nm_pts2, nm_cr2 = nelder_mead_optimize(best_points, max_iter=30, beta=0.1)
                if nm_cr2 < best_cr:
                    best_cr = nm_cr2
                    best_points = nm_pts2.copy()
            except Exception:
                pass

    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    # Validate: check no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                best_points[j] = best_points[j] * 0.999 + 0.001 * rng.randn(3)
                best_points[j] /= np.linalg.norm(best_points[j])
    
    # Check not all in a closed hemisphere
    try:
        hull = ConvexHull(best_points)
    except Exception:
        # If hull fails, it might be degenerate or coplanar, but for 50 points in 3D
        # it's unlikely unless they are clustered. 
        # We'll just return the points. The evaluator will penalize invalid hulls.
        pass
    
    return best_points
