"""Covering the sphere with 50 equal caps - optimized solution."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(N, rng):
    """Generate N random points on the unit sphere."""
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_fibonacci(N, rng, noise_scale=0.1):
    """Generate a perturbed Fibonacci spiral."""
    base = _fibonacci_spiral(N)
    noise = rng.randn(N, 3) * noise_scale
    pts = base + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _cover_radius(pts):
    """
    Compute the covering radius using the convex hull facets.
    The covering radius is the maximum circumradius over all facets.
    """
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    radii = []
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Spherical circumcenter
        n = np.cross(a - b, a - c)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            centroid = a + b + c
            norm_c = np.linalg.norm(centroid)
            if norm_c < 1e-12:
                continue
            cc = centroid / norm_c
        else:
            p1 = n / norm_n
            p2 = -p1
            
            min_dot1 = min(np.dot(p1, a), np.dot(p1, b), np.dot(p1, c))
            min_dot2 = min(np.dot(p2, a), np.dot(p2, b), np.dot(p2, c))
            
            if min_dot1 <= min_dot2:
                cc = p1
            else:
                cc = p2
        
        # Angular distance from circumcenter to nearest vertex
        dots = np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)])
        min_dot = np.min(dots)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        radii.append(np.arccos(min_dot))
    
    if not radii:
        return np.pi
    
    return np.max(radii)


def _cover_radius_fast(pts):
    """
    Fast covering radius estimate using only the maximum of pairwise distances
    as a proxy, plus a local check. This is an approximation but much faster.
    Actually, let's use a better approach: compute the convex hull and check
    only the facets with largest circumradius.
    """
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    # For speed, compute circumradius for all facets but vectorized
    simplices = hull.simplices
    n_facets = len(simplices)
    
    # Get all vertex triples
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Compute normals
    normals = np.cross(a - b, a - c)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Handle degenerate cases
    valid = norms[:, 0] > 1e-12
    if not np.all(valid):
        # Fall back to slower method for degenerate cases
        return _cover_radius(pts)
    
    normals = normals / norms
    
    # Two possible circumcenters
    p1 = normals
    p2 = -normals
    
    # Compute min dot product for each
    dots1 = np.minimum(np.minimum(np.einsum('ij,ij->i', p1, a), 
                                   np.einsum('ij,ij->i', p1, b)),
                        np.einsum('ij,ij->i', p1, c))
    dots2 = np.minimum(np.minimum(np.einsum('ij,ij->i', p2, a),
                                   np.einsum('ij,ij->i', p2, b)),
                        np.einsum('ij,ij->i', p2, c))
    
    # Choose the one with smaller min dot (larger angle)
    use_p1 = dots1 <= dots2
    min_dots = np.where(use_p1, dots1, dots2)
    
    # Clip and compute angles
    min_dots = np.clip(min_dots, -1.0, 1.0)
    radii = np.arccos(min_dots)
    
    return np.max(radii)


def _local_optimize(pts, max_iters=500, initial_step=0.01, time_budget=30.0):
    """
    Local optimization using gradient-like descent on the covering radius.
    Move points away from the circumcenters of critical facets.
    """
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius_fast(pts)
    no_improve_count = 0
    start_time = time.time()
    
    for it in range(max_iters):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        simplices = hull.simplices
        n_facets = len(simplices)
        
        # Get all vertex triples
        a = pts[simplices[:, 0]]
        b = pts[simplices[:, 1]]
        c = pts[simplices[:, 2]]
        
        # Compute normals
        normals = np.cross(a - b, a - c)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        valid = norms[:, 0] > 1e-12
        if not np.all(valid):
            break
        
        normals = normals / norms
        
        # Two possible circumcenters
        p1 = normals
        p2 = -normals
        
        # Compute min dot product for each
        dots1 = np.minimum(np.minimum(np.einsum('ij,ij->i', p1, a), 
                                       np.einsum('ij,ij->i', p1, b)),
                            np.einsum('ij,ij->i', p1, c))
        dots2 = np.minimum(np.minimum(np.einsum('ij,ij->i', p2, a),
                                       np.einsum('ij,ij->i', p2, b)),
                            np.einsum('ij,ij->i', p2, c))
        
        use_p1 = dots1 <= dots2
        min_dots = np.where(use_p1, dots1, dots2)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        radii = np.arccos(min_dots)
        
        # Find the maximum radius facet
        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        
        # Get the circumcenter of the critical facet
        cc = p1[max_idx] if use_p1[max_idx] else p2[max_idx]
        
        # Get the three vertices of the critical facet
        critical_simplex = simplices[max_idx]
        
        # For each vertex of the critical facet, try moving it to reduce the radius
        improved = False
        for idx in critical_simplex:
            v = pts[idx]
            
            # Direction from circumcenter to vertex (push vertex away from cc)
            dir_vec = v - cc
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-15:
                continue
            dir_vec = dir_vec / dir_norm
            
            # Try a small step in this direction
            new_pts = pts.copy()
            new_pts[idx] = v + step * dir_vec
            new_pts[idx] /= np.linalg.norm(new_pts[idx])
            
            new_radius = _cover_radius_fast(new_pts)
            
            if new_radius < best_radius - 1e-10:
                pts = new_pts
                best_radius = new_radius
                improved = True
                break
        
        if not improved:
            # Try smaller steps
            step *= 0.8
            if step < 1e-8:
                break
            no_improve_count += 1
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0
            # Slowly increase step if making progress
            if it % 20 == 0 and step < initial_step:
                step = min(step * 1.15, initial_step)
    
    return pts, best_radius


def _force_based_optimize(pts, max_iters=300, time_budget=30.0):
    """
    Force-based optimization: points repel each other based on their
    contribution to the covering radius. Points in dense areas are pushed
    toward sparse areas.
    """
    pts = pts.copy()
    start_time = time.time()
    best_radius = _cover_radius_fast(pts)
    best_pts = pts.copy()
    
    # Learning rate schedule
    base_lr = 0.05
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        lr = base_lr * (1.0 - 0.5 * it / max_iters)
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        simplices = hull.simplices
        n_facets = len(simplices)
        
        # Compute circumradius for each facet
        a = pts[simplices[:, 0]]
        b = pts[simplices[:, 1]]
        c = pts[simplices[:, 2]]
        
        normals = np.cross(a - b, a - c)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        valid = norms[:, 0] > 1e-12
        if not np.all(valid):
            break
        
        normals = normals / norms
        
        p1 = normals
        p2 = -normals
        
        dots1 = np.minimum(np.minimum(np.einsum('ij,ij->i', p1, a), 
                                       np.einsum('ij,ij->i', p1, b)),
                            np.einsum('ij,ij->i', p1, c))
        dots2 = np.minimum(np.minimum(np.einsum('ij,ij->i', p2, a),
                                       np.einsum('ij,ij->i', p2, b)),
                            np.einsum('ij,ij->i', p2, c))
        
        use_p1 = dots1 <= dots2
        min_dots = np.where(use_p1, dots1, dots2)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        radii = np.arccos(min_dots)
        
        # Compute forces: each point is pushed away from the circumcenters
        # of facets it belongs to, weighted by the facet's radius
        forces = np.zeros_like(pts)
        
        for i, simplex in enumerate(simplices):
            weight = radii[i]  # Larger radius facets get more weight
            
            cc = p1[i] if use_p1[i] else p2[i]
            
            for idx in simplex:
                v = pts[idx]
                dir_vec = v - cc
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-15:
                    forces[idx] += weight * dir_vec / dir_norm
        
        # Normalize forces and apply
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        valid_forces = force_norms > 1e-15
        forces[valid_forces] = forces[valid_forces] / force_norms[valid_forces]
        
        # Apply forces
        new_pts = pts + lr * forces
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius = _cover_radius_fast(new_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If no improvement, reduce learning rate
            base_lr *= 0.95
            if base_lr < 1e-4:
                break
    
    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius_fast(best_pts)
    
    # Phase 1: Force-based optimization from Fibonacci start
    if time.time() < deadline - 60:
        time_budget = min(40.0, deadline - 60 - (time.time() - start_time))
        optimized, opt_radius = _force_based_optimize(pts, max_iters=500, time_budget=time_budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 2: Local optimization on the best configuration
    if time.time() < deadline - 30:
        time_budget = min(25.0, deadline - 30 - (time.time() - start_time))
        optimized, opt_radius = _local_optimize(best_pts, max_iters=1000, initial_step=0.01, time_budget=time_budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 3: Multiple random restarts with force-based + local optimization
    rng = np.random.RandomState(42)
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate different initial configurations
        if restart % 4 == 0:
            random_pts = _perturbed_fibonacci(N, rng, noise_scale=0.05 + 0.01 * restart)
        elif restart % 4 == 1:
            random_pts = _random_sphere_points(N, rng)
        elif restart % 4 == 2:
            # Fibonacci with different phase
            i = np.arange(N) + 0.5 + restart * 0.1
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + restart * 0.3
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            random_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Scaled Fibonacci
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i * (1.0 + 0.1 * restart)
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            random_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        remaining = deadline - 20 - (time.time() - start_time)
        if remaining < 15:
            break
        
        # Time budget for this restart
        time_per_restart = remaining / (max_restarts - restart)
        
        # Force-based optimization
        if time_per_restart > 10:
            force_budget = min(time_per_restart * 0.5, 15.0)
            optimized, opt_radius = _force_based_optimize(random_pts, max_iters=200, time_budget=force_budget)
        else:
            optimized, opt_radius = random_pts, _cover_radius_fast(random_pts)
        
        # Local optimization if time remains
        if time.time() < deadline - 20:
            local_budget = min(deadline - 20 - (time.time() - start_time), 10.0)
            if local_budget > 5:
                optimized2, opt_radius2 = _local_optimize(optimized, max_iters=300, initial_step=0.005, time_budget=local_budget)
                if opt_radius2 < opt_radius:
                    optimized = optimized2
                    opt_radius = opt_radius2
        
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final refinement: more iterations on the best configuration
    if time.time() < deadline - 5:
        time_budget = deadline - 5 - (time.time() - start_time)
        if time_budget > 5:
            optimized, opt_radius = _local_optimize(best_pts, max_iters=2000, initial_step=0.003, time_budget=time_budget)
            if opt_radius < best_radius:
                best_pts = optimized
                best_radius = opt_radius
    
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    # Normalize to ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
