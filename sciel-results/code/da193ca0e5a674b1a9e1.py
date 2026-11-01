import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _circumcenter(p1, p2, p3):
    """
    Compute the circumcenter of a spherical triangle defined by unit vectors p1, p2, p3.
    Returns the unit vector pointing to the circumcenter.
    """
    # The circumcenter is in the direction of p1 + p2 + p3 if the triangle is small and acute.
    # More robustly, it is the intersection of the perpendicular bisectors.
    # On the sphere, the circumcenter C satisfies C.p1 = C.p2 = C.p3.
    # This means C is orthogonal to (p1-p2) and (p2-p3).
    # So C is parallel to cross(p1-p2, p2-p3).
    
    v1 = p1 - p2
    v2 = p2 - p3
    c = np.cross(v1, v2)
    n = np.linalg.norm(c)
    if n < 1e-10:
        # Degenerate triangle, return midpoint or average
        return _normalize(p1 + p2 + p3)
    c = c / n
    
    # Determine sign. The circumcenter should be on the same side of the triangle as the "center".
    # We can check the dot product with the average of the vertices.
    avg = _normalize(p1 + p2 + p3)
    if np.dot(c, avg) < 0:
        c = -c
    return c

def _covering_radius_deg_hull(config):
    """
    Compute the exact covering radius in degrees using convex hull facets.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        # Fallback to sampling if hull fails (e.g., points not in general position)
        grid = _fib_sample(20000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.degrees(np.max(max_angle))
    
    max_angle = 0.0
    
    # Check vertices
    # The farthest point from a vertex is the vertex itself? No, the distance from the vertex to the nearest site is 0.
    # But the covering radius is the max over ALL points y.
    # The function f(y) = min_i d(y, p_i) is maximized at vertices of the Voronoi diagram (circumcenters) or at infinity (not on sphere).
    # Actually, the maxima of the distance to the nearest site on the sphere occur at the circumcenters of the Voronoi cells (which correspond to hull faces) or at the poles if the cells wrap around?
    # For a convex hull, the Voronoi cells on the sphere are the dual to the faces.
    # The farthest point in a Voronoi cell is the circumcenter of the corresponding face.
    
    for simplex in hull.simplices:
        pts = config[simplex]
        c = _circumcenter(pts[0], pts[1], pts[2])
        
        # Check if the circumcenter is "valid" (i.e., the nearest site is actually one of the face vertices)
        # If the circumcenter is far, it might be outside the angular triangle, in which case the max is at a vertex.
        # However, the formula d(c, p_i) for i in simplex gives the circumradius.
        # We need to check if the nearest site to c is one of the simplex vertices.
        dots = config @ c
        nearest_idx = np.argmax(dots)
        
        # The distance is arccos(dots[nearest_idx])
        angle = np.arccos(np.clip(dots[nearest_idx], -1.0, 1.0))
        
        if angle > max_angle:
            max_angle = angle
            
    return np.degrees(max_angle)

def _lloyd_step(config, hull):
    """
    Perform one Lloyd iteration.
    Moves each point towards the spherical mean of the circumcenters of the faces for which it is the nearest site.
    """
    new_config = config.copy()
    
    # Compute circumcenters for all faces
    circumcenters = []
    face_indices = []
    for simplex in hull.simplices:
        pts = config[simplex]
        c = _circumcenter(pts[0], pts[1], pts[2])
        circumcenters.append(c)
        face_indices.append(simplex)
        
    circumcenters = np.array(circumcenters)
    
    # For each circumcenter, find the nearest site in the config
    # Dots: (num_faces, N)
    dots = circumcenters @ config.T
    nearest_idx = np.argmax(dots, axis=1)
    
    # Accumulate targets for each site
    # We want to move site i to the mean of all circumcenters where nearest_idx == i
    sums = np.zeros((N, 3))
    counts = np.zeros(N)
    
    for i, idx in enumerate(nearest_idx):
        sums[idx] += circumcenters[i]
        counts[idx] += 1
        
    for i in range(N):
        if counts[i] > 0:
            target = sums[i]
            n = np.linalg.norm(target)
            if n > 1e-10:
                target = target / n
                # Blend with current position to ensure stability
                # new_pos = normalize(config[i] + target) ? Or just target?
                # Lloyd's usually replaces. Let's try a step size.
                # If we replace, it might oscillate. A step size of 1.0 is standard Lloyd.
                # Let's use a step size of 0.5 to be safe, or 1.0 and see.
                # Actually, for covering, moving to the centroid of the "worst points" is good.
                new_config[i] = target
            else:
                # If sum is zero (symmetric cancellation), keep old position
                new_config[i] = config[i]
        else:
            # If no face is assigned to this site (isolated point), keep it or nudge it
            # This can happen if a point is inside the convex hull of others? 
            # But points are on the sphere, so they are vertices of the hull unless coincident.
            # If coincident, we should perturb.
            new_config[i] = config[i]
            
    return new_config

def _local_optimize_lloyd(config, deadline):
    """
    Optimize using Lloyd's algorithm.
    """
    config = config.copy()
    best_config = config.copy()
    best_score = _covering_radius_deg_hull(config)
    
    # To avoid getting stuck in local minima or oscillating, we can add a small random perturbation
    # or use simulated annealing.
    # Let's start with simple Lloyd.
    
    # We need to handle the case where the hull changes.
    # Recalculating hull every iteration is O(N log N) or O(N^2) depending on implementation.
    # For N=50, it's very fast.
    
    it = 0
    while time.time() < deadline - 5.0:
        try:
            hull = ConvexHull(config)
        except Exception:
            # If hull fails, maybe points are too close or coplanar.
            # Add small noise
            config += 1e-5 * np.random.randn(N, 3)
            config = np.array([_normalize(p) for p in config])
            continue
            
        new_config = _lloyd_step(config, hull)
        
        # Check for improvement
        new_score = _covering_radius_deg_hull(new_config)
        
        if new_score < best_score - 1e-10:
            best_score = new_score
            best_config = new_config.copy()
            config = new_config
            # Reset counter if we want to detect stagnation
        else:
            # If no improvement, we might be in a local minimum or oscillating.
            # Add a small perturbation to escape.
            # We can also try to move points randomly.
            # Let's try a random move for a random point.
            idx = np.random.randint(N)
            # Generate a random direction
            rand_dir = np.random.randn(3)
            rand_dir = _normalize(rand_dir)
            
            # Move point slightly
            new_pt = config[idx] + 0.1 * rand_dir
            new_pt = _normalize(new_pt)
            
            old_pt = config[idx].copy()
            config[idx] = new_pt
            
            # Evaluate
            try:
                hull2 = ConvexHull(config)
                perturbed_score = _covering_radius_deg_hull(config)
                if perturbed_score < best_score:
                    best_score = perturbed_score
                    best_config = config.copy()
                else:
                    config[idx] = old_pt
            except Exception:
                config[idx] = old_pt
        
        it += 1
        # Limit iterations to prevent infinite loops if stuck
        if it > 2000:
            break
            
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Start with Fibonacci spiral
    config = _fib_sample(N)
    
    # Run Lloyd optimization
    config = _local_optimize_lloyd(config, deadline)
    
    # Final normalization
    config = np.array([_normalize(p) for p in config])
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                # Perturb j
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config
