import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n

def _normalize_batch(pts):
    """Normalize batch of vectors."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _geodesic_dome(n_points_target):
    """
    Generate points on the sphere using geodesic subdivision of an icosahedron.
    We start with an icosahedron (12 vertices) and subdivide faces until we have
    approximately n_points_target vertices.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces (indices into verts)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    current_verts = verts
    current_faces = faces
    
    # Subdivide until we have enough points
    # Each subdivision increases vertices by roughly 4x
    # 12 -> 42 -> 162 -> 642
    # We want ~50. 12 is too few. 42 is close. 162 is too many.
    # Let's do one subdivision (42 points) and then select/merge or just use 42?
    # The task requires exactly N=50. 
    # If we use 42, we are missing 8 points. 
    # If we use 162, we have too many.
    # Strategy: Subdivide once to get 42 points. Then add 8 more points optimally?
    # Or, better: Subdivide to get 42, then use a "fill" strategy for the remaining 8?
    # Actually, let's just subdivide once to get 42 points, and then randomly sample 8 more points 
    # in the largest empty spaces? Or just use the 42 points and pad with random?
    # A better approach for N=50: 
    # 1. Start with icosahedron (12).
    # 2. Subdivide to 42.
    # 3. We need 8 more. We can find the 8 largest facets and add their circumcenters?
    # This creates a specific non-uniform structure but ensures coverage.
    
    # Let's do 1 subdivision:
    # Midpoints of edges
    edge_mids = {}
    new_verts = list(current_verts)
    new_faces = []
    
    def get_midpoint(v1_idx, v2_idx):
        key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
        if key not in edge_mids:
            mid = (current_verts[v1_idx] + current_verts[v2_idx]) / 2.0
            mid = _normalize(mid)
            edge_mids[key] = len(new_verts)
            new_verts.append(mid)
        return edge_mids[key]
    
    for face in current_faces:
        v1, v2, v3 = face
        m12 = get_midpoint(v1, v2)
        m23 = get_midpoint(v2, v3)
        m31 = get_midpoint(v3, v1)
        new_faces.append([v1, m12, m31])
        new_faces.append([v2, m23, m12])
        new_faces.append([v3, m31, m23])
        new_faces.append([m12, m23, m31])
        
    current_verts = np.array(new_verts)
    current_faces = new_faces
    
    # Now we have 42 points. We need 50.
    # Find the 8 largest facets (by circumradius) and add their circumcenters.
    # This will "fill in" the largest holes.
    hull = ConvexHull(current_verts)
    simplices = hull.simplices
    
    # Calculate circumradii
    radii = []
    centers = []
    for s in simplices:
        p0, p1, p2 = current_verts[s]
        v1 = p1 - p0
        v2 = p2 - p0
        c = np.cross(v1, v2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            continue
        c = c / c_norm
        # Distance from center to p0
        r = np.arccos(np.clip(np.dot(c, p0), -1, 1))
        radii.append(r)
        centers.append(c)
        
    radii = np.array(radii)
    centers = np.array(centers)
    
    # Sort by radius descending
    idx_sorted = np.argsort(radii)[::-1]
    
    # Add the top 8 centers
    new_points = []
    for idx in idx_sorted[:8]:
        new_points.append(centers[idx])
        
    # Combine
    all_points = np.vstack([current_verts, np.array(new_points)])
    
    # We have 50 points.
    # Note: The added points might be very close to existing vertices if the mesh is coarse,
    # but since we picked the largest facets, they should be distinct.
    
    return all_points[:N]

def _get_circumcenters_and_radii(config):
    """
    Compute circumcenters of convex hull facets and their angular distance
    to the nearest configuration point.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return np.array([]), np.array([]), np.array([], dtype=int)
    
    simplices = hull.simplices
    p0 = config[simplices[:, 0]]
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    c = np.cross(v1, v2)
    c_norms = np.linalg.norm(c, axis=1)
    
    valid_mask = c_norms > 1e-12
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([], dtype=int)
        
    c = c[valid_mask]
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    
    # Compute angular distance from each circumcenter to the nearest config point
    dots = c @ config.T  # (F, N)
    dots = np.clip(dots, -1.0, 1.0)
    nearest_idx = np.argmax(dots, axis=1)
    nearest_dots = dots[np.arange(len(c)), nearest_idx]
    radii = np.arccos(nearest_dots)
    
    return c, radii, nearest_idx

def _covering_radius(config):
    """Compute the covering radius in radians."""
    c, radii, _ = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        # Fallback: use grid sampling
        grid = _fib_sample(10000)
        dots = grid @ config.T
        min_dots = np.min(dots, axis=1)
        max_angle = np.arccos(np.clip(min_dots, -1.0, 1.0))
        return np.max(max_angle)
    return np.max(radii)

def _fib_sample(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _geodesic_smooth_step(config, lambda_smooth=0.5):
    """
    Spherical Laplacian smoothing step.
    Moves each point towards the weighted average of its neighbors.
    Weights are based on the inverse of the circumradius of the shared facets.
    """
    try:
        hull = ConvexHull(config)
    except Exception:
        return config
    
    # Get edges from hull
    # hull.equations are planes, hull.simplices are faces.
    # We need adjacency.
    # scipy.spatial.ConvexHull has .simplices.
    # We can build an adjacency list from simplices.
    adj = {i: set() for i in range(N)}
    # Also store edge weights (inverse radius)
    edge_weights = {}
    
    # First, compute radii for all facets
    c, radii, _ = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        return config
        
    # Map facet index to radius
    # hull.simplices order corresponds to c, radii order (assuming valid_mask passed all)
    # Note: _get_circumcenters_and_radii filters valid facets. 
    # We need to be careful if some facets are invalid.
    # For simplicity, assume all are valid or handle mismatch.
    # Let's just use the hull.simplices directly and recompute radii for weighting if needed,
    # or assume the mapping is 1:1 if no degenerate facets.
    
    # Let's rebuild the mapping safely.
    simplices = hull.simplices
    # We need to map each simplex in hull.simplices to a radius.
    # _get_circumcenters_and_radii returns c, radii for *valid* facets.
    # Let's just compute radii again here to be safe and aligned with hull.simplices.
    
    p0 = config[simplices[:, 0]]
    p1 = config[simplices[:, 1]]
    p2 = config[simplices[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = np.cross(v1, v2)
    norms = np.linalg.norm(cross, axis=1)
    
    valid = norms > 1e-12
    radii_local = np.zeros(len(simplices))
    for i in range(len(simplices)):
        if valid[i]:
            c_vec = cross[i] / norms[i]
            # Radius is angle to nearest point? No, circumradius is angle to vertex.
            # For a triangle on sphere, circumradius R satisfies cos(R) = dot(c, p0).
            # But we want to weight by "badness". Larger R is worse.
            # So weight = 1 / R or 1 / (R + eps).
            r_val = np.arccos(np.clip(np.dot(c_vec, p0[i]), -1, 1))
            radii_local[i] = r_val
        else:
            radii_local[i] = np.pi # Max weight penalty? Or 0?
            # If degenerate, ignore or give low weight.
            
    # Assign weights to edges
    for i, s in enumerate(simplices):
        v0, v1, v2 = s
        w = 1.0 / (radii_local[i] + 1e-6)
        adj[v0].add(v1)
        adj[v1].add(v0)
        adj[v1].add(v2)
        adj[v2].add(v0)
        adj[v0].add(v2)
        
        # Store weight for edge (v0, v1)
        # We need to accumulate weights if an edge is shared by 2 facets.
        # For now, let's just use uniform weight 1.0 for simplicity in the first pass,
        # or average the weights of the two adjacent facets.
        # To keep it simple and robust:
        pass
    
    # Let's refine: Use uniform weights for neighbors first, 
    # but the directive says "weighted by the inverse of the local facet circumradius".
    # Let's accumulate weights per edge.
    edge_w = {}
    for i, s in enumerate(simplices):
        v0, v1, v2 = s
        w = 1.0 / (radii_local[i] + 1e-6)
        for u, v in [(v0, v1), (v1, v2), (v2, v0)]:
            key = (min(u, v), max(u, v))
            edge_w[key] = edge_w.get(key, 0.0) + w
            
    # Apply smoothing
    new_config = config.copy()
    for i in range(N):
        neighbors = list(adj[i])
        if not neighbors:
            continue
            
        weighted_sum = np.zeros(3)
        total_w = 0.0
        for j in neighbors:
            key = (min(i, j), max(i, j))
            w = edge_w.get(key, 1.0)
            weighted_sum += w * config[j]
            total_w += w
            
        if total_w > 1e-12:
            avg_neighbor = weighted_sum / total_w
            # Spherical interpolation or just normalize?
            # Simple: Move towards avg_neighbor
            new_pt = config[i] + lambda_smooth * (avg_neighbor - config[i])
            new_config[i] = _normalize(new_pt)
            
    return new_config

def _project_to_sphere(x_flat):
    """Project flattened vector to unit sphere points."""
    pts = x_flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _lbfgs_objective(x_flat):
    """
    Objective function for L-BFGS-B: minimize sum of squared covering radii
    (using smooth approximation via softmax).
    """
    pts = _project_to_sphere(x_flat)
    c, radii, _ = _get_circumcenters_and_radii(pts)
    if len(c) == 0:
        return 1.0
    # Use soft-min to create smooth objective
    # Minimize sum of exp(radii / tau) to focus on large radii
    tau = 0.05
    return np.sum(np.exp(radii / tau))

def _subgradient_optimize(config, deadline, min_time_remaining=10.0):
    """
    Local optimization using subgradient method on the covering radius.
    """
    config = config.copy()
    c, radii, nearest_idx = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        return config
        
    current_R = np.max(radii)
    best_config = config.copy()
    best_R = current_R
    
    step_size = 0.02
    stagnation = 0
    max_stagnation = 100
    weight_sharpness = 50.0
    
    while time.time() < deadline - min_time_remaining:
        c, radii, nearest_idx = _get_circumcenters_and_radii(config)
        if len(c) == 0:
            break
            
        current_R = np.max(radii)
        
        if current_R < best_R - 1e-10:
            best_R = current_R
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                break
                
        # Compute forces: move points towards the circumcenters that define the max radius
        forces = np.zeros_like(config)
        # Exponential weighting to focus on the worst circumcenters
        weights = np.exp(-weight_sharpness * (current_R - radii))
        
        for k in range(len(c)):
            i = nearest_idx[k]
            # Vector from point i to circumcenter k
            v = c[k] - config[i]
            # Project onto tangent space at config[i]
            v_proj = v - np.dot(v, config[i]) * config[i]
            forces[i] += weights[k] * v_proj
        
        # Apply step
        new_config = config.copy()
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-12:
                dir_vec = forces[i] / f_norm
                new_pt = new_config[i] + step_size * dir_vec
                new_config[i] = _normalize(new_pt)
        
        # Evaluate new configuration
        c_new, radii_new, _ = _get_circumcenters_and_radii(new_config)
        if len(c_new) == 0:
            break
        new_R = np.max(radii_new)
        
        if new_R < current_R - 1e-9:
            config = new_config
            current_R = new_R
            # Increase step size slightly if improving significantly
            if current_R < best_R:
                step_size = min(step_size * 1.02, 0.05)
        else:
            # Reject move, reduce step size
            step_size *= 0.95
            if step_size < 0.001:
                break
                
    return best_config

def _polish(config, deadline, min_time_remaining=5.0):
    """
    High-precision polish with very small steps and sharp weights.
    """
    config = config.copy()
    c, radii, nearest_idx = _get_circumcenters_and_radii(config)
    if len(c) == 0:
        return config
        
    current_R = np.max(radii)
    best_config = config.copy()
    best_R = current_R
    
    step_size = 0.002
    stagnation = 0
    max_stagnation = 200
    weight_sharpness = 200.0
    
    while time.time() < deadline - min_time_remaining:
        c, radii, nearest_idx = _get_circumcenters_and_radii(config)
        if len(c) == 0:
            break
            
        current_R = np.max(radii)
        
        if current_R < best_R - 1e-12:
            best_R = current_R
            best_config = config.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stagnation:
                break
                
        forces = np.zeros_like(config)
        weights = np.exp(-weight_sharpness * (current_R - radii))
        
        for k in range(len(c)):
            i = nearest_idx[k]
            v = c[k] - config[i]
            v_proj = v - np.dot(v, config[i]) * config[i]
            forces[i] += weights[k] * v_proj
        
        new_config = config.copy()
        for i in range(N):
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 1e-12:
                dir_vec = forces[i] / f_norm
                new_pt = new_config[i] + step_size * dir_vec
                new_config[i] = _normalize(new_pt)
        
        c_new, radii_new, _ = _get_circumcenters_and_radii(new_config)
        if len(c_new) == 0:
            break
        new_R = np.max(radii_new)
        
        if new_R < current_R - 1e-12:
            config = new_config
            current_R = new_R
        else:
            step_size *= 0.9
            if step_size < 0.0001:
                break
                
    return best_config

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 seconds safety margin

    # Initial configuration: Geodesic Dome
    config = _geodesic_dome(N)
    
    # Phase 1: Geodesic Smoothing to relax the mesh
    # Run for a fixed number of iterations or until time limit
    for iteration in range(100):
        if time.time() > deadline - 60:
            break
        new_config = _geodesic_smooth_step(config, lambda_smooth=0.5)
        if np.allclose(config, new_config, atol=1e-8):
            break
        config = new_config
        
    # Phase 2: L-BFGS-B optimization on smooth objective
    if time.time() < deadline - 40:
        try:
            x0 = config.flatten()
            result = minimize(
                _lbfgs_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
            if result.success or result.fun < _lbfgs_objective(x0):
                config = _project_to_sphere(result.x)
        except Exception:
            pass
    
    # Phase 3: Subgradient optimization
    if time.time() < deadline - 20:
        config = _subgradient_optimize(config, deadline, min_time_remaining=15.0)
        
    # Phase 4: Polish
    if time.time() < deadline - 5:
        config = _polish(config, deadline, min_time_remaining=3.0)
        
    # Final normalization and validation
    config = _normalize_batch(config)
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(config[i] - config[j]) < 1e-6:
                config[j] = config[j] + 1e-4 * np.random.randn(3)
                config[j] = _normalize(config[j])
                
    return config
