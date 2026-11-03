"""Spherical Lloyd's algorithm with damped relaxation and improved initialization."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _spherical_circumcenter_fast(a, b, c):
    """Fast spherical circumcenter using cross products."""
    # p is orthogonal to (b-a) and (c-a)
    u = b - a
    v = c - a
    p_dir = np.cross(u, v)
    p_norm = np.linalg.norm(p_dir)
    if p_norm < 1e-14:
        centroid = a + b + c
        cn = np.linalg.norm(centroid)
        if cn < 1e-14:
            return None
        return centroid / cn
    p = p_dir / p_norm
    # Ensure correct side
    centroid = (a + b + c) / 3.0
    if np.dot(p, centroid) < 0:
        p = -p
    return p

def _facet_circumradius_fast(a, b, c):
    """Fast circumradius computation."""
    cc = _spherical_circumcenter_fast(a, b, c)
    if cc is None:
        return 0.0
    dot_val = np.clip(np.dot(cc, a), -1.0, 1.0)
    return np.arccos(dot_val)

def _compute_all_circumradii(points, hull):
    """Compute circumradius for all facets in one pass."""
    n_facets = len(hull.simplices)
    crs = np.zeros(n_facets)
    for i, simplex in enumerate(hull.simplices):
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        crs[i] = _facet_circumradius_fast(a, b, c)
    return crs

def _compute_covering_radius_fast(points):
    """Compute covering radius using precomputed circumradii."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    crs = _compute_all_circumradii(points, hull)
    return np.degrees(np.max(crs))

# Replaced by _spherical_circumcenter_fast above

def _spherical_triangle_area(a, b, c):
    """Compute spherical area of triangle using vector formula."""
    triple = np.dot(a, np.cross(b, c))
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    if abs(denom) < 1e-14 and abs(triple) < 1e-14:
        return 0.0
    return 2.0 * np.arctan2(abs(triple), denom)

# Replaced by _facet_circumradius_fast above

def _compute_covering_radius(points):
    """Compute the covering radius in degrees."""
    return _compute_covering_radius_fast(points)

def _lloyd_step_damped(points, damping=0.5, radius_alpha=1.0):
    """Perform one damped Lloyd's step with radius weighting.
    
    Weights are area * (circumradius)^radius_alpha.
    This biases movement to shrink large-radius cells.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    n_points = len(points)
    targets = np.zeros_like(points)
    
    # Map each point to its incident facets
    point_facets = [[] for _ in range(n_points)]
    for simplex in hull.simplices:
        for v in simplex:
            point_facets[v].append(simplex)
    
    for i in range(n_points):
        facets = point_facets[i]
        if not facets:
            targets[i] = points[i]
            continue
        
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        
        for simplex in facets:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            cc = _spherical_circumcenter_fast(a, b, c)
            area = _spherical_triangle_area(a, b, c)
            cr = _facet_circumradius_fast(a, b, c)
            
            if cc is not None and area > 1e-12 and cr > 1e-12:
                weight = area * (cr ** radius_alpha)
                weighted_sum += weight * cc
                total_weight += weight
        
        if total_weight > 1e-12:
            centroid = weighted_sum / total_weight
            cn = np.linalg.norm(centroid)
            if cn > 1e-12:
                targets[i] = centroid / cn
            else:
                targets[i] = points[i]
        else:
            targets[i] = points[i]
    
    # Apply damped update: move toward target by damping factor
    new_points = np.zeros_like(points)
    for i in range(n_points):
        dot_val = np.clip(np.dot(points[i], targets[i]), -1.0, 1.0)
        angle = np.arccos(dot_val)
        
        if angle < 1e-12:
            new_points[i] = points[i]
        else:
            move_angle = damping * angle
            sin_move = np.sin(move_angle)
            sin_total = np.sin(angle)
            
            if sin_total < 1e-12:
                new_points[i] = points[i]
            else:
                coeff1 = np.cos(move_angle)
                coeff2 = sin_move / sin_total
                new_points[i] = coeff1 * points[i] + coeff2 * targets[i]
                nn = np.linalg.norm(new_points[i])
                if nn > 1e-12:
                    new_points[i] = new_points[i] / nn
                else:
                    new_points[i] = points[i]
    
    return new_points

def _separation_check(points, min_sep=1e-8):
    """Check minimum pairwise separation. Returns True if all points are sufficiently separated."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_sep:
                return False
    return True

def _local_optimize_covering(points, max_time=10.0):
    """Local optimization: iteratively reduce the maximum facet circumradius.
    
    Strategy: find the worst facet, move its vertices toward the circumcenter
    with small steps, accept only if the max circumradius decreases.
    Uses multiple step sizes and tries moving different subsets of vertices.
    """
    start_time = time.time()
    current = points.copy()
    current_radius = _compute_covering_radius(current)
    
    step_sizes = [0.1, 0.05, 0.02, 0.01]
    no_improve_count = 0
    step_idx = 0
    
    while time.time() - start_time < max_time:
        if no_improve_count > 30:
            break
        
        try:
            hull = ConvexHull(current)
        except Exception:
            break
        
        # Find worst facet
        max_cr = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = current[simplex[0]], current[simplex[1]], current[simplex[2]]
            cr = _facet_circumradius_fast(a, b, c)
            if cr > max_cr:
                max_cr = cr
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Get circumcenter of worst facet
        a, b, c = current[worst_simplex[0]], current[worst_simplex[1]], current[worst_simplex[2]]
        cc = _spherical_circumcenter_fast(a, b, c)
        if cc is None:
            break
        
        # Try different step sizes
        improved = False
        for step_size in step_sizes:
            # Move vertices toward circumcenter
            trial = current.copy()
            for idx in worst_simplex:
                dot_val = np.clip(np.dot(trial[idx], cc), -1.0, 1.0)
                angle = np.arccos(dot_val)
                if angle > 1e-12:
                    sin_angle = np.sin(angle)
                    if sin_angle > 1e-12:
                        coeff1 = np.cos(step_size * angle)
                        coeff2 = np.sin(step_size * angle) / sin_angle
                        new_pos = coeff1 * trial[idx] + coeff2 * cc
                        nn = np.linalg.norm(new_pos)
                        if nn > 1e-12:
                            trial[idx] = new_pos / nn
            
            # Normalize
            norms = np.linalg.norm(trial, axis=1, keepdims=True)
            trial = trial / norms
            
            if not _separation_check(trial, min_sep=1e-8):
                continue
            
            new_radius = _compute_covering_radius(trial)
            if new_radius < current_radius - 1e-8:
                current = trial
                current_radius = new_radius
                no_improve_count = 0
                improved = True
                break
        
        if not improved:
            no_improve_count += 1
            # Cycle through step sizes
            step_idx = (step_idx + 1) % len(step_sizes)
            step_sizes = step_sizes[step_idx:] + step_sizes[:step_idx]
    
    return current

def _fibonacci_initialization(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _random_initialization(n, rng):
    """Generate n random points on sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _icosa_initialization(n):
    """Generate points based on icosahedron subdivision."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Subdivide each face and project to sphere
    # For 50 points, we need about 38 additional points beyond the 12
    # Each face can be subdivided into 4 sub-triangles
    from scipy.spatial import Delaunay
    
    # Create triangulation on sphere using Delaunay in 3D won't work directly
    # Instead, use a simpler approach: start with icosahedron and add midpoints
    points = [tuple(v) for v in ico]
    
    # Add edge midpoints for some edges
    added = 0
    for i in range(len(ico)):
        for j in range(i+1, len(ico)):
            if added >= n - len(ico):
                break
            # Check if this is an edge (distance ~ 1.051)
            d = np.linalg.norm(ico[i] - ico[j])
            if 0.9 < d < 1.2:  # Edge threshold
                mid = (ico[i] + ico[j]) / 2
                mid = mid / np.linalg.norm(mid)
                if all(np.linalg.norm(mid - p) > 0.05 for p in points):
                    points.append(tuple(mid))
                    added += 1
            if added >= n - len(ico):
                break
        if added >= n - len(ico):
            break
    
    # If we still need more points, add random ones
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.standard_normal(3)
        p = p / np.linalg.norm(p)
        if all(np.linalg.norm(p - q) > 0.05 for q in points):
            points.append(tuple(p))
    
    return np.array(points[:n])

def construct():
    """Construct 50 points on the unit sphere using spherical Lloyd's algorithm."""
    start_time = time.time()
    max_total_time = 110.0
    
    # Try multiple initializations
    initializations = [
        _fibonacci_initialization(N),
        _icosa_initialization(N),
        _random_initialization(N, np.random.default_rng(123)),
        _random_initialization(N, np.random.default_rng(456)),
    ]
    
    best_points = None
    best_radius = np.inf
    
    for init in initializations:
        if time.time() - start_time > max_total_time - 30.0:
            break
        
        points = init.copy()
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Phase 1: Lloyd's relaxation
        remaining_for_lloyd = max_total_time - 30.0 - (time.time() - start_time)
        if remaining_for_lloyd < 10.0:
            continue
        
        damping = 0.5
        iterations_per_phase = 30
        num_phases = min(6, int(remaining_for_lloyd / 5.0))
        
        for phase in range(num_phases):
            if time.time() - start_time > max_total_time - 30.0:
                break
            
            damping = max(0.1, 0.5 * (0.75 ** phase))
            radius_alpha = 1.0 + 0.5 * phase
            
            for _ in range(iterations_per_phase):
                if time.time() - start_time > max_total_time - 30.0:
                    break
                
                new_points = _lloyd_step_damped(points, damping=damping, radius_alpha=radius_alpha)
                
                norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                new_points = new_points / norms
                
                if not _separation_check(new_points, min_sep=1e-10):
                    continue
                
                radius = _compute_covering_radius(new_points)
                
                if radius < best_radius:
                    best_radius = radius
                    best_points = new_points.copy()
                
                points = new_points
        
        # Phase 2: Local optimization
        remaining_time = max_total_time - (time.time() - start_time)
        if remaining_time > 10.0:
            optimized = _local_optimize_covering(points, max_time=min(remaining_time - 5.0, 25.0))
            opt_radius = _compute_covering_radius(optimized)
            if opt_radius < best_radius:
                best_points = optimized
                best_radius = opt_radius
    
    return best_points
