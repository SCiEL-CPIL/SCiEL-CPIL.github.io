"""Repulsive force-based relaxation for covering the sphere with 50 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize

N = 50

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in degrees using Convex Hull facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_radius_deg = 0.0
    for simplex in hull.simplices:
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        
        denom = aa * bb - ab * ab
        if abs(denom) < 1e-15:
            continue
            
        s = (aa * bb - ab * ab) / (2.0 * denom)
        t = (aa * ab - aa * aa) / (2.0 * denom)
        
        circumcenter_2d = p0 + s * a + t * b
        R = np.linalg.norm(circumcenter_2d - p0)
        C_norm = np.linalg.norm(circumcenter_2d)
        
        if C_norm < 1e-15:
            theta = np.pi / 2.0
        else:
            cos_theta = (C_norm * C_norm + 1.0 - R * R) / (2.0 * C_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        radius_deg = np.degrees(theta)
        if radius_deg > max_radius_deg:
            max_radius_deg = radius_deg
            
    return max_radius_deg

def _repulsive_force(pts, target_radius_rad, power=2.0):
    """
    Compute repulsive forces between points. Points that are closer than 
    2*target_radius_rad repel each other with a force that increases as 
    they get closer.
    
    Args:
        pts: (N, 3) array of unit vectors
        target_radius_rad: target angular radius in radians
        power: exponent for the repulsion kernel
    
    Returns:
        (N, 3) array of forces
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Compute pairwise angular distances
    # cos(theta_ij) = dot(p_i, p_j)
    cos_theta = pts @ pts.T
    np.fill_diagonal(cos_theta, 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Target distance
    target = 2.0 * target_radius_rad
    
    for i in range(n):
        for j in range(i + 1, n):
            d = theta[i, j]
            if d < target and d > 1e-10:
                # Repulsion force: F = k * (target - d)^power * direction
                # Direction is along the great circle, perpendicular to both points
                # Simplified: push along the direction from j to i
                strength = (target - d) ** power
                # Force direction: from j toward i, projected onto tangent space
                # For simplicity, use the vector difference projected onto tangent
                diff = pts[i] - pts[j]
                # Project onto tangent plane at pts[i]
                tangent = diff - np.dot(diff, pts[i]) * pts[i]
                norm_tangent = np.linalg.norm(tangent)
                if norm_tangent > 1e-15:
                    tangent /= norm_tangent
                    forces[i] += strength * tangent
                    forces[j] -= strength * tangent
    
    return forces

def _attractive_force_toward_gaps(pts, samples, k_attract=0.1):
    """
    Compute attractive forces toward the largest gaps (farthest sample points).
    This helps distribute points more evenly.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Find the farthest sample point from the current configuration
    tree = cKDTree(pts)
    dists, _ = tree.query(samples)
    
    # Weight samples by how far they are (only the farthest few matter)
    sorted_idx = np.argsort(dists)[::-1]
    top_k = min(10, len(dists))
    top_indices = sorted_idx[:top_k]
    
    for idx in top_indices:
        sample = samples[idx]
        d = dists[idx]
        if d < 1e-10:
            continue
        
        # Find the nearest point to this sample
        _, nearest = tree.query(sample)
        p_nearest = pts[nearest]
        
        # Attractive force: pull p_nearest toward the sample direction
        # Project sample onto tangent space at p_nearest
        tangent = sample - np.dot(sample, p_nearest) * p_nearest
        norm_tangent = np.linalg.norm(tangent)
        if norm_tangent > 1e-15:
            tangent /= norm_tangent
            # Force strength proportional to how far the gap is
            strength = k_attract * d
            forces[nearest] += strength * tangent
    
    return forces

def _relax(pts, target_radius_rad, n_steps=500, learning_rate=0.01, 
           use_repulsion=True, use_attraction=True, seed=0):
    """
    Relax the configuration using force-based dynamics.
    """
    pts = _normalize(pts.copy())
    rng = np.random.RandomState(seed)
    
    for step in range(n_steps):
        forces = np.zeros_like(pts)
        
        if use_repulsion:
            forces += _repulsive_force(pts, target_radius_rad)
        
        if use_attraction:
            # Generate fresh samples each time for diversity
            samples = rng.randn(1000, 3)
            samples = _normalize(samples)
            forces += _attractive_force_toward_gaps(pts, samples)
        
        # Add small random perturbation to escape local minima
        forces += 0.01 * rng.randn(*pts.shape)
        
        # Update positions
        pts = pts + learning_rate * forces
        pts = _normalize(pts)
        
        # Gradually reduce learning rate
        if step % 100 == 99:
            learning_rate *= 0.8
    
    return pts

def _fibonacci_sphere(n, offset=0.5):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def construct():
    """
    Multi-start force-based relaxation for minimizing the covering radius.
    """
    best_pts = None
    best_radius = np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random starts
    rng = np.random.RandomState(123)
    for _ in range(3):
        pts = rng.randn(N, 3)
        inits.append(_normalize(pts))
    
    # 3. Structured start: icosahedron vertices + face centers
    phi_golden = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1,  phi_golden, 0], [ 1,  phi_golden, 0], [-1, -phi_golden, 0], [ 1, -phi_golden, 0],
        [0, -1,  phi_golden], [0,  1,  phi_golden], [0, -1, -phi_golden], [0,  1, -phi_golden],
        [ phi_golden, 0, -1], [ phi_golden, 0,  1], [-phi_golden, 0, -1], [-phi_golden, 0,  1]
    ], dtype=float)
    ico_verts = _normalize(ico_verts)
    # Add random points to fill up to 50
    remaining = N - len(ico_verts)
    extra = rng.randn(remaining, 3)
    extra = _normalize(extra)
    inits.append(_normalize(np.vstack([ico_verts, extra])))
    
    # Target radius: start with a reasonable estimate (~18 degrees = 0.314 rad)
    initial_target_rad = np.radians(18.0)
    
    for init_idx, init in enumerate(inits):
        try:
            # Phase 1: Force-based relaxation
            # Use a slightly larger target initially, then refine
            target_rad = initial_target_rad
            
            # First pass: broader relaxation
            pts_relaxed = _relax(
                init, 
                target_rad, 
                n_steps=300, 
                learning_rate=0.02,
                use_repulsion=True,
                use_attraction=True,
                seed=init_idx
            )
            
            # Evaluate current covering radius
            current_radius_deg = _covering_radius_exact(pts_relaxed)
            current_radius_rad = np.radians(current_radius_deg)
            
            # Phase 2: Fine-tune with exact target
            if current_radius_rad > 1e-6:
                pts_refined = _relax(
                    pts_relaxed,
                    current_radius_rad,
                    n_steps=200,
                    learning_rate=0.005,
                    use_repulsion=True,
                    use_attraction=False,
                    seed=init_idx + 100
                )
                
                refined_radius_deg = _covering_radius_exact(pts_refined)
                if refined_radius_deg < current_radius_deg:
                    pts_relaxed = pts_refined
                    current_radius_deg = refined_radius_deg
            
            # Phase 3: Exact polishing with Nelder-Mead
            def objective_exact(params):
                pts = _normalize(params.reshape(N, 3))
                return _covering_radius_exact(pts)
            
            result_exact = minimize(
                objective_exact,
                pts_relaxed.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            pts_final = _normalize(result_exact.x.reshape(N, 3))
            final_radius_deg = _covering_radius_exact(pts_final)
            
            if final_radius_deg < best_radius:
                best_radius = final_radius_deg
                best_pts = pts_final
                
        except Exception:
            continue
    
    if best_pts is None:
        best_pts = inits[0]
    
    return best_pts
