"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    
    if len(dist_sq_upper) == 0:
        return 1e10
    
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    if min_d_sq <= 1e-12:
        return 1e10
    
    return max_d_sq / min_d_sq

def _objective(points_flat):
    """Objective function for Nelder-Mead. Takes flat array of (N, D) points."""
    points = points_flat.reshape(N, D)
    return _compute_score(points)

def _generate_fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci sphere method."""
    points = np.zeros((n, D))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    for i in range(n):
        z = 1 - (i + 0.5) * 2.0 / n
        r = np.sqrt(1 - z * z)
        theta = phi * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    return points

def _generate_rings(n, num_rings):
    """
    Generate points arranged in rings.
    Distribute n points into num_rings.
    """
    points = []
    # Simple distribution: try to make rings as equal as possible
    # For n=14, maybe 2 poles and 12 on a ring? Or 4 rings of 3-4-4-3?
    # Let's try a generic ring generator.
    if num_rings == 1:
        # All on equator
        for i in range(n):
            theta = 2 * np.pi * i / n
            points.append([np.cos(theta), np.sin(theta), 0])
    else:
        # Distribute z-coordinates evenly
        # Example: num_rings=2 -> z = -0.5, 0.5? No, let's use poles and rings.
        # Strategy: 2 poles, remaining n-2 on rings.
        # For simplicity, let's just use specific known good topologies for n=14.
        pass
    return np.array(points)

def _initialization_candidates():
    """
    Generate a list of initial configurations for the optimizer.
    """
    candidates = []
    
    # 1. Fibonacci Sphere
    candidates.append(_generate_fibonacci_sphere(N))
    
    # 2. Random Sphere (multiple seeds)
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.random((N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
        
    # 3. Symmetric Ring Configuration: 2 Poles + 12 on Equator?
    # 12 points on a circle is a dodecagon.
    pts = np.zeros((N, D))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    for i in range(12):
        theta = 2 * np.pi * i / 12
        pts[2+i] = [np.cos(theta), np.sin(theta), 0]
    candidates.append(pts)
    
    # 4. Symmetric Ring Configuration: 2 Poles + 6 on Ring 1 + 6 on Ring 2?
    # Or 4 Poles (Tetrahedron) + 10 on rings?
    # Let's try: 2 Poles, 4 on Ring 1 (z=0.5), 4 on Ring 2 (z=0), 4 on Ring 3 (z=-0.5)?
    # Total 2+4+4+4 = 14.
    pts = np.zeros((N, D))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    
    # Ring 1: 4 points at z=0.5
    r1 = np.sqrt(1 - 0.5**2)
    for i in range(4):
        theta = 2 * np.pi * i / 4
        pts[2+i] = [r1 * np.cos(theta), r1 * np.sin(theta), 0.5]
        
    # Ring 2: 4 points at z=0
    r2 = 1.0
    for i in range(4):
        theta = 2 * np.pi * i / 4 + np.pi/4 # Offset
        pts[6+i] = [r2 * np.cos(theta), r2 * np.sin(theta), 0]
        
    # Ring 3: 4 points at z=-0.5
    r3 = np.sqrt(1 - 0.5**2)
    for i in range(4):
        theta = 2 * np.pi * i / 4
        pts[10+i] = [r3 * np.cos(theta), r3 * np.sin(theta), -0.5]
        
    candidates.append(pts)
    
    # 5. Truncated Tetrahedron + 2? 
    # Truncated Tetrahedron has 12 vertices. Adding 2 might be tricky.
    # Let's add a random perturbation to the Cuboctahedron (12 verts) + 2 random?
    # Cuboctahedron vertices are permutations of (0, 1, 1).
    cubocta = []
    perms = [(0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1),
             (1,0,1), (1,0,-1), (1,-0,1), (1,-0,-1), # Note: 0 is 0
             (-1,0,1), (-1,0,-1), (-1,0,1), (-1,0,-1)] # Wait, duplicates?
    # Standard cuboctahedron: 12 vertices.
    # Let's just use the ring/Fibonacci ones, they are robust.
    
    return candidates

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses Multi-start Nelder-Mead with geometric initializations.
    """
    best_score = 1e10
    best_points = None
    
    candidates = _initialization_candidates()
    
    # Run Nelder-Mead from each candidate
    for i, init_pts in enumerate(candidates):
        # Flatten for optimizer
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(N, D)
        except Exception as e:
            # Skip failed optimizations
            continue
            
    # Fallback to random if all failed
    if best_points is None:
        best_points = np.random.randn(N, D)
        
    return best_points
# EVOLVE-BLOCK-END
