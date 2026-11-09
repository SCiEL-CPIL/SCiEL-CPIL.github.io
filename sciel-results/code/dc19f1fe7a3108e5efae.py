# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _objective_flat(x):
    """
    Objective for Nelder-Mead: minimize -R.
    x is a flat array of 16*2 = 32 coordinates.
    """
    pts = x.reshape(16, 2)
    return -_score(pts)

def _generate_hexagonal_cluster_16():
    """
    Generate a hexagonal cluster of 16 points.
    Center (1) + Ring 1 (6) + Ring 2 (9 points from the 12 available in the next shell).
    We pick 9 points from the 12 in the second shell to form a somewhat symmetric shape.
    The 12 points in the second shell are at angles 0, 30, 60, ... 330 degrees? No.
    In a hex lattice with spacing 1:
    Ring 1: 6 points at distance 1, angles 0, 60, 120, 180, 240, 300.
    Ring 2: 12 points.
    The points in Ring 2 are at distance sqrt(3) from center? No.
    Let's list coordinates:
    (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1) -> Ring 1 (using axial coords)
    In Cartesian:
    (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    Ring 2 points (distance 2 from some Ring 1 points, or sqrt(3) from center? No, distance 2 from center is not right for hex lattice).
    Actually, the neighbors of the 6 Ring 1 points (excluding center and each other) form Ring 2.
    Total points in hex lattice within distance 2 from origin (in lattice steps):
    1 + 6 + 12 = 19.
    We want 16. So we take Center + 6 Ring 1 + 9 from Ring 2.
    To maintain symmetry, we can drop 3 points from Ring 2.
    Ring 2 has 12 points. They come in 4 sets of 3? Or 6 sets of 2?
    Let's just generate the 19 points and pick the 16 closest to the centroid?
    Or better, pick 9 points from Ring 2 that are as symmetric as possible.
    The 12 points in Ring 2 are at angles:
    0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330? No, that's not right for hex lattice.
    
    Let's use the lattice generation method again.
    """
    points = [[0.0, 0.0]]
    r1 = 1.0
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            dist = np.sqrt(x*x + y*y)
            if dist > 1.5 and dist < 2.5: # Ring 2 is roughly distance sqrt(3) ~ 1.732 to 2.0?
                # Actually, the 12 points in the second shell are at distance sqrt(3) (6 points) and distance 2 (6 points)?
                # Let's check distances.
                # (2, 0) -> dist 2
                # (1.5, sqrt(3)/2) -> dist sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732
                # (1, sqrt(3)) -> dist sqrt(1 + 3) = 2
                # So there are 6 points at dist sqrt(3) and 6 points at dist 2.
                # Total 12.
                hex_pts.append([x, y])
    
    hex_pts = np.array(hex_pts)
    # Sort by distance to origin
    dists = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists)
    # Take 9 points. To be symmetric, we should take all 6 at dist sqrt(3) and 3 at dist 2?
    # Or maybe take a symmetric subset of 9.
    # Let's just take the 9 closest.
    selected = hex_pts[idx[:9]]
    points.extend(selected.tolist())
    pts = np.array(points)
    return pts

def _generate_concentric_rings_16():
    """
    Generate 16 points as two concentric regular polygons.
    e.g., 8 on outer ring, 8 on inner ring, rotated.
    """
    n_outer = 8
    n_inner = 8
    r_outer = 1.0
    r_inner = 0.5 # Will be optimized
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_outer
    
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    pts = np.vstack([pts_outer, pts_inner])
    return pts

def _generate_random_symmetric_16():
    """
    Generate a random configuration with some symmetry.
    e.g., 4 points on a square, 4 on a smaller square rotated, 4 on a circle, 4 on another circle.
    """
    pts = []
    r1 = 1.0
    r2 = 0.5
    r3 = 0.8
    r4 = 0.3
    
    for r, angle_offset in [(r1, 0), (r2, np.pi/4), (r3, np.pi/8), (r4, 0)]:
        for i in range(4):
            angle = angle_offset + i * np.pi / 2
            pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(pts)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2

    # Hard deadline: 630s wall-clock, leave 30s margin
    deadline = time.time() + 600

    best_points = None
    best_score = -1.0

    rng = np.random.default_rng(42)

    initial_configs = []

    # Config 1: Hexagonal Cluster
    initial_configs.append(_generate_hexagonal_cluster_16())

    # Config 2: Concentric Rings (8+8)
    initial_configs.append(_generate_concentric_rings_16())

    # Config 3: Random Symmetric
    initial_configs.append(_generate_random_symmetric_16())
    
    # Config 4: Two concentric regular octagons (different ratio)
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r_outer = 1.0
    r_inner = 0.7
    pts1 = np.vstack([
        np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)]),
        np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    ])
    initial_configs.append(pts1)

    # Config 5: Random
    initial_configs.append(rng.standard_normal((n, d)))

    # Evaluate and improve each initial config
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break

        # Normalize to unit scale
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale

        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        # Nelder-Mead Optimization
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Flatten the points for Nelder-Mead
        x0 = init.flatten()
        
        # Set up options for Nelder-Mead
        options = {
            'maxiter': 2000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
        
        # Time budget per optimization run
        time_per_opt = min(remaining - 10, 120)
        start_time = time.time()
        
        try:
            # Use a wrapper to check time periodically? Nelder-Mead doesn't easily support callback for time check.
            # We can just let it run and hope it finishes, or break if it takes too long.
            # Since we have a deadline guard, we can't easily interrupt minimize.
            # We'll just run it and check time after.
            result = minimize(_objective_flat, x0, method='Nelder-Mead', options=options)
            
            # Check if we are still within time
            if time.time() < deadline - 10:
                optimized_pts = result.x.reshape(n, d)
                # Re-center and scale? No, Nelder-Mead optimizes the raw coordinates.
                # But we normalized init. The optimal solution might be at a different scale.
                # However, the score is scale-invariant. So we can just use the result.
                # But to be safe, let's re-center.
                optimized_pts = optimized_pts - optimized_pts.mean(axis=0)
                
                opt_score = _score(optimized_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized_pts.copy()
        except Exception:
            pass

    # --- Phase 2: Refine the best solution found so far with more iterations ---
    if best_points is not None and time.time() < deadline - 20:
        remaining = deadline - time.time()
        if remaining > 20:
            x0 = best_points.flatten()
            options = {
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
            
            try:
                result = minimize(_objective_flat, x0, method='Nelder-Mead', options=options)
                if time.time() < deadline - 5:
                    optimized_pts = result.x.reshape(n, d)
                    optimized_pts = optimized_pts - optimized_pts.mean(axis=0)
                    opt_score = _score(optimized_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized_pts.copy()
            except Exception:
                pass

    # --- Phase 3: Random restarts with Nelder-Mead if time permits ---
    while time.time() < deadline - 30:
        init = rng.standard_normal((n, d))
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale

        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        remaining = deadline - time.time()
        if remaining < 15:
            break
            
        x0 = init.flatten()
        options = {
            'maxiter': 1000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
        
        try:
            result = minimize(_objective_flat, x0, method='Nelder-Mead', options=options)
            if time.time() < deadline - 10:
                optimized_pts = result.x.reshape(n, d)
                optimized_pts = optimized_pts - optimized_pts.mean(axis=0)
                opt_score = _score(optimized_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized_pts.copy()
        except Exception:
            pass

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
