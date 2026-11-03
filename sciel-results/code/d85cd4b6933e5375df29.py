"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle (0,0), (1,0), (0.5, H)."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # Clip y to [0, H]
    y = np.clip(y, 0, H)
    
    # For a given y, x must be in [y/S3, 1 - y/S3]
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _is_inside_triangle(points, eps=1e-6):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    
    if np.any(y < -eps) or np.any(y > H + eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    
    # Check left edge: y <= S3 * x  =>  S3*x - y >= 0
    if np.any(S3 * x - y < -eps):
        return False
    
    # Check right edge: y <= S3 * (1-x)  =>  S3*(1-x) - y >= 0
    if np.any(S3 * (1 - x) - y < -eps):
        return False
    
    return True


def _objective(flat):
    """
    Objective function: minimize -min_area + penalty.
    flat: 2*N array
    """
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty for being outside (should be 0 after clipping, but just in case)
    # Actually clipping ensures inside, so we can just compute area.
    # But to be safe for the optimizer's view, we add a small penalty if somehow outside.
    # Since we clip, this is likely 0.
    
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _fs_init(n_points=11, n_iter=50, seed=42):
    """
    Farthest Point Sampling based initialization.
    Generates a well-spread set of points inside the triangle.
    """
    rng = np.random.RandomState(seed)
    
    # Start with a random point or a corner
    # Let's start with a point near the center
    points = [np.array([0.5, H / 3])]
    
    for _ in range(n_points - 1):
        # Generate many candidate points
        n_candidates = 1000
        # Uniform sampling in triangle:
        # x = r1 * (1 - sqrt(r2)) ... no, standard method:
        # r1, r2 uniform in [0,1]
        # if r1+r2 > 1, r1=1-r1, r2=1-r2
        # x = r1 + 0.5*r2
        # y = H * r2
        
        r1 = rng.rand(n_candidates)
        r2 = rng.rand(n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        cx = r1 + 0.5 * r2
        cy = H * r2
        candidates = np.column_stack([cx, cy])
        
        # Compute min distance from each candidate to existing points
        # points is a list of (2,) arrays
        # Convert to array for distance computation
        pts_arr = np.array(points)
        
        # dist_sq[i, j] = ||candidates[i] - pts_arr[j]||^2
        # Use broadcasting
        # candidates: (M, 2), pts_arr: (K, 2)
        # diff: (M, K, 2) -> too big? M=1000, K<=11. 1000*11*2 = 22000 floats. Fine.
        diff = candidates[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # For each candidate, find min dist to any existing point
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Pick the candidate with the largest min_dist_sq
        best_idx = np.argmax(min_dist_sq)
        points.append(candidates[best_idx])
    
    return np.array(points)


def _generate_inits():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. FS-based init with different seeds
    for seed in [42, 123, 456, 789]:
        pts = _fs_init(n_points=N, seed=seed)
        inits.append(pts.flatten())
        
    # 2. Structured symmetric init
    base_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.95, 0.05, 0.00],
        [0.05, 0.95, 0.00],
        [0.30, 0.05, 0.65],
        [0.05, 0.30, 0.65],
        [0.65, 0.30, 0.05],
        [0.33, 0.33, 0.34],
        [0.20, 0.55, 0.25],
        [0.55, 0.20, 0.25],
        [0.25, 0.20, 0.55],
        [0.45, 0.25, 0.30],
    ])
    # Ensure sum is 1
    base_bary[:, 2] = 1 - base_bary[:, 0] - base_bary[:, 1]
    
    b = base_bary[:, 1]
    c = base_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    pts = np.column_stack([x, y])
    inits.append(pts.flatten())
    
    # 3. Grid-like init
    grid_bary = np.array([
        [0.10, 0.10, 0.80],
        [0.40, 0.10, 0.50],
        [0.70, 0.10, 0.20],
        [0.20, 0.40, 0.40],
        [0.50, 0.40, 0.10],
        [0.80, 0.40, -0.20], # Will be clipped
        [0.30, 0.70, 0.00],
        [0.60, 0.70, -0.30], # Will be clipped
        [0.40, 0.25, 0.35],
        [0.60, 0.25, 0.15],
        [0.50, 0.50, 0.00],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    b = grid_bary[:, 1]
    c = grid_bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    pts = np.column_stack([x, y])
    pts = _clip_to_triangle(pts)
    inits.append(pts.flatten())
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    inits = _generate_inits()
    
    # Phase 1: Global search with Differential Evolution
    # We use a subset of time for DE
    de_time_budget = 60  # seconds
    
    if time.time() < deadline - de_time_budget:
        try:
            # Bounds for DE: x in [0,1], y in [0,H]
            # But we clip inside, so bounds can be loose
            bounds = [(0, 1), (0, H)] * N
            
            def de_objective(flat):
                points = flat.reshape(N, 2)
                points = _clip_to_triangle(points)
                return -_min_triangle_area_fast(points)
            
            # DE settings
            # maxiter and popsize trade-off with time
            # 150s total, 60s for DE.
            # DE can be slow. Let's try moderate settings.
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=30,
                popsize=10,
                tol=0.01,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.fun < 0:
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(de_result.x.reshape(N, 2))
                    
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    # Use the best point from DE and all initializations
    
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    candidates.extend(inits)
    
    # Also add some random perturbations of the best point found so far
    if best_points is not None:
        rng = np.random.RandomState(42)
        for _ in range(5):
            perturbed = best_points + rng.randn(N, 2) * 0.05
            perturbed = _clip_to_triangle(perturbed)
            candidates.append(perturbed.flatten())
    
    for i, init_flat in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        try:
            # Evaluate initial point
            init_points = _clip_to_triangle(init_flat.reshape(N, 2))
            init_area = _min_triangle_area_fast(init_points)
            
            # Run Nelder-Mead
            res = minimize(
                _objective,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res.x.reshape(N, 2))
                    
        except Exception:
            continue
    
    # Fallback
    if best_points is None:
        # Use the best initial
        for init_flat in inits:
            pts = _clip_to_triangle(init_flat.reshape(N, 2))
            score = _min_triangle_area_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts
                
    # Absolute fallback
    if best_points is None or not _is_inside_triangle(best_points, eps=1e-3):
        best_points = _fs_init(n_points=N, seed=0)
        best_points = _clip_to_triangle(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
