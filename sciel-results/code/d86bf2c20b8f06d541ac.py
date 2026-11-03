"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

H = np.sqrt(3) / 2


def _min_triangle_area_fast(points):
    """Fast vectorized min triangle area for N=11 points."""
    n = points.shape[0]
    # Precompute all pair differences
    # Use broadcasting: points[i] - points[j] for all i,j
    # Area of triangle (i,j,k) = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    
    # For N=11, there are C(11,3) = 165 triangles
    # Vectorize over all triples
    idx = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    d1 = p2 - p1  # (165, 2)
    d2 = p3 - p1  # (165, 2)
    
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _is_inside_triangle(points, eps=1e-6):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    
    # Basic bounds
    if np.any(y < -eps) or np.any(y > H + eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    
    # Left edge: line from (0,0) to (0.5, H), equation: y = sqrt(3)*x
    # Right edge: line from (1,0) to (0.5, H), equation: y = sqrt(3)*(1-x)
    # Bottom edge: y = 0
    
    # For points with x <= 0.5, must satisfy y <= sqrt(3)*x
    left_mask = x <= 0.5
    if np.any(left_mask):
        if np.any(y[left_mask] > np.sqrt(3) * x[left_mask] + eps):
            return False
    
    # For points with x > 0.5, must satisfy y <= sqrt(3)*(1-x)
    right_mask = ~left_mask
    if np.any(right_mask):
        if np.any(y[right_mask] > np.sqrt(3) * (1 - x[right_mask]) + eps):
            return False
    
    return True


def _clip_to_triangle(points):
    """Clip points to be inside the triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # Clip to basic bounds
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    points[:, 0] = x
    points[:, 1] = y
    
    # For left half, clip to left edge
    left_mask = x <= 0.5
    if np.any(left_mask):
        max_y_left = np.sqrt(3) * x[left_mask]
        y[left_mask] = np.minimum(y[left_mask], max_y_left)
    
    # For right half, clip to right edge
    right_mask = ~left_mask
    if np.any(right_mask):
        max_y_right = np.sqrt(3) * (1 - x[right_mask])
        y[right_mask] = np.minimum(y[right_mask], max_y_right)
    
    return points


def _objective_cartesian(flat):
    """
    Objective function for Nelder-Mead in Cartesian coordinates.
    flat: 2*N array [x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}]
    Returns: -min_area (we minimize this, so larger area = more negative = better)
    """
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    
    if not _is_inside_triangle(points, eps=1e-5):
        return 100.0  # Large penalty
    
    return -_min_triangle_area_fast(points)


def _generate_grid_initializations(n_starts=20, rng_seed=42):
    """
    Generate diverse initial configurations using structured grid + jitter.
    Returns list of (2*N,) flat arrays in Cartesian coordinates.
    """
    rng = np.random.RandomState(rng_seed)
    inits = []
    
    # Base symmetric configuration
    base_bary = np.array([
        [0.02, 0.02, 0.96],   # corner (0,0)
        [0.96, 0.02, 0.02],   # corner (1,0)
        [0.02, 0.96, 0.02],   # corner (0.5, H)
        [0.45, 0.02, 0.53],   # bottom-left
        [0.02, 0.45, 0.53],   # bottom-right
        [0.53, 0.45, 0.02],   # top
        [0.33, 0.33, 0.34],   # center
        [0.25, 0.55, 0.20],
        [0.55, 0.20, 0.25],
        [0.20, 0.25, 0.55],
        [0.45, 0.25, 0.30],
    ])
    
    # Convert barycentric to Cartesian
    def bary_to_cart(bary):
        # point = b*(1,0) + c*(0.5, H) where a+b+c=1
        b = bary[:, 1]
        c = bary[:, 2]
        x = b + 0.5 * c
        y = H * c
        return np.column_stack([x, y])
    
    base_cart = bary_to_cart(base_bary)
    inits.append(base_cart.flatten())
    
    # Generate additional starts with structured jitter
    for i in range(n_starts - 1):
        # Different jitter magnitudes and patterns
        jitter_scale = 0.03 + 0.02 * (i % 5)
        rng_i = np.random.RandomState(rng_seed + i * 1000)
        
        # Apply random jitter
        jitter = rng_i.randn(N, 2) * jitter_scale
        perturbed = base_cart + jitter
        
        # Try different symmetry breaking patterns
        if i % 3 == 1:
            # Shift some points
            perturbed[:3] += rng_i.randn(3, 2) * 0.05
        elif i % 3 == 2:
            # Rotate slightly around centroid
            cx, cy = perturbed.mean(axis=0)
            angle = rng_i.uniform(-0.1, 0.1)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for k in range(N):
                dx = perturbed[k, 0] - cx
                dy = perturbed[k, 1] - cy
                perturbed[k, 0] = cx + cos_a * dx - sin_a * dy
                perturbed[k, 1] = cy + sin_a * dx + cos_a * dy
        
        perturbed = _clip_to_triangle(perturbed)
        inits.append(perturbed.flatten())
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate initial configurations
    inits = _generate_grid_initializations(n_starts=15, rng_seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Multi-start Nelder-Mead optimization
    for i, init_flat in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        try:
            # Quick evaluation of initial point
            init_points = init_flat.reshape(N, 2)
            init_points = _clip_to_triangle(init_points)
            
            if not _is_inside_triangle(init_points, eps=1e-5):
                continue
            
            init_area = _min_triangle_area_fast(init_points)
            
            # Run Nelder-Mead
            res = minimize(
                _objective_cartesian,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': None
                }
            )
            
            if res.fun < 0:  # Valid solution
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res.x.reshape(N, 2))
                    
        except Exception:
            continue
    
    # If no valid result, use the best initial
    if best_points is None:
        for init_flat in inits:
            pts = _clip_to_triangle(init_flat.reshape(N, 2))
            if _is_inside_triangle(pts, eps=1e-4):
                score = _min_triangle_area_fast(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
    
    # Final fallback: hardcoded good configuration
    if best_points is None or not _is_inside_triangle(best_points, eps=1e-3):
        fallback_bary = np.array([
            [0.02, 0.02, 0.96],
            [0.96, 0.02, 0.02],
            [0.02, 0.96, 0.02],
            [0.45, 0.02, 0.53],
            [0.02, 0.45, 0.53],
            [0.53, 0.45, 0.02],
            [0.33, 0.33, 0.34],
            [0.25, 0.55, 0.20],
            [0.55, 0.20, 0.25],
            [0.20, 0.25, 0.55],
            [0.45, 0.25, 0.30],
        ])
        b = fallback_bary[:, 1]
        c = fallback_bary[:, 2]
        best_points = np.column_stack([b + 0.5 * c, H * c])
    
    # Ensure validity
    best_points = _clip_to_triangle(best_points)
    if not _is_inside_triangle(best_points, eps=1e-3):
        # Absolute last resort: vertices and centroid
        best_points = np.array([
            [0.01, 0.01],
            [0.99, 0.01],
            [0.5, H * 0.99],
            [0.25, 0.01],
            [0.75, 0.01],
            [0.5, H * 0.5],
            [0.5, H * 0.25],
            [0.3, H * 0.3],
            [0.7, H * 0.3],
            [0.5, H * 0.7],
            [0.5, H * 0.1],
        ])
    
    return best_points
# EVOLVE-BLOCK-END
