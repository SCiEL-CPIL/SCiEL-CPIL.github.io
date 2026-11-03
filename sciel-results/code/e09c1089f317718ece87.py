"""Rewritten program: SQP-style optimization with analytical gradients for max-min triangle area."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

H = np.sqrt(3) / 2


def _all_triples():
    """Precompute all C(11,3)=165 triples."""
    return np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])


_TRIPLES = _all_triples()
_N_TRIPLES = _TRIPLES.shape[0]


def _min_triangle_area_and_grad(points):
    """
    Compute min triangle area and its gradient w.r.t. points.
    Returns (min_area, grad) where grad is shape (N, 2).
    """
    n = points.shape[0]
    p1 = points[_TRIPLES[:, 0]]  # (165, 2)
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    
    d1 = p2 - p1  # (165, 2)
    d2 = p3 - p1
    
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]  # (165,)
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    # Gradient of area w.r.t. each point
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # For the min triangle, compute gradient contribution
    # d(area)/d(p_i) = 0.5 * sign(cross) * [-(p_k - p_i) x (p_j - p_i)]_perp ... 
    # Actually: cross = (p2-p1)_x * (p3-p1)_y - (p2-p1)_y * (p3-p1)_x
    # d(cross)/d(p1) = -[d(cross)/d(p2) + d(cross)/d(p3)]
    # d(cross)/d(p2) = [(p3-p1)_y, -(p3-p1)_x]
    # d(cross)/d(p3) = [-(p2-p1)_y, (p2-p1)_x]
    
    sign = np.sign(cross[min_idx])
    if sign == 0:
        sign = 1.0
    
    # Gradient of cross (not abs) for the min triangle
    # Let i, j, k be the triple
    i, j, k = _TRIPLES[min_idx]
    a = p1[min_idx]  # p_i
    b = p2[min_idx]  # p_j
    c = p3[min_idx]  # p_k
    
    d1 = b - a
    d2 = c - a
    
    # d(cross)/d(b) = [d2[1], -d2[0]]
    # d(cross)/d(c) = [-d1[1], d1[0]]
    # d(cross)/d(a) = -d(cross)/d(b) - d(cross)/d(c)
    
    grad_cross_b = np.array([d2[1], -d2[0]])
    grad_cross_c = np.array([-d1[1], d1[0]])
    grad_cross_a = -(grad_cross_b + grad_cross_c)
    
    # area = 0.5 * sign * cross (since sign = sign(cross))
    # d(area)/d(point) = 0.5 * sign * d(cross)/d(point)
    factor = 0.5 * sign
    
    grad = np.zeros((n, 2))
    grad[i] += factor * grad_cross_a
    grad[j] += factor * grad_cross_b
    grad[k] += factor * grad_cross_c
    
    return min_area, grad


def _is_inside_triangle(points, eps=1e-6):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    
    if np.any(y < -eps) or np.any(y > H + eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    
    left_mask = x <= 0.5
    if np.any(left_mask):
        if np.any(y[left_mask] > np.sqrt(3) * x[left_mask] + eps):
            return False
    
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
    
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    points[:, 0] = x
    points[:, 1] = y
    
    left_mask = x <= 0.5
    if np.any(left_mask):
        max_y_left = np.sqrt(3) * x[left_mask]
        y[left_mask] = np.minimum(y[left_mask], max_y_left)
    
    right_mask = ~left_mask
    if np.any(right_mask):
        max_y_right = np.sqrt(3) * (1 - x[right_mask])
        y[right_mask] = np.minimum(y[right_mask], max_y_right)
    
    return points


def _sqp_step(points, max_iter=200, tol=1e-10):
    """
    SQP-style optimization: maximize min triangle area.
    Uses active set of smallest triangles to form a linearized subproblem.
    """
    points = points.copy()
    n = points.shape[0]
    
    for it in range(max_iter):
        # Compute all areas and find the minimum
        p1 = points[_TRIPLES[:, 0]]
        p2 = points[_TRIPLES[:, 1]]
        p3 = points[_TRIPLES[:, 2]]
        
        d1 = p2 - p1
        d2 = p3 - p1
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        # Find active set: triangles with area close to min
        active_mask = areas <= min_area + 1e-8
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) == 0:
            break
        
        # Build linearized system: maximize t subject to
        # For each active triangle: grad_area_t . (x_new - x) + t <= 0
        # This is a QP: min -t s.t. A @ delta + t <= 0, ||delta|| <= step_size
        # Simplified: solve for delta that increases min area
        
        # Collect gradients for active triangles
        grads = []
        for idx in active_idx:
            i, j, k = _TRIPLES[idx]
            a = points[i]
            b = points[j]
            c = points[k]
            
            d1 = b - a
            d2 = c - a
            sign = np.sign(cross[idx])
            if sign == 0:
                sign = 1.0
            
            grad_b = np.array([d2[1], -d2[0]])
            grad_c = np.array([-d1[1], d1[0]])
            grad_a = -(grad_b + grad_c)
            
            factor = 0.5 * sign
            g = np.zeros(2 * n)
            g[2*i:2*i+2] += factor * grad_a
            g[2*j:2*j+2] += factor * grad_b
            g[2*k:2*k+2] += factor * grad_c
            grads.append(g)
        
        grads = np.array(grads)  # (m, 2n)
        
        # Solve: max t s.t. grads @ delta + t * ones <= 0
        # Equivalent: min ||delta||^2 s.t. grads @ delta <= -t
        # Use a simple projected gradient / active set approach
        
        # Simple approach: take step in direction that increases all active areas
        # Average gradient direction
        avg_grad = np.mean(grads, axis=0)
        
        # Normalize
        norm_avg = np.linalg.norm(avg_grad)
        if norm_avg < 1e-15:
            break
        avg_grad /= norm_avg
        
        # Line search
        best_step = 0.01
        best_area = min_area
        
        for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
            new_points = points + step * avg_grad
            new_points = _clip_to_triangle(new_points)
            
            if not _is_inside_triangle(new_points, eps=1e-5):
                continue
            
            new_area = _min_triangle_area_fast(new_points)
            if new_area > best_area:
                best_area = new_area
                best_step = step
        
        if best_step <= 0:
            break
        
        points = _clip_to_triangle(points + best_step * avg_grad)
        
        if abs(best_area - min_area) < tol:
            break
    
    return points


def _min_triangle_area_fast(points):
    """Fast vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _objective_flat(flat):
    """Objective for scipy minimize: negative min area."""
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    
    if not _is_inside_triangle(points, eps=1e-5):
        return 100.0
    
    return -_min_triangle_area_fast(points)


def _generate_initializations(n_starts=20, rng_seed=42):
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(rng_seed)
    inits = []
    
    # Symmetric base configuration
    base_bary = np.array([
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
    
    def bary_to_cart(bary):
        b = bary[:, 1]
        c = bary[:, 2]
        x = b + 0.5 * c
        y = H * c
        return np.column_stack([x, y])
    
    base_cart = bary_to_cart(base_bary)
    inits.append(base_cart.flatten())
    
    # Additional starts
    for i in range(n_starts - 1):
        jitter_scale = 0.02 + 0.015 * (i % 6)
        rng_i = np.random.RandomState(rng_seed + i * 1000)
        jitter = rng_i.randn(N, 2) * jitter_scale
        perturbed = base_cart + jitter
        perturbed = _clip_to_triangle(perturbed)
        inits.append(perturbed.flatten())
    
    # Random starts
    for i in range(5):
        rng_i = np.random.RandomState(rng_seed + 10000 + i * 7)
        # Random points in triangle
        u = rng_i.rand(N)
        v = rng_i.rand(N)
        # Map to equilateral triangle
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        x = u + 0.5 * v
        y = H * v
        rand_pts = np.column_stack([x, y])
        rand_pts = _clip_to_triangle(rand_pts)
        inits.append(rand_pts.flatten())
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # Generate initial configurations
    inits = _generate_initializations(n_starts=15, rng_seed=42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multi-start Nelder-Mead with gradient-aware refinement
    for i, init_flat in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        try:
            init_points = init_flat.reshape(N, 2)
            init_points = _clip_to_triangle(init_points)
            
            if not _is_inside_triangle(init_points, eps=1e-5):
                continue
            
            # Run Nelder-Mead
            res = minimize(
                _objective_flat,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
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
    
    # Phase 2: SQP refinement on best points
    if best_points is not None and time.time() < deadline - 20:
        try:
            refined = _sqp_step(best_points, max_iter=100)
            if _is_inside_triangle(refined, eps=1e-5):
                refined_score = _min_triangle_area_fast(refined)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined
        except Exception:
            pass
    
    # Phase 3: Additional SQP from multiple starting points near best
    if best_points is not None and time.time() < deadline - 15:
        for i in range(5):
            if time.time() > deadline - 10:
                break
            try:
                rng_i = np.random.RandomState(42 + i * 100)
                perturbed = best_points + rng_i.randn(N, 2) * 0.02
                perturbed = _clip_to_triangle(perturbed)
                
                if not _is_inside_triangle(perturbed, eps=1e-5):
                    continue
                
                refined = _sqp_step(perturbed, max_iter=80)
                if _is_inside_triangle(refined, eps=1e-5):
                    score = _min_triangle_area_fast(refined)
                    if score > best_score:
                        best_score = score
                        best_points = refined
            except Exception:
                continue
    
    # Final fallback
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
    
    best_points = _clip_to_triangle(best_points)
    if not _is_inside_triangle(best_points, eps=1e-3):
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
