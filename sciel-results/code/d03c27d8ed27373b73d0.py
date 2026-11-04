<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates (a, b, c) to Cartesian.
    point = b*(1,0) + c*(0.5, H)
    so x = b + 0.5*c, y = H*c
    """
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _structured_inits():
    """Generate diverse structured initial configurations."""
    inits = []
    
    # Symmetric configuration near corners and center
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
    inits.append(_bary_to_cart(base_bary))
    
    # Three rows
    row_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.35, 0.05, 0.60],
        [0.65, 0.05, 0.30],
        [0.95, 0.05, 0.00],
        [0.15, 0.35, 0.50],
        [0.45, 0.35, 0.20],
        [0.75, 0.35, -0.10],
        [0.25, 0.65, 0.10],
        [0.55, 0.65, -0.20],
        [0.35, 0.55, 0.10],
        [0.65, 0.55, -0.20],
    ])
    row_bary[:, 2] = 1 - row_bary[:, 0] - row_bary[:, 1]
    inits.append(_bary_to_cart(row_bary))
    
    # Concentric pattern
    cent_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.10, 0.40, 0.50],
        [0.50, 0.40, 0.10],
        [0.40, 0.10, 0.50],
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.50, 0.05, 0.45],
        [0.20, 0.55, 0.25],
        [0.55, 0.25, 0.20],
        [0.25, 0.20, 0.55],
        [0.45, 0.30, 0.25],
    ])
    inits.append(_bary_to_cart(cent_bary))
    
    # Grid-based
    grid_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.30, 0.05, 0.65],
        [0.55, 0.05, 0.40],
        [0.80, 0.05, 0.15],
        [0.15, 0.30, 0.55],
        [0.40, 0.30, 0.30],
        [0.65, 0.30, 0.05],
        [0.25, 0.55, 0.20],
        [0.50, 0.55, -0.05],
        [0.35, 0.45, 0.20],
        [0.60, 0.45, -0.05],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    inits.append(_bary_to_cart(grid_bary))
    
    # More symmetric patterns
    sym_bary = np.array([
        [0.03, 0.03, 0.94],
        [0.94, 0.03, 0.03],
        [0.03, 0.94, 0.03],
        [0.47, 0.03, 0.50],
        [0.03, 0.47, 0.50],
        [0.50, 0.47, 0.03],
        [0.33, 0.33, 0.34],
        [0.22, 0.53, 0.25],
        [0.53, 0.22, 0.25],
        [0.25, 0.22, 0.53],
        [0.42, 0.28, 0.30],
    ])
    inits.append(_bary_to_cart(sym_bary))
    
    return inits


def _objective_cartesian(flat):
    """Objective for Nelder-Mead: minimize negative min triangle area with penalty."""
    points = flat.reshape(N, 2)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty for being outside the triangle
    eps = 1e-8
    penalty = 0.0
    penalty += np.sum(np.maximum(0, -y - eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * x + eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * (1 - x) + eps)) * 100
    
    min_area = _min_triangle_area_fast(points)
    return -min_area + penalty


def _refine_from(points, deadline):
    """Refine a configuration using multi-start Nelder-Mead with random perturbations."""
    best_points = points.copy()
    best_score = _min_triangle_area_fast(points)
    
    rng = np.random.RandomState(42)
    
    # Try multiple perturbations and Nelder-Mead refinements
    for attempt in range(8):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            start = best_points.copy()
        else:
            # Random perturbation of current best
            scale = 0.03 / (1 + 0.3 * attempt)
            perturbation = rng.randn(N, 2) * scale
            start = best_points + perturbation
            start = _clip_to_triangle(start)
        
        try:
            res = minimize(
                _objective_cartesian,
                start.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': None
                }
            )
            
            if res.fun < 0:  # valid (no penalty)
                candidate = res.x.reshape(N, 2)
                candidate = _clip_to_triangle(candidate)
                score = _min_triangle_area_fast(candidate)
                
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception:
            continue
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Generate structured initial configurations
    inits = _structured_inits()
    
    # For each initial configuration, refine with multi-start Nelder-Mead
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        init_pts = _clip_to_triangle(init_pts)
        init_score = _min_triangle_area_fast(init_pts)
        
        # Refine this configuration
        refined_pts, refined_score = _refine_from(init_pts, deadline)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # If we have time, do additional random restarts
    if best_points is not None and time.time() < deadline - 15:
        rng = np.random.RandomState(123)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            
            # Random configuration
            bary = rng.dirichlet(np.ones(N), size=N)
            random_pts = _bary_to_cart(bary)
            random_pts = _clip_to_triangle(random_pts)
            
            refined_pts, refined_score = _refine_from(random_pts, deadline)
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
    
    # Final fallback
    if best_points is None or not np.all(_clip_to_triangle(best_points) == best_points):
        # Use the first structured init as absolute fallback
        best_points = _clip_to_triangle(inits[0])
    
    # Ensure all points are valid
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates (a, b, c) to Cartesian.
    point = b*(1,0) + c*(0.5, H)
    so x = b + 0.5*c, y = H*c
    """
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1 - b - c
    return np.column_stack([a, b, c])


def _all_dets_from_bary(bary):
    """Compute all 165 determinant values (twice the signed area) from barycentric coords."""
    # In barycentric coords, the area ratio of triangle (i,j,k) to the big triangle
    # is the absolute value of the determinant of the 3x3 matrix with rows being
    # the barycentric coordinates of points i, j, k.
    b1 = bary[_TRIPLES[:, 0]]
    b2 = bary[_TRIPLES[:, 1]]
    b3 = bary[_TRIPLES[:, 2]]
    
    # det = a1*(b2*c3 - c2*b3) - b1*(a2*c3 - c2*a3) + c1*(a2*b3 - b2*a3)
    det = (b1[:, 0] * (b2[:, 1] * b3[:, 2] - b2[:, 2] * b3[:, 1]) -
           b1[:, 1] * (b2[:, 0] * b3[:, 2] - b2[:, 2] * b3[:, 0]) +
           b1[:, 2] * (b2[:, 0] * b3[:, 1] - b2[:, 1] * b3[:, 0]))
    return det


def _min_area_ratio_from_bary(bary):
    """Minimum area ratio (relative to big triangle area) from barycentric coords."""
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))


def _project_to_bary_simplex(bary):
    """Project barycentric coordinates onto the simplex (a,b,c >= 0, a+b+c=1)."""
    bary = bary.copy()
    # Ensure non-negative and sum to 1
    bary = np.clip(bary, 0, None)
    row_sums = bary.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary = bary / row_sums
    return bary


def _objective_smooth(flat, beta):
    """Smooth epigraph objective in barycentric coords.
    
    Variables: flat = [bary_0_0, bary_0_1, bary_0_2, ..., bary_10_0, bary_10_1, bary_10_2, t]
    where t is the lower bound on |det| for all triples.
    
    Objective: -t + (1/beta) * sum(log(1 + exp(beta*(t - |det_i|))))
    This smoothly penalizes violations of |det_i| >= t.
    """
    bary = flat[:N * 3].reshape(N, 3)
    t = flat[N * 3]
    
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    # Smooth max penalty: sum of log(1 + exp(beta*(t - |det_i|)))
    # Use numerically stable computation
    z = beta * (t - abs_dets)
    # log(1 + exp(z)) = max(0, z) + log(1 + exp(-|z|))
    smooth_log = np.maximum(0, z) + np.log1p(np.exp(-np.abs(z)))
    penalty = np.sum(smooth_log) / beta
    
    return -t + penalty


def _objective_slsqp(flat):
    """Objective for SLSQP: minimize -t with constraints |det_i| >= t."""
    bary = flat[:N * 3].reshape(N, 3)
    t = flat[N * 3]
    
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    # Penalty for constraint violations
    violations = np.maximum(0, t - abs_dets)
    penalty = np.sum(violations) * 10
    
    return -t + penalty


def _constraints_slsqp(flat):
    """Constraints for SLSQP: each |det_i| >= t."""
    bary = flat[:N * 3].reshape(N, 3)
    t = flat[N * 3]
    
    dets = _all_dets_from_bary(bary)
    abs_dets = np.abs(dets)
    
    # Each constraint: abs_det_i - t >= 0
    constraints = []
    for i in range(_N_TRIPLES):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: _abs_det_at(x, i) - x[N * 3]
        })
    return constraints


def _abs_det_at(flat, idx):
    """Compute |det| for a specific triple."""
    bary = flat[:N * 3].reshape(N, 3)
    dets = _all_dets_from_bary(bary)
    return np.abs(dets[idx])


def _structured_init_fps():
    """Generate initial barycentric configurations using farthest-point sampling style."""
    inits = []
    
    # Symmetric hexagonal pattern
    hex_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.15, 0.45, 0.40],
        [0.55, 0.45, 0.00],
        [0.45, 0.15, 0.40],
        [0.15, 0.15, 0.70],
        [0.70, 0.15, 0.15],
        [0.45, 0.70, -0.15],
        [0.25, 0.25, 0.50],
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.50, 0.50, 0.00],
    ])
    hex_bary[:, 2] = 1 - hex_bary[:, 0] - hex_bary[:, 1]
    hex_bary = _project_to_bary_simplex(hex_bary)
    inits.append(hex_bary)
    
    # Grid pattern
    grid_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.30, 0.05, 0.65],
        [0.55, 0.05, 0.40],
        [0.80, 0.05, 0.15],
        [0.15, 0.30, 0.55],
        [0.40, 0.30, 0.30],
        [0.65, 0.30, 0.05],
        [0.25, 0.55, 0.20],
        [0.50, 0.55, -0.05],
        [0.35, 0.45, 0.20],
        [0.60, 0.45, -0.05],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    grid_bary = _project_to_bary_simplex(grid_bary)
    inits.append(grid_bary)
    
    # Corner + center + midpoints
    corner_bary = np.array([
        [0.02, 0.02, 0.96],
        [0.96, 0.02, 0.02],
        [0.02, 0.96, 0.02],
        [0.48, 0.02, 0.50],
        [0.02, 0.48, 0.50],
        [0.50, 0.48, 0.02],
        [0.33, 0.33, 0.34],
        [0.20, 0.55, 0.25],
        [0.55, 0.20, 0.25],
        [0.25, 0.20, 0.55],
        [0.45, 0.28, 0.27],
    ])
    corner_bary = _project_to_bary_simplex(corner_bary)
    inits.append(corner_bary)
    
    return inits


def _local_optimize_nm(bary_init, t_init, deadline, beta_start=100, beta_end=500):
    """Local optimization using Nelder-Mead with annealing on beta."""
    best_bary = bary_init.copy()
    best_t = t_init
    best_score = _min_area_ratio_from_bary(bary_init)
    
    rng = np.random.RandomState(42)
    
    # Multi-stage beta annealing
    betas = [beta_start, 200, 300, 500]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        # Initialize flat vector
        flat = np.concatenate([best_bary.flatten(), [best_t]])
        
        try:
            res = minimize(
                _objective_smooth,
                flat,
                args=(beta,),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if res.success or res.fun < _objective_smooth(flat, beta):
                candidate_bary = res.x[:N * 3].reshape(N, 3)
                candidate_t = res.x[N * 3]
                
                # Project to valid barycentric coordinates
                candidate_bary = _project_to_bary_simplex(candidate_bary)
                
                score = _min_area_ratio_from_bary(candidate_bary)
                
                if score > best_score:
                    best_score = score
                    best_bary = candidate_bary.copy()
                    best_t = min(candidate_t, score)
        except Exception:
            continue
    
    return best_bary, best_t, best_score


def _local_optimize_slsqp(bary, t, deadline):
    """Polish with SLSQP using explicit constraints."""
    flat = np.concatenate([bary.flatten(), [t]])
    
    # Define constraints more efficiently
    cons = []
    for i in range(_N_TRIPLES):
        def make_constraint(idx):
            def constraint(x):
                bary_x = x[:N * 3].reshape(N, 3)
                dets = _all_dets_from_bary(bary_x)
                return np.abs(dets[idx]) - x[N * 3]
            return constraint
        cons.append({'type': 'ineq', 'fun': make_constraint(i)})
    
    # Bounds: barycentric coords in [0,1], t in [0, 0.5]
    bounds = []
    for i in range(N * 3):
        bounds.append((0, 1))
    bounds.append((0, 0.5))
    
    try:
        res = minimize(
            _objective_slsqp,
            flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if res.success:
            candidate_bary = res.x[:N * 3].reshape(N, 3)
            candidate_t = res.x[N * 3]
            candidate_bary = _project_to_bary_simplex(candidate_bary)
            
            score = _min_area_ratio_from_bary(candidate_bary)
            
            if score > _min_area_ratio_from_bary(bary):
                return candidate_bary, min(candidate_t, score), score
    except Exception:
        pass
    
    return bary, t, _min_area_ratio_from_bary(bary)


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_bary = None
    best_score = -np.inf
    
    # Generate structured initial configurations
    inits = _structured_init_fps()
    
    # For each initial configuration, refine with epigraph optimization
    for init_idx, init_bary in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        init_bary = _project_to_bary_simplex(init_bary)
        init_t = _min_area_ratio_from_bary(init_bary)
        
        # Stage 1: Nelder-Mead with annealing
        refined_bary, refined_t, refined_score = _local_optimize_nm(
            init_bary, init_t, deadline
        )
        
        # Stage 2: SLSQP polish
        if time.time() < deadline - 15:
            polished_bary, polished_t, polished_score = _local_optimize_slsqp(
                refined_bary, refined_t, deadline
            )
            if polished_score > refined_score:
                refined_bary = polished_bary
                refined_t = polished_t
                refined_score = polished_score
        
        if refined_score > best_score:
            best_score = refined_score
            best_bary = refined_bary.copy()
    
    # Additional random restarts if time permits
    if best_bary is not None and time.time() < deadline - 20:
        rng = np.random.RandomState(123)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random Dirichlet configuration
            random_bary = rng.dirichlet(np.ones(N), size=N)
            random_bary = _project_to_bary_simplex(random_bary)
            random_t = _min_area_ratio_from_bary(random_bary)
            
            refined_bary, refined_t, refined_score = _local_optimize_nm(
                random_bary, random_t, deadline
            )
            
            if time.time() < deadline - 5:
                polished_bary, polished_t, polished_score = _local_optimize_slsqp(
                    refined_bary, refined_t, deadline
                )
                if polished_score > refined_score:
                    refined_bary = polished_bary
                    refined_t = polished_t
                    refined_score = polished_score
            
            if refined_score > best_score:
                best_score = refined_score
                best_bary = refined_bary.copy()
    
    # Convert to Cartesian and ensure validity
    if best_bary is None:
        best_bary = _structured_init_fps()[0]
    
    best_points = _bary_to_cart(best_bary)
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
