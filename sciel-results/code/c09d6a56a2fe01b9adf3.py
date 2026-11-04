"""Rewritten program for Heilbronn triangles task using smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]

# Vertices of the equilateral triangle
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])


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


def _all_triangle_areas(points):
    """Compute all triangle areas for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas


def _soft_min(areas, k=50.0):
    """
    Smooth approximation of the minimum using log-sum-exp.
    soft_min(x) ≈ -log(sum(exp(-k*x_i)))/k
    This is differentiable and approximates min(x) as k→∞.
    """
    # For numerical stability, subtract the min
    x = -k * areas
    x_max = np.max(x)
    log_sum = x_max + np.log(np.sum(np.exp(x - x_max)))
    return -log_sum / k


def _soft_min_grad(areas, k=50.0):
    """
    Gradient of the soft-min with respect to areas.
    d(soft_min)/d(areas_i) = (k * exp(-k*areas_i)) / sum_j(exp(-k*areas_j))
    This is a softmax-like distribution weighted towards the minimum.
    """
    x = -k * areas
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    weights = exp_x / np.sum(exp_x)
    return k * weights


def _objective_smooth(flat, k=50.0):
    """
    Smooth objective: minimize -soft_min(areas) + penalty for being outside triangle.
    flat: 2*N array of point coordinates
    """
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    areas = _all_triangle_areas(points)
    sm = _soft_min(areas, k)
    return -sm


def _objective_smooth_grad(flat, k=50.0):
    """
    Gradient of the smooth objective.
    Returns (value, gradient) for L-BFGS-B.
    """
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    areas = _all_triangle_areas(points)
    sm = _soft_min(areas, k)
    
    # Gradient of soft_min w.r.t. areas
    w = _soft_min_grad(areas, k)
    
    # Gradient of areas w.r.t. points
    # area_i = 0.5 * |cross| where cross = (p2-p1) x (p3-p1)
    # We need d(area_i)/d(p_j) for each point j
    # This is complex, so we use numerical differentiation or just return the value
    # For L-BFGS-B, we can let it use numerical gradients
    
    return -sm


def _objective_nonsmooth(flat):
    """
    Nonsmooth objective for SLSQP/Nelder-Mead: minimize -min_area.
    flat: 2*N array
    """
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _generate_fps_initialization(n_points=11, n_iter=50, seed=42):
    """
    Farthest Point Sampling based initialization.
    """
    rng = np.random.RandomState(seed)
    
    # Start with a point near the center
    points = [np.array([0.5, H / 3])]
    
    for _ in range(n_points - 1):
        n_candidates = 2000
        r1 = rng.rand(n_candidates)
        r2 = rng.rand(n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        cx = r1 + 0.5 * r2
        cy = H * r2
        candidates = np.column_stack([cx, cy])
        
        pts_arr = np.array(points)
        diff = candidates[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        min_dist_sq = np.min(dist_sq, axis=1)
        
        best_idx = np.argmax(min_dist_sq)
        points.append(candidates[best_idx])
    
    return np.array(points)


def _structured_inits():
    """Generate structured initial configurations based on symmetry."""
    inits = []
    
    # 1. Symmetric hexagonal-like arrangement
    center = np.array([0.5, H/3])
    r1 = 0.15
    r2 = 0.28
    
    pts = [center]
    for i in range(6):
        angle = i * np.pi / 3
        x = 0.5 + r1 * np.cos(angle)
        y = H/3 + r1 * np.sin(angle)
        pts.append(np.array([x, y]))
    for angle in [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6]:
        x = 0.5 + r2 * np.cos(angle)
        y = H/3 + r2 * np.sin(angle)
        pts.append(np.array([x, y]))
    
    pts = np.array(pts)
    pts = _clip_to_triangle(pts)
    inits.append(pts.flatten())
    
    # 2. Grid-like with corners emphasized
    corner_pts = np.array([
        [0.05, 0.02],
        [0.95, 0.02],
        [0.5, H - 0.02],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.35, H - 0.15],
        [0.65, H - 0.15],
        [0.5, H/3],
        [0.3, H/5],
        [0.7, H/5],
        [0.5, 2*H/5],
    ])
    corner_pts = _clip_to_triangle(corner_pts)
    inits.append(corner_pts.flatten())
    
    # 3. Barycentric structured init
    bary_configs = [
        [[0.05, 0.05, 0.90],
         [0.95, 0.05, 0.00],
         [0.05, 0.95, 0.00],
         [0.50, 0.50, 0.00],
         [0.50, 0.00, 0.50],
         [0.00, 0.50, 0.50],
         [1/3, 1/3, 1/3],
         [0.25, 0.25, 0.50],
         [0.25, 0.50, 0.25],
         [0.50, 0.25, 0.25],
         [0.40, 0.30, 0.30]],
    ]
    for bary in bary_configs:
        bary = np.array(bary)
        bary[:, 2] = 1 - bary[:, 0] - bary[:, 1]
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = H * bary[:, 2]
        pts = np.column_stack([x, y])
        pts = _clip_to_triangle(pts)
        inits.append(pts.flatten())
    
    # 4. FS-based with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 3141]:
        pts = _generate_fps_initialization(n_points=N, seed=seed)
        inits.append(pts.flatten())
    
    return inits


def _simulated_annealing(init_points, n_iter=5000, t_start=0.1, t_end=0.001, seed=42):
    """
    Simulated annealing to refine a set of points.
    Uses the nonsmooth objective but with random perturbations.
    """
    rng = np.random.RandomState(seed)
    points = _clip_to_triangle(init_points.copy())
    current_score = _min_triangle_area_fast(points)
    best_points = points.copy()
    best_score = current_score
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        # Random perturbation
        n_perturb = rng.randint(1, 4)  # Perturb 1 to 3 points
        indices = rng.choice(N, size=n_perturb, replace=False)
        perturbation = rng.randn(n_perturb, 2) * 0.05
        new_points = points.copy()
        new_points[indices] += perturbation
        new_points = _clip_to_triangle(new_points)
        
        new_score = _min_triangle_area_fast(new_points)
        
        # Accept if better, or with probability exp((new-old)/T)
        if new_score >= current_score:
            points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            delta = new_score - current_score
            prob = np.exp(delta / max(t, 1e-10))
            if rng.rand() < prob:
                points = new_points
                current_score = new_score
        
        t *= cooling
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    inits = _structured_inits()
    
    # Phase 1: Global search with multiple random restarts + local refinement
    # Use differential evolution if available and time permits
    de_time_budget = 40  # seconds
    
    if time.time() < deadline - de_time_budget:
        try:
            from scipy.optimize import differential_evolution
            
            bounds = [(0, 1), (0, H)] * N
            
            def de_objective(flat):
                points = flat.reshape(N, 2)
                points = _clip_to_triangle(points)
                return -_min_triangle_area_fast(points)
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=30,
                popsize=15,
                tol=0.005,
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
    
    # Phase 2: Local refinement with multiple methods on all candidates
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    candidates.extend(inits)
    
    # Add random perturbations of the best point found so far
    if best_points is not None:
        rng = np.random.RandomState(42)
        for _ in range(10):
            perturbed = best_points + rng.randn(N, 2) * 0.06
            perturbed = _clip_to_triangle(perturbed)
            candidates.append(perturbed.flatten())
    
    # Try SLSQP, Nelder-Mead, and L-BFGS-B on each candidate
    for i, init_flat in enumerate(candidates):
        if time.time() > deadline - 8:
            break
            
        init_points = _clip_to_triangle(init_flat.reshape(N, 2))
        
        # Method 1: SLSQP
        try:
            bounds = [(0, 1), (0, H)] * N
            res_slsqp = minimize(
                _objective_nonsmooth,
                init_flat,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res_slsqp.fun < 0:
                score = -res_slsqp.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res_slsqp.x.reshape(N, 2))
        except Exception:
            pass
        
        # Method 2: Nelder-Mead
        if time.time() > deadline - 8:
            break
        try:
            res_nm = minimize(
                _objective_nonsmooth,
                init_flat,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_nm.fun < 0:
                score = -res_nm.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res_nm.x.reshape(N, 2))
        except Exception:
            pass
        
        # Method 3: L-BFGS-B with smooth objective
        if time.time() > deadline - 8:
            break
        try:
            bounds = [(0, 1), (0, H)] * N
            res_lbfgs = minimize(
                _objective_smooth,
                init_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15}
            )
            
            if res_lbfgs.fun < 0:
                score = -res_lbfgs.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res_lbfgs.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: Simulated annealing on the best point
    if best_points is not None and time.time() < deadline - 10:
        try:
            sa_points, sa_score = _simulated_annealing(
                best_points, 
                n_iter=3000, 
                t_start=0.08, 
                t_end=0.001, 
                seed=42
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points
        except Exception:
            pass
    
    # Phase 4: Final polishing with high-precision SLSQP + Nelder-Mead
    if best_points is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1), (0, H)] * N
            res_final1 = minimize(
                _objective_nonsmooth,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if res_final1.fun < 0:
                score = -res_final1.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res_final1.x.reshape(N, 2))
        except Exception:
            pass
        
        if time.time() < deadline - 3:
            try:
                res_final2 = minimize(
                    _objective_nonsmooth,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res_final2.fun < 0:
                    score = -res_final2.fun
                    if score > best_score:
                        best_score = score
                        best_points = _clip_to_triangle(res_final2.x.reshape(N, 2))
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        for init_flat in inits:
            pts = _clip_to_triangle(init_flat.reshape(N, 2))
            score = _min_triangle_area_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts
                
    # Absolute fallback
    if best_points is None:
        best_points = _generate_fps_initialization(n_points=N, seed=0)
        best_points = _clip_to_triangle(best_points)
        
    return best_points
