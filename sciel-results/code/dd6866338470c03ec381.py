"""Rewritten program for Heilbronn triangles task with stronger optimization."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]

# Vertices of the equilateral triangle
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])


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


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian.
    bary: (n, 3) array with columns (alpha, beta, gamma) where alpha+beta+gamma=1
    Vertex 0 = (0,0), Vertex 1 = (1,0), Vertex 2 = (0.5, H)
    """
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])


def _is_inside_triangle(points, eps=1e-6):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    
    if np.any(y < -eps) or np.any(y > H + eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    
    if np.any(S3 * x - y < -eps):
        return False
    
    if np.any(S3 * (1 - x) - y < -eps):
        return False
    
    return True


def _objective(flat):
    """Objective function: minimize -min_area."""
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _generate_fps_initialization(n_points=N, n_iter=50, seed=42):
    """
    Farthest Point Sampling based initialization.
    """
    rng = np.random.RandomState(seed)
    
    points = [np.array([0.5, H / 3])]
    
    for _ in range(n_points - 1):
        n_candidates = 1000
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
    
    # 1. Symmetric arrangement with center and rings
    center = np.array([0.5, H/3])
    r1 = 0.12
    r2 = 0.22
    r3 = 0.32
    
    pts = [center]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        x = 0.5 + r1 * np.cos(angle)
        y = H/3 + r1 * np.sin(angle)
        pts.append(np.array([x, y]))
    for i in range(4):
        angle = i * np.pi / 2 + np.pi/4
        x = 0.5 + r2 * np.cos(angle)
        y = H/3 + r2 * np.sin(angle)
        pts.append(np.array([x, y]))
    
    pts = np.array(pts)
    pts = _clip_to_triangle(pts)
    inits.append(pts.flatten())
    
    # 2. Corner-emphasized grid
    corner_pts = np.array([
        [0.05, 0.02],
        [0.95, 0.02],
        [0.5, H - 0.02],
        [0.15, 0.10],
        [0.85, 0.10],
        [0.35, H - 0.12],
        [0.65, H - 0.12],
        [0.5, H/3],
        [0.30, H/5],
        [0.70, H/5],
        [0.5, 2*H/5],
    ])
    corner_pts = _clip_to_triangle(corner_pts)
    inits.append(corner_pts.flatten())
    
    # 3. Barycentric structured
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
        pts = _bary_to_cart(bary)
        pts = _clip_to_triangle(pts)
        inits.append(pts.flatten())
    
    # 4. FPS-based with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        pts = _generate_fps_initialization(n_points=N, seed=seed)
        inits.append(pts.flatten())
    
    return inits


def _simulated_annealing(points, n_iter=5000, t0=0.05, t_end=0.001, seed=42):
    """
    Simulated Annealing to escape local optima.
    """
    rng = np.random.RandomState(seed)
    current = points.copy()
    current_score = _min_triangle_area_fast(current)
    best = current.copy()
    best_score = current_score
    
    t = t0
    cooling = (t_end / t0) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        t *= cooling
        
        # Generate perturbation
        scale = 0.05 * t / t0
        perturbed = current + rng.randn(N, 2) * scale
        perturbed = _clip_to_triangle(perturbed)
        
        perturbed_score = _min_triangle_area_fast(perturbed)
        delta = perturbed_score - current_score
        
        if delta > 0 or rng.rand() < np.exp(delta / max(t, 1e-10)):
            current = perturbed
            current_score = perturbed_score
            
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    inits = _structured_inits()
    
    # Evaluate all initial configurations
    for init_flat in inits:
        pts = _clip_to_triangle(init_flat.reshape(N, 2))
        score = _min_triangle_area_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Phase 1: Differential Evolution (global search)
    de_time_budget = 40
    if time.time() < deadline - de_time_budget:
        try:
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
    
    # Phase 2: Local refinement with multiple methods on diverse starts
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    candidates.extend(inits)
    
    # Add perturbations of best point
    if best_points is not None:
        rng = np.random.RandomState(42)
        for _ in range(6):
            perturbed = best_points + rng.randn(N, 2) * 0.06
            perturbed = _clip_to_triangle(perturbed)
            candidates.append(perturbed.flatten())
    
    for i, init_flat in enumerate(candidates):
        if time.time() > deadline - 8:
            break
            
        init_points = _clip_to_triangle(init_flat.reshape(N, 2))
        
        # SLSQP
        try:
            bounds = [(0, 1), (0, H)] * N
            res = minimize(
                _objective,
                init_flat,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res.x.reshape(N, 2))
        except Exception:
            pass
        
        # Nelder-Mead
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                _objective,
                init_flat,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: Simulated Annealing for escape from local optima
    if best_points is not None and time.time() < deadline - 10:
        try:
            sa_points, sa_score = _simulated_annealing(
                best_points, n_iter=3000, t0=0.04, t_end=0.001, seed=42
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points
        except Exception:
            pass
    
    # Phase 4: Final polishing with L-BFGS-B
    if best_points is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1), (0, H)] * N
            res = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15}
            )
            
            if res.fun < 0:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res.x.reshape(N, 2))
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
                
    if best_points is None or not _is_inside_triangle(best_points, eps=1e-3):
        best_points = _generate_fps_initialization(n_points=N, seed=0)
        best_points = _clip_to_triangle(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
