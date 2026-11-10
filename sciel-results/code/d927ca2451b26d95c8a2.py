# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools
import math
from scipy.optimize import basinhopping, minimize, differential_evolution

N = 13
# Precompute triangle indices for efficiency
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull
    Returns 0.0 if invalid
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Vectorized area calculation
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def soft_min_score(points, temperature=0.01):
    """
    Smooth surrogate for the objective function.
    We want to maximize A_min / A_hull.
    So we minimize -A_min / A_hull.
    A_min = min(areas).
    Soft min: -T * log(sum(exp(-area/T)))
    """
    if not np.all(np.isfinite(points)):
        return 1e10
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
    
    if A_hull <= 0:
        return 1e10
        
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min of areas
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum
    
    # Objective to minimize: -soft_min_area / A_hull
    return -soft_min_area / A_hull

def nelder_mead(f, x0, max_iter=5000, tol=1e-10, time_limit=None, start_time=None):
    """
    Nelder-Mead optimization algorithm with time limit support.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 60.0
    
    n = len(x0)
    # Initial simplex
    simplex = [x0.copy()]
    # Standard Nelder-Mead initialization
    for i in range(n):
        xi = x0.copy()
        if xi[i] != 0:
            xi[i] *= 1.05
        else:
            xi[i] += 0.05
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    # Nelder-Mead parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrinkage
    
    best_idx = 0
    for it in range(max_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Simple reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + 0.5 * (xr - centroid)
                fc = f(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            else:
                # Inside contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
        
        # Track best
        current_best = np.argmin(fvals)
        if fvals[current_best] < fvals[best_idx]:
            best_idx = current_best
    
    return simplex[best_idx], fvals[best_idx]

def generate_initial_candidates(rng):
    """
    Generate diverse initial candidates based on geometric templates.
    """
    candidates = []
    
    # Template 1: Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts)
    
    # Template 2: 1 center + 12 outer (regular 12-gon)
    center = np.array([0.0, 0.0])
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([center, outer])
    candidates.append(pts)
    
    # Template 3: 1 center + 6 inner + 6 outer
    center = np.array([0.0, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([center, inner, outer])
    candidates.append(pts)
    
    # Template 4: 2 center + 5 inner + 6 outer
    c1 = np.array([-0.1, 0.0])
    c2 = np.array([0.1, 0.0])
    angles_in = np.linspace(0, 2*np.pi, 5, endpoint=False)
    inner = 0.5 * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    angles_out = np.linspace(0, 2*np.pi, 6, endpoint=False)
    outer = 1.0 * np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    pts = np.vstack([c1, c2, inner, outer])
    candidates.append(pts)
    
    # Template 5: Grid-like
    xs = np.array([-1, -0.33, 0.33, 1.0])
    ys = np.array([-1, -0.33, 0.33, 1.0])
    grid = np.array([[x, y] for x in xs for y in ys])
    pts = grid[:13]
    candidates.append(pts)
    
    # Template 6: Random
    pts = rng.uniform(-1, 1, size=(13, 2))
    candidates.append(pts)
    
    # Template 7: Fibonacci spiral
    golden_angle = 2.39996
    radii = np.sqrt(np.arange(13) / 13.0)
    angles = np.arange(13) * golden_angle
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts)
    
    # Template 8: Barycentric-based - regular hull with interior points
    # Fix hull as regular 13-gon, place 8 interior points via barycentric coords
    hull_angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    hull_pts = np.column_stack([np.cos(hull_angles), np.sin(hull_angles)])
    
    # Create a triangulation of the hull (fan from vertex 0)
    # Interior points placed at barycentric coordinates within triangles
    interior_pts = []
    # Place points in different regions
    for i in range(8):
        # Choose a triangle in the fan triangulation
        tri_idx = (i * 3) % 13
        v1 = hull_pts[tri_idx]
        v2 = hull_pts[(tri_idx + 1) % 13]
        v3 = hull_pts[0]
        
        # Random barycentric coordinates
        u = rng.uniform(0.1, 0.9)
        v = rng.uniform(0, 0.9 - u)
        w = 1.0 - u - v
        
        pt = u * v1 + v * v2 + w * v3
        interior_pts.append(pt)
    
    pts = np.vstack([hull_pts[:5], np.array(interior_pts)])
    candidates.append(pts)
    
    # Template 9: Two concentric rings
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_ring = 0.4 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    outer_ring = 1.0 * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    pts = np.vstack([inner_ring, outer_ring])
    candidates.append(pts)
    
    # Template 10: Hexagonal grid subset
    hex_pts = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            x = i + 0.5 * j
            y = j * math.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:13])
    hex_pts = hex_pts - np.mean(hex_pts, axis=0)
    max_norm = np.max(np.linalg.norm(hex_pts, axis=1))
    if max_norm > 1e-6:
        hex_pts = hex_pts / max_norm
    candidates.append(hex_pts)
    
    return candidates

def heilbronn_convex13() -> np.ndarray:
    """
    Optimize 13 points to maximize the minimum triangle area ratio.
    Uses multiple optimization strategies with adaptive temperature.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 15.0  # 15s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Generate initial candidates
    candidates = generate_initial_candidates(rng)
    
    # Normalize initial points to unit scale for better optimization behavior
    normalized_candidates = []
    for base in candidates:
        init_pts_centered = base - np.mean(base, axis=0)
        max_norm = np.max(np.linalg.norm(init_pts_centered, axis=1))
        if max_norm > 1e-6:
            init_pts_scaled = init_pts_centered / max_norm
        else:
            init_pts_scaled = init_pts_centered
        normalized_candidates.append(init_pts_scaled.flatten())
    
    # Strategy 1: Nelder-Mead with multiple starts and adaptive temperature
    def objective_with_temp(x_flat, temp):
        pts = x_flat.reshape(13, 2)
        return soft_min_score(pts, temperature=temp)
    
    # Run Nelder-Mead on top candidates with different temperatures
    n_nm_runs = 4
    temps = [0.05, 0.03, 0.01, 0.005]
    
    for i in range(min(n_nm_runs, len(normalized_candidates))):
        current_time = time.time()
        if current_time > deadline:
            break
        
        x0 = normalized_candidates[i]
        temp = temps[i % len(temps)]
        
        try:
            x_opt, f_opt = nelder_mead(
                lambda x: objective_with_temp(x, temp),
                x0,
                max_iter=3000,
                tol=1e-12,
                time_limit=deadline - current_time - 10.0,
                start_time=start_time
            )
            
            pts_opt = x_opt.reshape(13, 2)
            score = compute_score(pts_opt)
            
            if score > best_score:
                best_score = score
                best_points = pts_opt.copy()
                
        except Exception as e:
            pass
    
    # Strategy 2: Basin Hopping with L-BFGS-B
    current_time = time.time()
    if current_time < deadline:
        n_bh_runs = 3
        time_per_bh = (deadline - current_time) / n_bh_runs
        
        for i in range(min(n_bh_runs, len(normalized_candidates))):
            current_time = time.time()
            if current_time > deadline:
                break
            
            x0 = normalized_candidates[i]
            
            def bh_objective(x_flat):
                pts = x_flat.reshape(13, 2)
                return soft_min_score(pts, temperature=0.05)
            
            def bh_local_optimizer(x_flat):
                res = minimize(
                    lambda x: soft_min_score(x.reshape(13, 2), temperature=0.02),
                    x_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                return res.x, res.fun, res.nfev
            
            try:
                bh_result = basinhopping(
                    bh_objective, 
                    x0, 
                    minimizer_kwargs=dict(method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-6}),
                    stepsize=0.5,
                    niter=30,
                    seed=42+i,
                    disp=False
                )
                
                pts_opt = bh_result.x.reshape(13, 2)
                score = compute_score(pts_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = pts_opt.copy()
                    
            except Exception as e:
                pass
    
    # Strategy 3: Differential Evolution for global exploration
    current_time = time.time()
    if current_time < deadline - 60:
        def de_objective(x_flat):
            pts = x_flat.reshape(13, 2)
            return soft_min_score(pts, temperature=0.1)
        
        bounds = [(-1.5, 1.5)] * 26
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol'
            )
            
            pts_opt = de_result.x.reshape(13, 2)
            score = compute_score(pts_opt)
            
            if score > best_score:
                best_score = score
                best_points = pts_opt.copy()
                
        except Exception as e:
            pass
    
    # Final Polish: Use a very low temperature to sharpen the minimum
    if best_points is not None:
        polish_deadline = deadline - 5.0
        current_time = time.time()
        if current_time < polish_deadline:
            x0 = best_points.flatten()
            
            # Multi-stage polish with decreasing temperatures
            polish_temps = [0.01, 0.005, 0.001]
            
            for temp in polish_temps:
                current_time = time.time()
                if current_time > polish_deadline:
                    break
                
                def obj_func_polish(x_flat, t=temp):
                    pts = x_flat.reshape(13, 2)
                    return soft_min_score(pts, temperature=t)
                
                try:
                    res = minimize(
                        obj_func_polish,
                        x0,
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12}
                    )
                    pts_polished = res.x.reshape(13, 2)
                    
                    score = compute_score(pts_polished)
                    if score > best_score:
                        best_score = score
                        best_points = pts_polished.copy()
                        x0 = pts_polished.flatten()
                except Exception:
                    pass
            
            # Ultra-fine final polish
            current_time = time.time()
            if current_time < polish_deadline:
                try:
                    x_final, f_final = nelder_mead(
                        lambda x: soft_min_score(x.reshape(13, 2), temperature=0.0005),
                        x0,
                        max_iter=5000,
                        tol=1e-14,
                        time_limit=polish_deadline - current_time,
                        start_time=start_time
                    )
                    pts_final = x_final.reshape(13, 2)
                    score = compute_score(pts_final)
                    if score > best_score:
                        best_score = score
                        best_points = pts_final.copy()
                except Exception:
                    pass

    # Fallback
    if best_points is None:
        best_points = candidates[0]
        
    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        best_points = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False))])
        
    return best_points

# EVOLVE-BLOCK-END
