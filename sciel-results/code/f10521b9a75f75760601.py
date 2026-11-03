import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _objective_soft(flat_points, beta=50.0):
    """Soft-min objective: differentiable approximation of -min_area/hull_area."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_feasibility(flat_points, t_init=None):
    """Feasibility-based objective: maximize t subject to all triangle areas >= t * hull_area.
    Uses a penalty formulation: minimize hull_area/t + lambda * sum(max(0, t*hull_area - area_i)^2)"""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    # Target: minimize hull_area / min(areas)
    # Equivalently, maximize min(areas) / hull_area
    # Use soft penalty for triangles below target
    min_a = np.min(areas)
    if min_a < 1e-12:
        return 1e10
    
    # Soft penalty: encourage all areas to be at least some fraction of the max
    # This helps avoid degenerate configurations
    target = min_a
    violations = np.maximum(0.0, target - areas)
    penalty = np.sum(violations**2)
    
    return hull_area / min_a + 0.01 * penalty

def _objective_soft_multi_beta(flat_points, betas=None):
    """Multi-scale soft-min: average over multiple beta values for robust landscape."""
    if betas is None:
        betas = [20.0, 50.0, 100.0, 200.0]
    vals = []
    for b in betas:
        vals.append(_objective_soft(flat_points, beta=b))
    return np.mean(vals)

# --- Centrally symmetric parameterization ---
# For N=14, we use 7 independent vertex pairs (v_i, -v_i).
# Parameterize by 7 angles and 7 radii: v_i = r_i * (cos(theta_i), sin(theta_i))

def _sym_params_to_points(params):
    """Generate 14 points from 7 angles and 7 radii (centrally symmetric)."""
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([half, -half])

def _objective_sym(params):
    """Hard objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    return -_score(points)

def _objective_sym_soft(params, beta=150.0):
    """Soft objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _objective_sym_feasibility(params):
    """Feasibility-based objective for symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0:
        return 1e10
    min_a = np.min(areas)
    if min_a < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    target = min_a
    violations = np.maximum(0.0, target - areas)
    penalty = np.sum(violations**2)
    return hull_area / min_a + 0.01 * penalty

def _objective_sym_soft_multi(params, betas=None):
    """Multi-scale soft objective for symmetric params."""
    if betas is None:
        betas = [50.0, 100.0, 200.0, 400.0]
    vals = []
    for b in betas:
        vals.append(_objective_sym_soft(params, beta=b))
    return np.mean(vals)

def _generate_sym_inits():
    """Generate initial parameter vectors for centrally symmetric optimization."""
    inits = []
    
    # Regular heptagon-based (7 equally spaced angles)
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inits.append(np.concatenate([base_angles, np.ones(7)]))
    
    # Rotated versions
    for rot in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.5, 1.7, 2.0, 2.3]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(7)]))
    
    # Non-uniform angle patterns with varying radii
    for seed in range(40):
        rng = np.random.default_rng(seed * 11 + 5)
        angles = np.sort(rng.uniform(0, 2*np.pi, 7))
        radii = 1.0 + rng.normal(0, 0.2, 7)
        radii = np.maximum(radii, 0.4)
        inits.append(np.concatenate([angles, radii]))
    
    # Pattern: alternating radii
    for r_pattern in [
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
        [1.0, 0.7, 0.9, 0.7, 1.0, 0.7, 0.9],
        [1.2, 0.9, 1.1, 0.9, 1.2, 0.9, 1.1],
        [1.0, 0.6, 0.9, 0.6, 1.0, 0.6, 0.9],
        [0.9, 1.1, 0.8, 1.1, 0.9, 1.1, 0.8],
        [1.0, 0.85, 1.0, 0.85, 1.0, 0.85, 1.0],
        [1.1, 0.75, 1.05, 0.75, 1.1, 0.75, 1.05],
    ]:
        a = base_angles + 0.2
        inits.append(np.concatenate([a, np.array(r_pattern)]))
    
    # Jittered regular
    for js in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        rng = np.random.default_rng(999 + int(js*100))
        a = base_angles + rng.normal(0, js, 7)
        r = 1.0 + rng.normal(0, js*0.5, 7)
        r = np.maximum(r, 0.4)
        inits.append(np.concatenate([a, r]))
    
    # Two-ring symmetric: 3 inner + 4 outer
    for r_in in [0.4, 0.5, 0.6]:
        a_in = np.linspace(0, 2*np.pi, 3, endpoint=False)
        a_out = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        angles = np.concatenate([a_in, a_out])
        radii = np.concatenate([np.full(3, r_in), np.ones(4)])
        inits.append(np.concatenate([angles, radii]))
    
    # Three-ring: 2 inner + 3 middle + 2 outer
    for r_in, r_mid in [(0.35, 0.65), (0.45, 0.75), (0.55, 0.85)]:
        a_in = np.linspace(0, np.pi, 2, endpoint=False)
        a_mid = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
        a_out = np.linspace(np.pi, 2*np.pi, 2, endpoint=False)
        angles = np.concatenate([a_in, a_mid, a_out])
        radii = np.concatenate([np.full(2, r_in), np.full(3, r_mid), np.ones(2)])
        inits.append(np.concatenate([angles, radii]))
    
    # Fibonacci-like angle spacing
    phi = (1 + np.sqrt(5)) / 2
    fib_angles = np.sort(np.array([((i * phi) % 1.0) * 2 * np.pi for i in range(7)]))
    inits.append(np.concatenate([fib_angles, np.ones(7)]))
    inits.append(np.concatenate([fib_angles + 0.3, np.ones(7)]))
    
    # Near-square with one perturbed pair
    sq_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 0.5, 2.6, 4.2])
    inits.append(np.concatenate([sq_angles, np.ones(7)]))
    inits.append(np.concatenate([sq_angles + 0.1, [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8]]))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(points_init, method='Nelder-Mead', maxiter=2000, tol=1e-8):
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(_objective, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
            elif method == 'L-BFGS-B':
                res = minimize(_objective, pts.flatten(), method='L-BFGS-B', 
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft(points_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        """Refine using soft-min objective for smoother landscape."""
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # 1. Initializations
    initializations = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initializations.append(x_reg)
    
    # Two concentric rings (7+7) with various radii and offsets
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        initializations.append(pts)
    
    # Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        initializations.append(pts)
    
    # Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (1.1, 0.9)]:
        a = np.linspace(0, 2*np.pi, N, endpoint=False)
        initializations.append(np.column_stack([rx*np.cos(a), ry*np.sin(a)]))
    
    # Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N+1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    x_halton = np.array(halton_pts)
    initializations.append(x_halton)
    
    # Random points
    rng = np.random.default_rng(123)
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(x_rand)

    # 2. Refine initializations
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
            
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=1500)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        if time.time() < deadline - 25.0:
            cand_pts2, cand_score2 = refine(cand_pts, 'Powell', maxiter=500)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)

    # 3. Symmetric optimization (KEY STRUCTURAL CHANGE)
    # Exploit central symmetry: optimize 7 points, reflect to get 14
    if time.time() < deadline - 30.0:
        sym_inits = _generate_sym_inits()
        
        for i, flat_init in enumerate(sym_inits):
            if time.time() > deadline - 25.0:
                break
            
            # Nelder-Mead on symmetric parameters
            try:
                res_sym = minimize(_objective_sym, flat_init, method='Nelder-Mead',
                                   options={'maxiter': 4000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                if res_sym.fun < 0:
                    pts14 = _sym_params_to_points(res_sym.x)
                    cand_score = -res_sym.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass
            
            # Powell on symmetric parameters
            if time.time() < deadline - 20.0:
                try:
                    res_sym_pow = minimize(_objective_sym, flat_init, method='Powell',
                                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9})
                    if res_sym_pow.fun < 0:
                        pts14 = _sym_params_to_points(res_sym_pow.x)
                        cand_score = -res_sym_pow.fun
                        update_best(pts14, cand_score)
                except Exception:
                    pass
            
            # Feasibility-based symmetric refinement
            if time.time() < deadline - 18.0:
                try:
                    res_sym_feas = minimize(_objective_sym_feasibility, flat_init, method='L-BFGS-B',
                                            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
                    pts14f = _sym_params_to_points(res_sym_feas.x)
                    cand_scoref = _score(pts14f)
                    update_best(pts14f, cand_scoref)
                except Exception:
                    pass

    # 4. Differential Evolution on symmetric space (14D instead of 28D)
    if time.time() < deadline - 30.0:
        def de_sym_objective(flat):
            if np.any(~np.isfinite(flat)):
                return 1.0
            pts = _sym_params_to_points(flat)
            if np.allclose(pts, pts[0]):
                return 1.0
            return -_score(pts)
            
        try:
            bounds = [(-np.pi, np.pi)] * 7 + [(0.1, 2.0)] * 7
            res_de = differential_evolution(
                de_sym_objective, 
                bounds, 
                seed=42, 
                maxiter=80, 
                popsize=15, 
                tol=1e-5, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < 0:
                pts14 = _sym_params_to_points(res_de.x)
                cand_score = -res_de.fun
                update_best(pts14, cand_score)
        except Exception:
            pass
        
        # Second DE run with different seed if time allows
        if time.time() < deadline - 15.0:
            try:
                res_de2 = differential_evolution(
                    de_sym_objective, 
                    bounds, 
                    seed=123, 
                    maxiter=40, 
                    popsize=10, 
                    tol=1e-5, 
                    mutation=(0.3, 0.8), 
                    recombination=0.8,
                    polish=True,
                    init='sobol'
                )
                if res_de2.success or res_de2.fun < 0:
                    pts14 = _sym_params_to_points(res_de2.x)
                    cand_score = -res_de2.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # 5. Final polish on the global best - multi-stage
    if best_points is not None and time.time() < deadline - 15.0:
        # Stage 1: Multi-beta soft refinement
        if time.time() < deadline - 12.0:
            try:
                center = np.mean(best_points, axis=0)
                pts_norm = best_points - center
                max_norm = np.max(np.linalg.norm(pts_norm, axis=1))
                if max_norm > 0:
                    pts_norm = pts_norm / max_norm
                res_multi = minimize(lambda x: _objective_soft_multi_beta(x), pts_norm.flatten(),
                                     method='L-BFGS-B',
                                     options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
                cand_pts = res_multi.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
            except Exception:
                pass
        
        # Stage 2: Hard Nelder-Mead with tight tolerance
        if time.time() < deadline - 10.0:
            cand_pts, cand_score = refine(best_points, 'Nelder-Mead', maxiter=8000, tol=1e-10)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
        
        # Stage 3: Powell with tight tolerance
        if time.time() < deadline - 8.0:
            cand_pts, cand_score = refine(best_points, 'Powell', maxiter=5000, tol=1e-10)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
        
        # Stage 4: Feasibility-based refinement
        if time.time() < deadline - 7.0:
            try:
                center = np.mean(best_points, axis=0)
                pts_norm = best_points - center
                max_norm = np.max(np.linalg.norm(pts_norm, axis=1))
                if max_norm > 0:
                    pts_norm = pts_norm / max_norm
                res_feas = minimize(_objective_feasibility, pts_norm.flatten(),
                                    method='L-BFGS-B',
                                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11})
                cand_pts = res_feas.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
            except Exception:
                pass
        
        # Stage 5: If best is symmetric, refine in symmetric space with both objectives
        if time.time() < deadline - 5.0:
            try:
                # Check if best is approximately centrally symmetric
                center = np.mean(best_points, axis=0)
                pts_centered = best_points - center
                # Find pairing: for each point, find its antipode
                sym_params_est = []
                used = set()
                for i in range(N_HALF):
                    best_j = -1
                    best_dist = np.inf
                    for j in range(N_HALF, N):
                        if j in used:
                            continue
                        d = np.linalg.norm(pts_centered[i] + pts_centered[j])
                        if d < best_dist:
                            best_dist = d
                            best_j = j
                    if best_j >= 0:
                        used.add(best_j)
                        mid = (pts_centered[i] + pts_centered[best_j]) / 2
                        v = (pts_centered[i] - pts_centered[best_j]) / 2
                        angle = np.arctan2(v[1], v[0])
                        radius = np.linalg.norm(v)
                        sym_params_est.extend([angle, max(radius, 0.1)])
                if len(sym_params_est) == 14:
                    sym_flat = np.array(sym_params_est)
                    # Try hard symmetric objective
                    res_sym_final = minimize(_objective_sym, sym_flat, method='Nelder-Mead',
                                             options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
                    if res_sym_final.fun < 0:
                        pts14 = _sym_params_to_points(res_sym_final.x)
                        cand_score = -res_sym_final.fun
                        update_best(pts14, cand_score)
                    
                    # Also try feasibility-based symmetric objective
                    if time.time() < deadline - 4.0:
                        res_sym_feas = minimize(_objective_sym_feasibility, sym_flat, method='L-BFGS-B',
                                                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11})
                        pts14f = _sym_params_to_points(res_sym_feas.x)
                        cand_scoref = _score(pts14f)
                        update_best(pts14f, cand_scoref)
            except Exception:
                pass

    # 6. Random restarts if time allows
    if best_points is not None and time.time() < deadline - 12.0:
        rng2 = np.random.default_rng(42)
        for k in range(10):
            if time.time() > deadline - 7.0:
                break
            noise = rng2.normal(0, 0.04 + 0.015*k, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=1200)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
            
            # Also try soft refinement on perturbed
            if time.time() < deadline - 6.0:
                cand_pts2, cand_score2 = refine_soft(perturbed, method='L-BFGS-B', maxiter=2000, beta=100.0)
                if cand_pts2 is not None:
                    update_best(cand_pts2, cand_score2)

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
