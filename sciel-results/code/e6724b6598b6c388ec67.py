import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
# C(14, 3) = 364
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

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    return -_score(pts)

def _all_areas(points):
    """Compute all triangle areas for soft-min objective."""
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

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

def _fourier_radial(theta, coeffs):
    r = np.zeros_like(theta)
    KH = len(coeffs) // 2
    r += coeffs[0]
    for k in range(1, KH + 1):
        r += coeffs[k] * np.cos(k * theta)
        r += coeffs[KH + k] * np.sin(k * theta)
    return r

def _fourier_to_points(coeffs, angles):
    radii = _fourier_radial(angles, coeffs)
    radii = np.maximum(radii, 0.01)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

def _fourier_score(coeffs, angles):
    pts = _fourier_to_points(coeffs, angles)
    return _score(pts)

KHARMONICS = 3

def _fourier_objective(flat):
    n_coeffs = 2 * KHARMONICS + 1
    coeffs = flat[:n_coeffs]
    angles = flat[n_coeffs:]
    angles = np.sort(angles)
    return -_fourier_score(coeffs, angles)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
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

    # 1. Initializations - Diverse set including ring structures
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
    
    # Jittered regular polygon
    for js in [0.05, 0.1, 0.2]:
        initializations.append(x_reg + rng.normal(0, js, size=(N, 2)))

    # 2. Refine initializations with multiple methods
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
            
        # Nelder-Mead
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=1000)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
        
        # Soft-min refinement (differentiable, good for L-BFGS-B)
        if time.time() < deadline - 25.0 and cand_pts is not None:
            cand_pts2, cand_score2 = refine_soft(cand_pts, 'L-BFGS-B', maxiter=2000, beta=50.0)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)
        
        # Powell on best so far
        if time.time() < deadline - 25.0 and best_points is not None:
            cand_pts3, cand_score3 = refine(best_points, 'Powell', maxiter=300)
            if cand_pts3 is not None:
                update_best(cand_pts3, cand_score3)

    # 3. Fourier-based optimization
    if time.time() < deadline - 40.0:
        n_coeffs = 2 * KHARMONICS + 1
        
        fourier_inits = []
        
        coeffs_circle = np.zeros(n_coeffs)
        coeffs_circle[0] = 1.0
        angles_uniform = np.linspace(0, 2*np.pi, N, endpoint=False)
        flat_init = np.concatenate([coeffs_circle, angles_uniform])
        fourier_inits.append(flat_init.copy())
        
        rng_fourier = np.random.default_rng(7)
        for _ in range(3):
            coeffs_pert = np.zeros(n_coeffs)
            coeffs_pert[0] = 1.0
            coeffs_pert[1:] += rng_fourier.normal(0, 0.05, size=n_coeffs-1)
            angles_pert = angles_uniform + rng_fourier.normal(0, 0.1, size=N)
            flat_pert = np.concatenate([coeffs_pert, angles_pert])
            fourier_inits.append(flat_pert)
        
        coeffs_ellipse = np.zeros(n_coeffs)
        coeffs_ellipse[0] = 1.0
        coeffs_ellipse[1] = 0.2
        angles_ellipse = np.linspace(0, 2*np.pi, N, endpoint=False)
        flat_ellipse = np.concatenate([coeffs_ellipse, angles_ellipse])
        fourier_inits.append(flat_ellipse)
        
        for i, flat_init in enumerate(fourier_inits):
            if time.time() > deadline - 35.0:
                break
            
            try:
                res_fm = minimize(_fourier_objective, flat_init, method='Nelder-Mead',
                                  options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
                if res_fm.fun < 0:
                    n_coeffs_local = 2 * KHARMONICS + 1
                    coeffs_opt = res_fm.x[:n_coeffs_local]
                    angles_opt = np.sort(res_fm.x[n_coeffs_local:])
                    cand_pts = _fourier_to_points(coeffs_opt, angles_opt)
                    cand_score = -res_fm.fun
                    update_best(cand_pts, cand_score)
            except Exception:
                pass
            
            if time.time() < deadline - 30.0 and best_points is not None:
                try:
                    res_pow = minimize(_fourier_objective, flat_init, method='Powell',
                                       options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
                    if res_pow.fun < 0:
                        n_coeffs_local = 2 * KHARMONICS + 1
                        coeffs_opt = res_pow.x[:n_coeffs_local]
                        angles_opt = np.sort(res_pow.x[n_coeffs_local:])
                        cand_pts = _fourier_to_points(coeffs_opt, angles_opt)
                        cand_score = -res_pow.fun
                        update_best(cand_pts, cand_score)
                except Exception:
                    pass

    # 4. Differential Evolution for global exploration
    if time.time() < deadline - 35.0:
        def de_objective(flat_points):
            pts = flat_points.reshape(N, 2)
            if np.allclose(pts, pts[0]):
                return 1.0
            return -_score(pts)
            
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            res_de = differential_evolution(
                de_objective, 
                bounds, 
                seed=42, 
                maxiter=40, 
                popsize=10, 
                tol=1e-4, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < 0:
                cand_pts = res_de.x.reshape(N, 2)
                cand_score = -res_de.fun
                update_best(cand_pts, cand_score)
        except Exception:
            pass

    # 5. Final polish on the global best with multiple strategies
    if best_points is not None and time.time() < deadline - 15.0:
        # Nelder-Mead
        cand_pts, cand_score = refine(best_points, 'Nelder-Mead', maxiter=1500)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
        
        # Soft-min L-BFGS-B
        if time.time() < deadline - 15.0:
            cand_pts2, cand_score2 = refine_soft(best_points, 'L-BFGS-B', maxiter=3000, beta=100.0)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)
        
        # Powell
        if time.time() < deadline - 10.0:
            cand_pts3, cand_score3 = refine(best_points, 'Powell', maxiter=1000)
            if cand_pts3 is not None:
                update_best(cand_pts3, cand_score3)
        
        # Another Nelder-Mead pass
        if time.time() < deadline - 5.0:
            cand_pts4, cand_score4 = refine(best_points, 'Nelder-Mead', maxiter=1000)
            if cand_pts4 is not None:
                update_best(cand_pts4, cand_score4)

    # 6. Random restarts if time allows
    if best_points is not None and time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(42)
        for _ in range(4):
            if time.time() > deadline - 5.0:
                break
            noise = rng2.normal(0, 0.1, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=500)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
            if time.time() < deadline - 5.0:
                cand_pts2, cand_score2 = refine_soft(cand_pts, 'L-BFGS-B', maxiter=1000, beta=50.0)
                if cand_pts2 is not None:
                    update_best(cand_pts2, cand_score2)

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
