"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _all_areas(points):
    """Vectorized computation of all triangle areas."""
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return 0.0
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective_soft_sym(flat, beta=50.0):
    """Soft-min objective for centrally symmetric parameterization."""
    pts_half = flat.reshape(N_HALF, 2)
    pts = np.vstack([pts_half, -pts_half])
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

def _objective_sym(flat):
    """Hard objective for centrally symmetric parameterization."""
    pts_half = flat.reshape(N_HALF, 2)
    pts = np.vstack([pts_half, -pts_half])
    return -_score(pts)

def _objective_soft_full(flat, beta=50.0):
    """Soft-min objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
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

def _objective_full(flat):
    """Hard objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _normalize(pts):
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
        return pts

    def refine_sym(flat_init, method='Nelder-Mead', maxiter=2000, tol=1e-10):
        """Refine in the 14D symmetric space."""
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective_sym, flat_init, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(_objective_sym, flat_init, method='Powell',
                               options={'maxiter': maxiter, 'xatol': tol, 'ftol': tol})
            else:
                return None, -np.inf
            cand_pts = np.vstack([res.x.reshape(N_HALF, 2), -res.x.reshape(N_HALF, 2)])
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft_sym(flat_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        """Refine using soft-min in symmetric space."""
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft_sym(x, beta=beta), flat_init,
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft_sym(x, beta=beta), flat_init,
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft_sym(x, beta=beta), flat_init,
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            cand_pts = np.vstack([res.x.reshape(N_HALF, 2), -res.x.reshape(N_HALF, 2)])
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_full(flat_init, method='Nelder-Mead', maxiter=2000, tol=1e-10):
        """Refine in the full 28D space."""
        try:
            if method == 'Nelder-Mead':
                res = minimize(_objective_full, flat_init, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(_objective_full, flat_init, method='Powell',
                               options={'maxiter': maxiter, 'xatol': tol, 'ftol': tol})
            else:
                return None, -np.inf
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft_full(flat_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        """Refine using soft-min in full space."""
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft_full(x, beta=beta), flat_init,
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft_full(x, beta=beta), flat_init,
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft_full(x, beta=beta), flat_init,
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # --- Phase 1: Symmetric Differential Evolution (14D) ---
    if time.time() < deadline - 50.0:
        def de_sym_obj(flat):
            pts_half = flat.reshape(N_HALF, 2)
            pts = np.vstack([pts_half, -pts_half])
            if np.allclose(pts, pts[0]):
                return 1e10
            return -_score(pts)
        
        try:
            bounds_sym = [(-1, 1)] * (N_HALF * 2)
            de_res = differential_evolution(de_sym_obj, bounds_sym, seed=42,
                                            maxiter=100, popsize=15, tol=1e-6,
                                            polish=True, init='latinhypercube')
            if de_res.success or de_res.fun < 0:
                cand_pts = np.vstack([de_res.x.reshape(N_HALF, 2), -de_res.x.reshape(N_HALF, 2)])
                cand_score = -de_res.fun
                update_best(cand_pts, cand_score)
        except Exception:
            pass

    # --- Phase 2: Multi-start local refinement in symmetric space ---
    rng = np.random.RandomState(42)
    
    # Initial candidates
    candidates = []
    
    # Regular 14-gon projected to symmetric half
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    # Take first half as symmetric representative
    half_reg = regular[:N_HALF]
    candidates.append(half_reg.flatten())
    
    # Random symmetric starts
    for _ in range(5):
        rand_half = rng.uniform(-1, 1, size=(N_HALF, 2))
        candidates.append(rand_half.flatten())
    
    # If we have a best from DE, add it
    if best_points is not None:
        # Check if best_points is approximately symmetric
        center = np.mean(best_points, axis=0)
        best_centered = best_points - center
        # Try to find symmetric half
        half_best = best_centered[:N_HALF]
        candidates.append(half_best.flatten())
        
        # Also try full space refinement on best
        if time.time() < deadline - 30:
            flat_full = best_centered.flatten()
            for beta in [50.0, 100.0]:
                pts, sc = refine_soft_full(flat_full, method='L-BFGS-B', maxiter=500, beta=beta)
                if pts is not None:
                    update_best(pts, sc)
            pts, sc = refine_full(flat_full, method='Powell', maxiter=1000, tol=1e-10)
            if pts is not None:
                update_best(pts, sc)

    # Refine each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
        
        # Soft-min refinement first to navigate nonsmooth landscape
        for beta in [50.0, 100.0]:
            pts, sc = refine_soft_sym(cand, method='L-BFGS-B', maxiter=1000, beta=beta)
            if pts is not None:
                update_best(pts, sc)
                if sc > best_score:
                    # Update candidate with improved point for next round
                    half_pts = pts[:N_HALF]
                    cand = half_pts.flatten()
        
        # Hard refinement for precision
        pts, sc = refine_sym(cand, method='Nelder-Mead', maxiter=2000, tol=1e-10)
        if pts is not None:
            update_best(pts, sc)
        
        pts, sc = refine_sym(cand, method='Powell', maxiter=1000, tol=1e-10)
        if pts is not None:
            update_best(pts, sc)

    # --- Phase 3: Final full-space polish on best symmetric solution ---
    if best_points is not None and time.time() < deadline - 10:
        center = np.mean(best_points, axis=0)
        best_centered = best_points - center
        flat_full = best_centered.flatten()
        
        # Try a few more refinements in full space
        for beta in [100.0, 200.0]:
            pts, sc = refine_soft_full(flat_full, method='L-BFGS-B', maxiter=500, beta=beta)
            if pts is not None:
                update_best(pts, sc)
        
        pts, sc = refine_full(flat_full, method='Nelder-Mead', maxiter=1000, tol=1e-11)
        if pts is not None:
            update_best(pts, sc)
        
        pts, sc = refine_full(flat_full, method='Powell', maxiter=500, tol=1e-11)
        if pts is not None:
            update_best(pts, sc)

    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_points
# EVOLVE-BLOCK-END
