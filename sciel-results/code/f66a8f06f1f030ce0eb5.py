import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _hard_objective(points_flat):
        """
        Objective function for optimization.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # --- Structural Parameterization ---
    # We hypothesize the optimal solution has a ring-like structure.
    # We parameterize points by:
    # - k: number of points on outer ring (k=1..N)
    # - r_outer, r_inner: radii
    # - theta_offset: angular shift
    # - wobble: small random perturbations to break symmetry
    
    def generate_structural_points(params):
        """
        params: [k, r_outer, r_inner, theta_offset, wobble_seed]
        k: number of outer points (integer part)
        """
        k = int(round(params[0]))
        k = np.clip(k, 1, N)
        r_outer = params[1]
        r_inner = params[2]
        theta_offset = params[3]
        wobble_scale = params[4]
        
        n_outer = k
        n_inner = N - k
        
        points = np.zeros((N, 2))
        
        # Outer ring
        if n_outer > 0:
            angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + theta_offset
            points[:n_outer, 0] = r_outer * np.cos(angles_outer)
            points[:n_outer, 1] = r_outer * np.sin(angles_outer)
        
        # Inner ring (or center if n_inner is small)
        if n_inner > 0:
            if n_inner == 1:
                points[n_outer, 0] = 0
                points[n_outer, 1] = 0
            else:
                # Inner ring offset by half step to interleave
                angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + theta_offset + np.pi/n_inner
                points[n_outer:, 0] = r_inner * np.cos(angles_inner)
                points[n_outer:, 1] = r_inner * np.sin(angles_inner)
        
        # Add wobble
        if wobble_scale > 1e-6:
            rng = np.random.RandomState(int(wobble_scale * 10000) % 10000)
            points += rng.randn(N, 2) * wobble_scale
            
        return points

    def structural_objective(params):
        """
        Objective for structural search.
        params: [k, r_outer, r_inner, theta_offset, wobble_scale]
        """
        pts = generate_structural_points(params)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _refine_nelder_mead(points):
        """Nelder-Mead refinement on full 26D space."""
        try:
            res = opt.minimize(
                _hard_objective,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            return candidate, score
        except Exception:
            return points, compute_score(points)

    def _refine_powell(points):
        """Powell refinement for the hard objective."""
        try:
            res = opt.minimize(
                _hard_objective,
                points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            return candidate, score
        except Exception:
            return points, compute_score(points)

    best_score = -float('inf')
    best_points = None

    # Phase 1: Structural Search using CMA-ES (simulated via differential evolution or Nelder-Mead on low-dim)
    # Since scipy doesn't have built-in CMA-ES, we use a robust global optimizer on the 5D structural space.
    # We will use Differential Evolution (DE) for the structural parameters.
    
    # Bounds for structural parameters:
    # k: [1, 13]
    # r_outer: [0.5, 1.5]
    # r_inner: [0.0, 1.5]
    # theta_offset: [0, 2*pi]
    # wobble_scale: [0, 0.2]
    
    bounds = [
        (1, 13),
        (0.5, 1.5),
        (0.0, 1.5),
        (0, 2 * np.pi),
        (0, 0.2)
    ]
    
    # Run DE for structural search
    # Budget: ~60s for structural search
    de_time_limit = 60.0
    
    try:
        # Use a moderate population size for DE
        res_de = opt.minimize(
            structural_objective,
            x0=np.array([7.0, 1.0, 0.5, 0.0, 0.0]),
            method='differential_evolution',
            bounds=bounds,
            options={
                'maxiter': 50,
                'popsize': 20,
                'tol': 1e-6,
                'seed': 42,
                'polish': False, # We will polish manually
                'init': 'sobol'
            },
            callback=lambda x, convergence: (
                convergence if (time.time() - start_time) > de_time_limit else None
            )
        )
        
        if res_de.x is not None:
            best_struct_params = res_de.x
            # Generate points from best structural params
            pts_struct = generate_structural_points(best_struct_params)
            pts_struct = _normalize_points(pts_struct)
            score_struct = compute_score(pts_struct)
            
            if score_struct > best_score:
                best_score = score_struct
                best_points = pts_struct.copy()
                
    except Exception as e:
        print(f"DE failed: {e}")

    # Phase 2: Local Refinement of the best structural candidate
    # Also try refining a few diverse structural candidates (best 3 from DE history if available, else just best)
    
    # For simplicity and robustness, we refine the best structural point found.
    # We also run a few random structural seeds in parallel if time permits.
    
    candidates_to_refine = []
    if best_points is not None:
        candidates_to_refine.append(best_points)
        
    # Add a few diverse manual structural guesses to the mix for refinement
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, 13, endpoint=False)
    pts_reg13 = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates_to_refine.append(_normalize_points(pts_reg13))
    
    # 2. 7 outer, 6 inner
    angles_o = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_i = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    pts_76 = np.vstack([
        np.column_stack([np.cos(angles_o), np.sin(angles_o)]),
        np.column_stack([0.5 * np.cos(angles_i), 0.5 * np.sin(angles_i)])
    ])
    candidates_to_refine.append(_normalize_points(pts_76))
    
    # 3. 12 outer, 1 center
    angles_o = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts_121 = np.vstack([
        np.column_stack([np.cos(angles_o), np.sin(angles_o)]),
        np.array([[0.0, 0.0]])
    ])
    candidates_to_refine.append(_normalize_points(pts_121))

    # Refine each candidate with Nelder-Mead and Powell
    for i, cand in enumerate(candidates_to_refine):
        if time.time() > start_time + time_limit - 10:
            break
            
        # Nelder-Mead (good for non-smooth, low-dim, but 26D is borderline. Use adaptive)
        # We split the remaining time among candidates
        remaining_time = start_time + time_limit - 10 - time.time()
        if remaining_time <= 0:
            break
            
        # Try Nelder-Mead
        if time.time() < start_time + time_limit - 5:
            pts_nm, score_nm = _refine_nelder_mead(cand)
            if score_nm > best_score:
                best_score = score_nm
                best_points = pts_nm.copy()
                
        # Try Powell on the result of NM (or original if NM failed)
        if time.time() < start_time + time_limit - 2:
            start_pw = best_points if best_points is not None and score_nm > compute_score(cand) else cand
            # Actually, always start from the better of original and NM
            score_orig = compute_score(cand)
            if score_nm > score_orig:
                start_pw = best_points
            else:
                start_pw = cand
                
            pts_pw, score_pw = _refine_powell(start_pw)
            if score_pw > best_score:
                best_score = score_pw
                best_points = pts_pw.copy()

    # Fallback
    if best_points is None:
        best_points = _normalize_points(np.column_stack([np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)), 
                                                          np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False))]))
        best_score = compute_score(best_points)

    return best_points
