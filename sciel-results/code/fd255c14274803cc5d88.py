import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools
from itertools import combinations

N = 13

# Precompute all triples for vectorized area computation
_TRIPLES = list(combinations(range(N), 3))
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])
_N_TRIPLES = len(_TRIPLES)

def _all_triangle_areas(pts):
    """Compute all C(n,3) triangle areas efficiently using vectorization."""
    x = pts[:, 0]
    y = pts[:, 1]
    v1x = x[_J_IDX] - x[_I_IDX]
    v1y = y[_J_IDX] - y[_I_IDX]
    v2x = x[_K_IDX] - x[_I_IDX]
    v2y = y[_K_IDX] - y[_I_IDX]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    areas = _all_triangle_areas(pts)
    return np.min(areas) if len(areas) > 0 else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _soft_min_area(areas, alpha):
    """
    Soft minimum of areas using log-sum-exp.
    Returns: -1/alpha * log(sum(exp(-alpha * A_k)))
    This is a lower bound on min(A_k), tight as alpha -> inf.
    """
    if len(areas) == 0:
        return 0.0
    M = np.min(areas)
    shifted = areas - M
    # Guard against overflow
    exp_terms = np.exp(-alpha * shifted)
    if np.any(~np.isfinite(exp_terms)):
        return M
    log_sum = -alpha * M + np.log(np.sum(exp_terms))
    return -log_sum / alpha

def _neg_soft_score(params, alpha):
    pts = params.reshape(-1, 2)
    areas = _all_triangle_areas(pts)
    soft_min = _soft_min_area(areas, alpha)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return -(soft_min / hull_a)

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _concentric_params_to_points(params, n_circles, counts):
    """
    Convert parameter vector to point positions.
    params: [r1, theta1, r2, theta2, ..., rn, thetan] for each circle
    counts: list of how many points per circle
    """
    pts = []
    idx = 0
    for c in range(n_circles):
        r = params[idx]
        theta0 = params[idx + 1]
        idx += 2
        n_c = counts[c]
        if n_c == 1:
            # Single point: just place at angle theta0
            pts.append([r * np.cos(theta0), r * np.sin(theta0)])
        elif n_c > 1:
            angles = theta0 + np.linspace(0, 2 * np.pi, n_c, endpoint=False)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts.extend(zip(x, y))
    pts = np.array(pts)
    if len(pts) < N:
        # Pad with center points if necessary (should not happen with correct counts)
        pad = np.zeros((N - len(pts), 2))
        pts = np.vstack([pts, pad])
    elif len(pts) > N:
        pts = pts[:N]
    return pts

def _generate_symmetric_initializations():
    """Generate diverse concentric circle configurations for initial evaluation."""
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two-ring configurations
    splits = [(7, 6), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1), (6, 7), (5, 8)]
    for n1, n2 in splits:
        for r2 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
            ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / max(n2, 1)
            ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
            ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
            inits.append(np.vstack([ring1, ring2]))
            
    # Three-ring configurations
    triples = [(6, 4, 3), (5, 4, 4), (7, 3, 3), (8, 3, 2), (5, 5, 3), (6, 5, 2), (4, 4, 5)]
    for n1, n2, n3 in triples:
        for r2 in [0.4, 0.5, 0.6]:
            for r3 in [0.2, 0.3, 0.4]:
                ang1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                ang2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
                ang3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n3
                ring1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
                ring2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
                ring3 = np.column_stack([r3 * np.cos(ang3), r3 * np.sin(ang3)])
                inits.append(np.vstack([ring1, ring2, ring3]))
                
    return inits

def _optimize_symmetric_config(n_circles, counts, deadline, seed):
    """
    Optimize a specific symmetric configuration using Differential Evolution
    on the reduced parameter space (radii and angles).
    """
    n_params = 2 * n_circles
    
    def _neg_score_sym(params):
        pts = _concentric_params_to_points(params, n_circles, counts)
        return -_score(pts)
    
    # Bounds: radius in [0, 1.5], angle in [0, 2*pi]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.5))  # radius
        bounds.append((0.0, 2 * np.pi))  # angle offset
    
    try:
        # Use a smaller population and fewer iterations for speed, as this is called many times
        res = differential_evolution(
            _neg_score_sym,
            bounds,
            maxiter=20,
            popsize=10,
            tol=1e-4,
            seed=seed,
            polish=True,
            init='sobol'
        )
        best_params = res.x
        best_pts = _concentric_params_to_points(best_params, n_circles, counts)
        return best_pts, _score(best_pts)
    except Exception:
        return None, 0.0

def _refine_with_soft_min(pts, deadline):
    """
    Refine points using soft-min annealing.
    This helps escape local minima of the hard-min by smoothing the objective.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Anneal alpha from low to high
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
        
        try:
            def _neg_soft_closure(params, a=alpha):
                return _neg_soft_score(params, a)
            
            res = minimize(
                _neg_soft_closure,
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            soft_pts = res.x.reshape(-1, 2)
            soft_score = _score(soft_pts)
            if soft_score > current_score:
                current_pts = soft_pts
                current_score = soft_score
        except Exception:
            pass
    
    return current_pts

def _refine_with_nelder_mead(pts, deadline, maxiter=500):
    """
    Refine points using Nelder-Mead on exact score.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 3.0:
        return current_pts
    
    try:
        res = minimize(
            _neg_score,
            current_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        nm_pts = res.x.reshape(-1, 2)
        nm_score = _score(nm_pts)
        if nm_score > current_score:
            current_pts = nm_pts
            current_score = nm_score
    except Exception:
        pass
    
    return current_pts

def _targeted_worst_triangle_opt(pts, deadline):
    """
    Identify the worst (minimum area) triangles and try to improve them
    by moving the involved points.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    if time.time() > deadline - 5.0:
        return current_pts
    
    # Find all minimum-area triangles
    areas = _all_triangle_areas(current_pts)
    min_area = np.min(areas)
    
    # Find triples with area close to minimum
    threshold = min_area * 1.05
    worst_triples = []
    for idx, area in enumerate(areas):
        if area <= threshold:
            i, j, k = _TRIPLES[idx]
            worst_triples.append((i, j, k, area))
            
    # Sort by area (worst first)
    worst_triples.sort(key=lambda x: x[3])
    
    # For each worst triple, try small perturbations of its points
    rng = np.random.RandomState(42)
    for (i, j, k, area) in worst_triples[:10]:  # Limit to top 10 worst
        if time.time() > deadline - 2.0:
            break
        
        # Try moving each of the 3 points in random directions
        for trial in range(30):
            trial_pts = current_pts.copy()
            # Perturb one of the three points
            idx = [i, j, k][trial % 3]
            dx = rng.randn() * 0.05
            dy = rng.randn() * 0.05
            trial_pts[idx] += np.array([dx, dy])
            
            trial_score = _score(trial_pts)
            if trial_score > current_score:
                current_pts = trial_pts
                current_score = trial_score
    
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Phase 1: Evaluate standard symmetric initializations
    inits = _generate_symmetric_initializations()
    scored_inits = []
    for init_pts in inits:
        if time.time() > deadline - 100.0:
            break
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
    
    # Sort by score descending
    scored_inits.sort(key=lambda x: -x[0])
    
    # Phase 2: Symmetry-exploiting optimization on various circle configurations
    # This is the most promising phase: DE on reduced parameter space
    circle_configs = [
        (1, [13]),
        (2, [7, 6]),
        (2, [8, 5]),
        (2, [9, 4]),
        (2, [10, 3]),
        (2, [11, 2]),
        (2, [12, 1]),
        (3, [6, 5, 2]),
        (3, [7, 4, 2]),
        (3, [8, 3, 2]),
        (3, [5, 5, 3]),
        (3, [6, 4, 3]),
        (3, [7, 3, 3]),
        (3, [4, 4, 5]),
        (3, [5, 4, 4]),
        (4, [5, 4, 3, 1]),
        (4, [6, 3, 3, 1]),
        (4, [4, 4, 3, 2]),
        (4, [5, 3, 3, 2]),
        (4, [6, 4, 2, 1]),
    ]
    
    for n_circles, counts in circle_configs:
        if time.time() > deadline - 20.0:
            break
        seed = n_circles * 1000 + sum(counts)
        sym_pts, sym_score = _optimize_symmetric_config(n_circles, counts, deadline, seed)
        if sym_pts is not None and sym_score > best_score:
            best_score = sym_score
            best_pts = sym_pts.copy()
    
    # Phase 3: Refine top candidates with soft-min and Nelder-Mead
    # We take the best few from Phase 1 and the best from Phase 2
    candidates = []
    if len(scored_inits) > 0:
        candidates.append(scored_inits[0][1])
    candidates.append(best_pts)
    
    # Deduplicate candidates roughly
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            if np.linalg.norm(c - uc) < 1e-3:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)
    
    for pts in unique_candidates:
        if time.time() > deadline - 15.0:
            break
        
        # Soft-min annealing
        refined = _refine_with_soft_min(pts, deadline)
        
        # Nelder-Mead polish
        refined = _refine_with_nelder_mead(refined, deadline)
        
        # Targeted worst-triangle optimization
        refined = _targeted_worst_triangle_opt(refined, deadline)
        
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined.copy()
    
    # Phase 4: Full-space differential evolution if time permits (less reliable but can find non-symmetric solutions)
    if time.time() < deadline - 30.0:
        try:
            bounds = [(-1.5, 1.5)] * (2 * N)
            res_de = differential_evolution(
                _neg_score,
                bounds,
                maxiter=20,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=True,
                init='sobol'
            )
            de_pts = res_de.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 3.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    return best_pts
