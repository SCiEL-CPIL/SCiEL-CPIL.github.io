"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                          (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return float(np.min(areas)) / hull_area

def _soft_min_objective(x, beta=100.0):
    """
    Soft-min of triangle areas / hull area, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _nelder_mead_polish(pts, max_iter=5000, tol=1e-12):
    """Nelder-Mead polish on exact objective for final refinement."""
    try:
        res = minimize(
            _exact_objective,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _three_ring(n1, n2, n3, r1=1.0, r2=0.65, r3=0.35):
    """Points on three concentric rings with offset angles."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / max(n2, 1)
    a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / max(n3, 1)
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    return np.vstack([p1, p2, p3])

def _initial_points(seed=0):
    """Generate initial points: regular polygon with slight perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n + 1):
        f /= base
        seq[i - 1] = 0.0
        j = i
        while j > 0:
            seq[i - 1] += (j % base) * f
            j //= base
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence, scaled to unit circle."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    pts = np.column_stack([x, y])
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    radii[radii < 1e-10] = 1.0
    pts = pts / radii[:, np.newaxis]
    return pts

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    best_score = 0.0
    best_pts = _initial_points(0)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon with various perturbations
    for seed in range(5):
        candidates.append(_initial_points(seed))
    
    # 2. Two concentric rings (various splits and radii)
    for n_outer in range(5, 12):
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Three concentric rings with more configurations
    for n1 in range(4, 9):
        for n2 in range(3, N - n1):
            n3 = N - n1 - n2
            if n3 >= 2:
                for r2 in [0.45, 0.55, 0.65, 0.75]:
                    for r3 in [0.15, 0.25, 0.35, 0.45]:
                        if r3 < r2:
                            candidates.append(_three_ring(n1, n2, n3, 1.0, r2, r3))
    
    # 4. Halton sequence points
    candidates.append(_halton_points(N))
    
    # 5. Random points on/near circle with more samples
    rng = np.random.default_rng(42)
    for i in range(30):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Perturbed regular polygon with various noise levels
    for noise_scale in [0.05, 0.1, 0.15, 0.2]:
        for i in range(10):
            noise = rng.normal(0, noise_scale, (N, 2))
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            base = np.column_stack([np.cos(angles), np.sin(angles)])
            candidates.append(base + noise)
        
    # 7. "Grid-like" or structured perturbations
    for i in range(15):
        pts = np.zeros((N, 2))
        for j in range(N):
            angle = 2 * np.pi * j / N + rng.normal(0, 0.15)
            radius = 0.75 + rng.normal(0, 0.15)
            pts[j, 0] = radius * np.cos(angle)
            pts[j, 1] = radius * np.sin(angle)
        candidates.append(pts)
    
    # 8. Points with angular clustering (some points closer together)
    for i in range(10):
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Create some clustering
        cluster_idx = rng.choice(N, size=3, replace=False)
        for idx in cluster_idx:
            base_angles[idx] += rng.normal(0, 0.2)
        base_angles = np.sort(base_angles)
        radii = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
        candidates.append(pts)

    # Evaluate all candidates with exact score
    for pts in candidates:
        if time.time() > deadline - 20:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Sort candidates by score, optimize top ones
    scored_candidates = []
    for pts in candidates:
        s = _score(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top candidates for optimization
    top_candidates = [pts for _, pts in scored_candidates[:min(15, len(scored_candidates))]]
    
    # Phase 1: Soft-min L-BFGS-B with progressive beta
    betas = [30.0, 80.0, 150.0, 300.0]
    
    for beta in betas:
        if time.time() > deadline - 40:
            break
        
        # Refine more candidates at lower betas, fewer at higher betas
        if beta < 100:
            num_to_refine = min(8, len(top_candidates))
        else:
            num_to_refine = min(5, len(top_candidates))
        
        for init_pts in top_candidates[:num_to_refine]:
            if time.time() > deadline - 35:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 1200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
                    
            except Exception:
                pass
        
        # Update top candidates with the best
        if best_pts is not None:
            top_candidates = [best_pts.copy()] + [p for p in top_candidates if not np.allclose(p, best_pts)][:7]
    
    # Phase 2: Nelder-Mead polish on the best solution
    if time.time() < deadline - 25:
        polished = _nelder_mead_polish(best_pts, max_iter=5000, tol=1e-13)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished.copy()
    
    # Phase 3: Final high-precision Nelder-Mead if time permits
    if time.time() < deadline - 10:
        final_polished = _nelder_mead_polish(best_pts, max_iter=8000, tol=1e-15)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_polished.copy()
    
    # Phase 4: One more ultra-fine polish if there's time
    if time.time() < deadline - 3:
        ultra_polished = _nelder_mead_polish(best_pts, max_iter=3000, tol=1e-16)
        ultra_score = _score(ultra_polished)
        if ultra_score > best_score:
            best_score = ultra_score
            best_pts = ultra_polished.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
