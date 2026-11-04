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
    Soft-min of normalized triangle areas, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Normalize areas by hull area to get dimensionless scores
    norm_areas = areas / hull_area
    if np.any(norm_areas <= 0):
        return 1e6
    
    # Soft min using log-sum-exp trick for numerical stability
    neg_beta_areas = -beta * norm_areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_norm = -log_sum_exp / beta
    
    return -soft_min_norm

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _nelder_mead_polish(pts, max_iter=2000, tol=1e-12):
    """Polish points using Nelder-Mead on exact objective."""
    try:
        res = minimize(
            _exact_objective,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True, 'maxfev': max_iter * 10}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _initial_points(seed=0):
    """Generate initial points: regular polygon with slight perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts += rng.normal(0, 0.03, size=pts.shape)
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

def _three_ring(n1, n2, n3, r1=1.0, r2=0.65, r3=0.35):
    """Points on three concentric rings."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n3
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    return np.vstack([p1, p2, p3])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_score = 0.0
    best_pts = _initial_points(0)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_initial_points(0))
    
    # 2. Concentric rings (various splits and radii) - expanded range
    for n_outer in range(5, 11):
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
                candidates.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    
    # 3. Three-ring configurations - expanded
    for n1 in range(4, 9):
        for n2 in range(3, 9):
            n3 = N - n1 - n2
            if n3 >= 2:
                for r2 in [0.4, 0.5, 0.6, 0.7]:
                    for r3 in [0.2, 0.3, 0.4, 0.5]:
                        if r3 < r2:
                            candidates.append(_three_ring(n1, n2, n3, 1.0, r2, r3))
    
    # 4. Halton sequence points
    candidates.append(_halton_points(N))
    
    # 5. Random points on/near circle - more samples
    rng = np.random.default_rng(42)
    for i in range(50):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.4, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Perturbed regular polygon - varied noise levels
    for noise_scale in [0.05, 0.1, 0.15]:
        for i in range(20):
            noise = rng.normal(0, noise_scale, (N, 2))
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            base = np.column_stack([np.cos(angles), np.sin(angles)])
            candidates.append(base + noise)
        
    # 7. "Grid-like" or structured perturbations
    for i in range(20):
        pts = np.zeros((N, 2))
        for j in range(N):
            angle = 2 * np.pi * j / N + rng.normal(0, 0.1)
            radius = 0.7 + rng.normal(0, 0.15)
            pts[j, 0] = radius * np.cos(angle)
            pts[j, 1] = radius * np.sin(angle)
        candidates.append(pts)
    
    # 8. Square-based configurations
    for i in range(10):
        # Points near corners and edges of a square
        corners = np.array([[0.7, 0.7], [0.7, -0.7], [-0.7, -0.7], [-0.7, 0.7]])
        edge_pts = []
        for j in range(9):
            t = rng.uniform(0, 1)
            edge_idx = j % 4
            if edge_idx == 0:
                pt = [corners[0, 0], -0.7 + t * 1.4]
            elif edge_idx == 1:
                pt = [0.7 - t * 1.4, corners[1, 1]]
            elif edge_idx == 2:
                pt = [corners[2, 0], -0.7 + t * 1.4]
            else:
                pt = [0.7 - t * 1.4, corners[3, 1]]
            edge_pts.append(pt)
        pts = np.vstack([corners, np.array(edge_pts)])
        pts += rng.normal(0, 0.05, pts.shape)
        candidates.append(pts)

    # Evaluate all candidates with exact score
    for pts in candidates:
        if time.time() > deadline:
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
    
    # Take more top candidates to allow for better exploration
    top_candidates = [pts for _, pts in scored_candidates[:min(20, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        # Refine top candidates with L-BFGS-B
        num_to_refine = min(10, len(top_candidates)) if beta > 200 else min(15, len(top_candidates))
        
        for init_pts in top_candidates[:num_to_refine]:
            if time.time() > deadline - 8:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200, 'maxfun': 50000}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
                    
            except Exception:
                pass
        
        # Update top candidates: keep the best one and a few others for diversity
        if best_pts is not None:
            top_candidates = [best_pts.copy()] + [p for p in top_candidates if not np.allclose(p, best_pts, atol=1e-6)][:7]
    
    # Final Nelder-Mead polish on the best point
    if time.time() < deadline - 15:
        polished = _nelder_mead_polish(best_pts, max_iter=5000, tol=1e-14)
        pol_score = _score(polished)
        if pol_score > best_score:
            best_score = pol_score
            best_pts = polished
    
    # Additional L-BFGS-B passes with high beta if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for final_beta in [2000.0, 5000.0]:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    _soft_min_objective,
                    best_pts.flatten(),
                    args=(final_beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 300, 'maxfun': 80000}
                )
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
            except Exception:
                pass
        
        # Final high-precision Nelder-Mead polish
        if time.time() < deadline - 3:
            polished = _nelder_mead_polish(best_pts, max_iter=8000, tol=1e-15)
            pol_score = _score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_pts = polished
    
    return best_pts
# EVOLVE-BLOCK-END
