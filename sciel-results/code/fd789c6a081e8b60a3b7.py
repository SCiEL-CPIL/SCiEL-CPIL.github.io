"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import random

N = 13


def _compute_score(pts):
    """Compute min triangle area / convex hull area."""
    n = pts.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Cross product magnitude / 2
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _score_fast(pts):
    """Vectorized score computation."""
    n = pts.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    
    # Vectorized triangle area computation
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    pi = pts[i_idx[mask]]
    pj = pts[j_idx[mask]]
    pk = pts[k_idx[mask]]
    
    cross = np.abs((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - 
                   (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
    min_area = 0.5 * np.min(cross)
    
    return min_area / hull_area


def _nelder_mead_minimize(pts, max_iter=200, tol=1e-10):
    """Local optimization using Nelder-Mead on the negative log-sum-exp surrogate."""
    n = pts.shape[0]
    
    def surrogate(params):
        """Log-sum-exp surrogate for min triangle area (smooth approximation)."""
        p = params.reshape(n, 2)
        # Compute all triangle areas
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        
        pi = p[i_idx[mask]]
        pj = p[j_idx[mask]]
        pk = p[k_idx[mask]]
        
        cross = np.abs((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - 
                       (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0]))
        areas = 0.5 * cross
        
        # Log-sum-exp with large beta to approximate min
        beta = 500.0
        log_sum = np.log(np.sum(np.exp(-beta * areas)))
        return -log_sum / beta  # Negative to minimize
    
    # Initial simplex
    x0 = pts.flatten()
    simplex = [x0.copy()]
    for i in range(2 * n):
        perturbation = x0.copy()
        perturbation[i % (2 * n)] += 0.01
        simplex.append(perturbation)
    
    # Nelder-Mead implementation
    best_x = x0.copy()
    best_f = surrogate(x0)
    
    for iteration in range(max_iter):
        # Sort simplex by function value
        fvals = [surrogate(x) for x in simplex]
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - simplex[-1])
        f_ref = surrogate(x_ref)
        
        if f_ref < fvals[0]:
            # Expansion
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - simplex[-1])
            f_exp = surrogate(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        else:
            # Contraction
            rho = 0.5
            x_con = centroid + rho * (simplex[-1] - centroid)
            f_con = surrogate(x_con)
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    
    best_x = simplex[np.argmin([surrogate(x) for x in simplex])]
    return best_x.reshape(n, 2)


def construct():
    """Construct 13 points to maximize min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Multi-start with simulated annealing
    n_starts = 30
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Initialize with structured points + random perturbation
        if start_idx == 0:
            # Regular grid-like initialization
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.ones(N) * 0.8
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            pts += np_rng.normal(0, 0.05, size=(N, 2))
        elif start_idx == 1:
            # Two concentric rings
            angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
            pts = np.vstack([
                np.column_stack([0.9 * np.cos(angles_outer), 0.9 * np.sin(angles_outer)]),
                np.column_stack([0.4 * np.cos(angles_inner), 0.4 * np.sin(angles_inner)])
            ])
            pts += np_rng.normal(0, 0.03, size=(N, 2))
        else:
            # Random initialization within unit disk
            theta = np_rng.uniform(0, 2 * np.pi, N)
            r = np.sqrt(np_rng.uniform(0, 1, N)) * 0.9
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        # Simulated annealing
        T = 0.1
        T_min = 1e-6
        cooling = 0.95
        max_iter_sa = 500
        
        current_pts = pts.copy()
        current_score = _score_fast(current_pts)
        best_local_pts = current_pts.copy()
        best_local_score = current_score
        
        for sa_iter in range(max_iter_sa):
            if time.time() > deadline:
                break
            
            # Generate perturbation
            step_size = 0.05 * (T / 0.1) + 0.005
            new_pts = current_pts + np_rng.normal(0, step_size, size=(N, 2))
            
            new_score = _score_fast(new_pts)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_pts = current_pts.copy()
            else:
                delta = current_score - new_score
                if delta < 1e-10 or np_rng.random() < np.exp(-delta / T):
                    current_pts = new_pts
                    current_score = new_score
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Local refinement with Nelder-Mead
        if time.time() < deadline - 10:
            refined_pts = _nelder_mead_minimize(best_local_pts, max_iter=100)
            refined_score = _score_fast(refined_pts)
            if refined_score > best_local_score:
                best_local_pts = refined_pts
                best_local_score = refined_score
        
        # Update global best
        if best_local_score > best_score:
            best_score = best_local_score
            best_pts = best_local_pts.copy()
    
    # Phase 2: Further refinement with multiple Nelder-Mead passes
    if best_pts is not None and time.time() < deadline - 5:
        for refine_pass in range(3):
            if time.time() > deadline:
                break
            refined = _nelder_mead_minimize(best_pts, max_iter=150)
            refined_score = _score_fast(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined.copy()
    
    # Final verification and return
    if best_pts is None:
        # Fallback to random points
        return np_rng.random((N, 2))
    
    final_score = _score_fast(best_pts)
    return best_pts
