import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random
import sys

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Use a diverse set of initial configurations (random, icosahedron-based, etc.).
    2. For each configuration, run a two-phase optimization:
       a. Smooth surrogate optimization (minimize sum of 1/d^2) with L-BFGS-B,
          using RMS normalization to handle scale invariance.
       b. Polish with Nelder-Mead on the true (d_min/d_max)^2 objective.
    3. Keep track of the best result and return it.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 20s margin before 630s kill

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Compute pairwise squared distances
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        ratio_sq = d_min_sq / d_max_sq
        return ratio_sq

    def smooth_objective(points):
        """
        Barrier function objective: maximize min distance subject to max distance <= 1.
        
        Uses log-barrier to enforce the upper bound constraint smoothly.
        We minimize: -log(d_min_sq) + barrier_penalty
        where barrier_penalty = mu * sum(log(1 - d_ij_sq)) for d_ij_sq < 1
        """
        dists_sq = pdist(points, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        d_min_sq = np.min(dists_sq)
        
        # Log barrier for constraint d_ij_sq <= 1 (with margin)
        margin = 1.0  # We want d_max_sq <= 1
        violations = dists_sq - margin
        # Only apply barrier to points that are close to or exceeding the bound
        close_to_bound = violations > -0.1  # Within 0.1 of the bound
        if np.any(close_to_bound):
            barrier_terms = violations[close_to_bound]
            # Soft barrier: use log(1 + exp(beta * v)) approximation
            beta = 50.0
            barrier = np.sum(np.logaddexp(0, beta * barrier_terms)) / beta
        else:
            barrier = 0.0
        
        # Objective: -log(d_min_sq) + barrier
        obj = -np.log(d_min_sq) + barrier
        return obj

    def normalize_rms(points):
        """Normalize points to have unit RMS radius."""
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    best_score = 0.0
    best_points = None

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Random points on sphere
    for _ in range(8):
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)

    # 2. Icosahedron + 2 random points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts.astype(float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    for _ in range(3):
        p1 = np.random.randn(1, 3)
        p1 /= np.linalg.norm(p1)
        p2 = np.random.randn(1, 3)
        p2 /= np.linalg.norm(p2)
        initial_configs.append(np.vstack([ico_verts, p1, p2]))
    
    # 3. Random points with small perturbation from center
    for _ in range(5):
        pts = np.random.randn(n, d) * 0.5
        initial_configs.append(pts)
    
    # 4. Grid-like starting points
    for _ in range(3):
        pts = np.zeros((n, d))
        indices = np.random.choice(n, n, replace=False)
        for i in range(n):
            pts[i] = np.array([
                np.sin(indices[i] * 0.7),
                np.cos(indices[i] * 1.3),
                np.sin(indices[i] * 2.1)
            ])
        initial_configs.append(pts)

    # Optimization Loop
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
            
        try:
            # Phase 1: Barrier function optimization with L-BFGS-B
            def constrained_smooth_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return smooth_objective(pts)

            res_smooth = minimize(
                constrained_smooth_obj, 
                init_pts.flatten(), 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            current_pts = normalize_rms(res_smooth.x.reshape(n, d))
            
            # Phase 2: Nelder-Mead Polish on True Objective
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            # Use larger initial simplex for NM
            initial_simplex = current_pts.flatten().copy()
            
            res_nm = minimize(
                nm_obj, 
                initial_simplex, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True}
            )
            
            final_pts = normalize_rms(res_nm.x.reshape(n, d))
            
            # Phase 3: High-precision polish with BFGS on barrier objective
            if time.time() < deadline - 20:
                def final_polish_obj(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return smooth_objective(pts)
                
                res_polish = minimize(
                    final_polish_obj,
                    final_pts.flatten(),
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-14}
                )
                
                polished_pts = normalize_rms(res_polish.x.reshape(n, d))
                polished_score = true_score(polished_pts)
                
                if polished_score > true_score(final_pts):
                    final_pts = polished_pts
            
            # Calculate score
            score = true_score(final_pts)
            if score > best_score:
                best_score = score
                best_points = final_pts.copy()
                
        except Exception as e:
            continue

    # If we have time, do additional random restarts with NM only
    if best_points is not None and time.time() < deadline - 30:
        for _ in range(8):
            if time.time() > deadline - 10:
                break
            try:
                # Start from best points with small perturbation
                perturbed = best_points + np.random.randn(n, d) * 0.005
                perturbed = normalize_rms(perturbed)
                
                def nm_obj2(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return -true_score(pts)
                
                res_nm2 = minimize(
                    nm_obj2, 
                    perturbed.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14,
                             'adaptive': True}
                )
                
                final_pts2 = normalize_rms(res_nm2.x.reshape(n, d))
                
                # Additional BFGS polish
                if time.time() < deadline - 15:
                    def polish_obj(x_flat):
                        pts = x_flat.reshape(n, d)
                        pts = normalize_rms(pts)
                        return smooth_objective(pts)
                    
                    res_polish2 = minimize(
                        polish_obj,
                        final_pts2.flatten(),
                        method='BFGS',
                        options={'maxiter': 100, 'gtol': 1e-14}
                    )
                    polished2 = normalize_rms(res_polish2.x.reshape(n, d))
                    if true_score(polished2) > true_score(final_pts2):
                        final_pts2 = polished2
                
                score2 = true_score(final_pts2)
                if score2 > best_score:
                    best_score = score2
                    best_points = final_pts2.copy()
            except Exception:
                continue

    # Fallback if all failed
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts = ico_verts.astype(float)
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p1 = np.array([0.0, 0.0, 1.0])
        p2 = np.array([0.0, 0.0, -1.0])
        best_points = np.vstack([ico_verts, p1, p2])

    # Final validation
    if not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts = ico_verts.astype(float)
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        best_points = np.vstack([ico_verts, np.array([0.0,0.0,1.0]), np.array([0.0,0.0,-1.0])])

    return best_points
