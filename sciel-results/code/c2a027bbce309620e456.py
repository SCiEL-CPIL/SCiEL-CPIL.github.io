import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Diverse initial configurations (sphere, icosahedron, random, grid-like).
    2. For each configuration, multi-phase optimization:
       a. Barrier-function smooth surrogate with L-BFGS-B (log-barrier for max-dist constraint).
       b. Nelder-Mead polish on true (d_min/d_max)^2 objective.
       c. High-precision BFGS polish on barrier objective.
    3. Additional perturbation-based restarts if time permits.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 615  # 15s margin before 630s kill

    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        return d_min_sq / d_max_sq

    def barrier_objective(points):
        """
        Smooth barrier objective: -log(d_min_sq) + soft_barrier(d_max_sq)
        We want to maximize d_min_sq while keeping d_max_sq <= 1.
        Normalization: we'll normalize so that RMS = 1, and the barrier
        penalizes distances exceeding a threshold.
        """
        dists_sq = pdist(points, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        d_min_sq = np.min(dists_sq)
        
        # Soft barrier for d_max_sq <= 1.0
        # Use log-sum-exp style: sum(log(1 + exp(beta*(d_ij_sq - 1))))/beta
        # This is 0 when all d_ij_sq << 1, and grows when any d_ij_sq > 1
        beta = 30.0
        excess = dists_sq - 1.0
        # Only compute barrier for terms near or above the bound to save time
        mask = excess > -0.5
        if np.any(mask):
            barrier = np.sum(np.logaddexp(0, beta * excess[mask])) / beta
        else:
            barrier = 0.0
        
        return -np.log(d_min_sq) + barrier

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
    
    # 1. Random points on unit sphere
    for _ in range(10):
        pts = np.random.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts /= norms
        initial_configs.append(pts)

    # 2. Icosahedron-based (12 vertices) + 2 more
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    for _ in range(4):
        p1 = np.random.randn(1, 3)
        p1 /= np.linalg.norm(p1)
        p2 = np.random.randn(1, 3)
        p2 /= np.linalg.norm(p2)
        initial_configs.append(np.vstack([ico_verts, p1, p2]))
    
    # 3. Random Gaussian points (varied scales)
    for scale in [0.3, 0.5, 0.7, 1.0]:
        pts = np.random.randn(n, d) * scale
        initial_configs.append(pts)
    
    # 4. Structured: points on two concentric shells
    for _ in range(3):
        shell1 = np.random.randn(7, d)
        shell1 /= np.linalg.norm(shell1, axis=1, keepdims=True)
        shell2 = np.random.randn(7, d)
        shell2 /= np.linalg.norm(shell2, axis=1, keepdims=True)
        shell2 *= 0.5  # inner shell
        initial_configs.append(np.vstack([shell1, shell2]))
    
    # 5. Grid-like deterministic starting points
    for _ in range(3):
        pts = np.zeros((n, d))
        indices = np.random.choice(n, n, replace=False)
        for i in range(n):
            idx = indices[i]
            pts[i] = np.array([
                np.sin(idx * 0.7 + 0.3),
                np.cos(idx * 1.3 + 0.5),
                np.sin(idx * 2.1 + 0.1)
            ])
        initial_configs.append(pts)

    # Optimization Loop
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
            
        try:
            # Phase 1: Barrier function optimization with L-BFGS-B
            def constrained_barrier_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return barrier_objective(pts)

            res_smooth = minimize(
                constrained_barrier_obj, 
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

            res_nm = minimize(
                nm_obj, 
                current_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True}
            )
            
            final_pts = normalize_rms(res_nm.x.reshape(n, d))
            score_nm = true_score(final_pts)
            
            # Phase 3: High-precision BFGS polish on barrier objective
            if time.time() < deadline - 25:
                def final_polish_obj(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return barrier_objective(pts)
                
                res_polish = minimize(
                    final_polish_obj,
                    final_pts.flatten(),
                    method='BFGS',
                    options={'maxiter': 300, 'gtol': 1e-14}
                )
                
                polished_pts = normalize_rms(res_polish.x.reshape(n, d))
                polished_score = true_score(polished_pts)
                
                if polished_score > score_nm:
                    final_pts = polished_pts
                    score_nm = polished_score
            
            # Calculate score
            if score_nm > best_score:
                best_score = score_nm
                best_points = final_pts.copy()
                
        except Exception:
            continue

    # Additional perturbation-based restarts if time permits
    if best_points is not None and time.time() < deadline - 40:
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            try:
                # Start from best points with small perturbation
                perturbation_scale = 0.01 * (1.0 + 0.5 * restart)
                perturbed = best_points + np.random.randn(n, d) * perturbation_scale
                perturbed = normalize_rms(perturbed)
                
                def nm_obj2(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return -true_score(pts)
                
                res_nm2 = minimize(
                    nm_obj2, 
                    perturbed.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                             'adaptive': True}
                )
                
                final_pts2 = normalize_rms(res_nm2.x.reshape(n, d))
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
        ], dtype=float)
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
        ], dtype=float)
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        best_points = np.vstack([ico_verts, np.array([0.0,0.0,1.0]), np.array([0.0,0.0,-1.0])])

    return best_points
