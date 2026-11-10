import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Assume points lie on the unit sphere to minimize d_max.
    2. Use a smooth surrogate objective (sum of inverse squared distances) for gradient-based optimization.
    3. Polish with the exact non-smooth objective using Nelder-Mead.
    4. Use multiple restarts with diverse initializations (random, icosahedron-based).
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 10s margin from 630s limit

    def compute_score(pts):
        # pts is (n, 3)
        # Normalize to unit sphere to enforce constraint during optimization
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts_norm = pts / norms
        
        dists = pdist(pts_norm, metric='sqeuclidean')
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        if d_max_sq < 1e-12:
            return 0.0, pts_norm
            
        # Ratio of distances is sqrt(ratio of sq distances)
        ratio = np.sqrt(d_min_sq / d_max_sq)
        score = ratio * ratio # (d_min/d_max)^2
        return score, pts_norm

    def smooth_objective(params):
        # params is flat array of size n*3
        pts = params.reshape(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts_norm = pts / norms
        
        # Compute pairwise squared distances
        dists_sq = pdist(pts_norm, metric='sqeuclidean')
        
        # Avoid division by zero
        dists_sq = np.maximum(dists_sq, 1e-6)
        
        # Objective: Minimize sum of inverse squared distances
        # This pushes points apart
        obj = np.sum(1.0 / dists_sq)
        return obj

    def exact_objective(params):
        pts = params.reshape(n, 3)
        score, _ = compute_score(pts)
        return -score

    best_score = 0.0
    best_points = None

    # Initializations
    inits = []
    
    # 1. Icosahedron vertices + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 points on z-axis (symmetry axis)
    for z1 in [0.0, 0.5, 0.8, 1.0, -0.5, -0.8, -1.0]:
        for z2 in [0.0, 0.5, 0.8, 1.0, -0.5, -0.8, -1.0]:
            if np.linalg.norm([0,0,z1] - [0,0,z2]) < 1e-6:
                continue
            p1 = np.array([0, 0, z1])
            p2 = np.array([0, 0, z2])
            p1 = p1 / np.linalg.norm(p1) if np.linalg.norm(p1) > 1e-6 else np.array([0,0,1])
            p2 = p2 / np.linalg.norm(p2) if np.linalg.norm(p2) > 1e-6 else np.array([0,0,-1])
            pts = np.vstack([ico_verts, p1, p2])
            inits.append(pts.flatten())

    # 2. Random points on sphere
    np.random.seed(42)
    for _ in range(50):
        pts = np.random.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts.flatten())

    # 3. Perturbed Icosahedron
    for _ in range(20):
        pts = ico_verts.copy()
        # Add 2 random points
        p1 = np.random.randn(3)
        p2 = np.random.randn(3)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([pts, p1, p2])
        # Perturb all points
        noise = np.random.randn(n, 3) * 0.1
        pts += noise
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts.flatten())

    # Optimization Loop
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            # Step 1: Smooth optimization (L-BFGS-B is good for smooth objectives)
            # We optimize the raw coordinates, but the objective normalizes them.
            # This effectively optimizes on the sphere.
            res_smooth = minimize(
                smooth_objective, 
                init, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            best_params = res_smooth.x
            current_score, current_pts = compute_score(best_params)
            
            # Step 2: Polish with Nelder-Mead on the exact objective
            # NM doesn't use gradients, good for non-smooth min/max
            res_exact = minimize(
                exact_objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            final_params = res_exact.x
            final_score, final_pts = compute_score(final_params)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts

        except Exception:
            continue

    # Fallback
    if best_points is None:
        # Use the icosahedron + 2 poles as a safe fallback
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p1 = np.array([0, 0, 1])
        p2 = np.array([0, 0, -1])
        best_points = np.vstack([ico_verts, p1, p2])
        
    return best_points
