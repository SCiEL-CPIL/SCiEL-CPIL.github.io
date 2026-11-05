import numpy as np
import time
from scipy.optimize import minimize
import math

def _log_score(points):
    """
    Compute log(Delta) for a set of points.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    Delta = prod_{i!=j} (d_ij / D) where D is the diameter.
    log(Delta) = sum_{i!=j} log(d_ij) - 56 * log(D)
    """
    points = np.array(points)
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 0):
        return -np.inf
        
    D = np.max(valid_dists)
    if D <= 0:
        return -np.inf
        
    # log(Delta) = sum(log(d_ij)) - 56 * log(D)
    # Note: The score is Delta / 8^8. 
    # We maximize log(Delta) - 8*log(8). 
    # Since 8*log(8) is constant, maximizing log(Delta) is equivalent.
    # However, the problem asks for Delta/n^n. 
    # Let's stick to maximizing log(Delta) for optimization, 
    # and compute final score as exp(log_Delta - 8*log(8)).
    
    return np.sum(np.log(valid_dists)) - 56 * np.log(D)

def _score(points):
    """
    Compute the actual score Delta / 8^8.
    """
    log_val = _log_score(points)
    if not np.isfinite(log_val):
        return 0.0
    return np.exp(log_val - 8 * np.log(8))

def _mds_2d(dist_matrix):
    """
    Recover 2D coordinates from a distance matrix using Classical MDS.
    dist_matrix: (8, 8) matrix of distances.
    Returns: (8, 2) array of coordinates.
    """
    n = dist_matrix.shape[0]
    # Square the distances
    D2 = dist_matrix ** 2
    
    # Double centering
    # J = I - (1/n) * 11^T
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    
    # Eigendecomposition
    # We want the top 2 eigenvalues
    # B is symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2
    # If eigenvalues are negative, set to 0
    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    
    # Coordinates
    coords = eigenvectors[:, :2] * sqrt_eigenvalues[:2]
    
    return coords

def _neg_smooth_log_score_coords(x_flat, beta=100.0):
    """
    Objective for coordinate optimization with smooth diameter constraint.
    x_flat: 16 floats (8 points * 2 coords)
    """
    points = x_flat.reshape(8, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    mask = ~np.eye(8, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 1e-10):
        return 1e10
        
    log_dists = np.log(valid_dists)
    sum_log_dists = np.sum(log_dists)
    
    # Smooth max of log(dists)
    # max(x) approx (1/beta) * log(sum(exp(beta * x)))
    max_log_dist = (1.0 / beta) * np.log(np.sum(np.exp(beta * log_dists)))
    
    # log(Delta) = sum_log_dists - 56 * max_log_dist
    log_delta = sum_log_dists - 56 * max_log_dist
    
    if not np.isfinite(log_delta):
        return 1e10
        
    return -log_delta

def _neg_smooth_log_score_dist(y, beta=50.0):
    """
    Objective for distance-space optimization.
    y: log of 28 pairwise distances (i<j).
    We reconstruct the full distance matrix, check PSD condition (approx), 
    and compute log(Delta).
    """
    # 28 distances for 8 points
    # Indices for i < j
    i_idx, j_idx = np.triu_indices(8, k=1)
    d_vals = np.exp(y)
    
    # Reconstruct full distance matrix
    D = np.zeros((8, 8))
    D[i_idx, j_idx] = d_vals
    D[j_idx, i_idx] = d_vals
    
    # Diameter is max(D)
    D_max = np.max(D)
    
    # log(Delta) = sum_{i!=j} log(d_ij) - 56 * log(D_max)
    # sum_{i!=j} log(d_ij) = 2 * sum_{i<j} log(d_ij) = 2 * sum(y)
    sum_log_d = 2 * np.sum(y)
    
    # Smooth max of log(d_ij) for i!=j
    # We have 56 values in D (excluding diagonal). 
    # But we only have 28 unique. 
    # log(D_max) = max(log(D[i,j]) for i!=j)
    # Since D is symmetric, max is same as max of the 28 unique values.
    log_d_vals = y
    max_log_d = (1.0 / beta) * np.log(np.sum(np.exp(beta * log_d_vals)))
    
    log_delta = sum_log_d - 56 * max_log_d
    
    # Penalty for non-PSD / non-Euclidean distance matrix
    # Check if G = -0.5 J D^2 J is PSD
    D2 = D ** 2
    J = np.eye(8) - np.ones((8, 8)) / 8
    G = -0.5 * J @ D2 @ J
    
    # Eigenvalues of G should be >= 0
    eigvals = np.linalg.eigvalsh(G)
    min_eig = np.min(eigvals)
    
    # If min_eig is significantly negative, penalize
    if min_eig < -1e-5:
        penalty = -100.0 * min_eig
    else:
        penalty = 0.0
        
    return -log_delta + penalty

def _regular_octagon():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _two_concentric_squares(r1=1.0, r2=0.7):
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    angles2 = angles1 + np.pi/4
    p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([p1, p2])

def _random_initial(rng):
    # Random points in disk
    radii = rng.uniform(0.5, 1.5, 8)
    angles = rng.uniform(0, 2 * np.pi, 8)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = 0.0
    best_points = _regular_octagon()
    
    # Update best with regular octagon
    s = _score(best_points)
    if s > best_score:
        best_score = s
    
    rng = np.random.default_rng(123)
    
    # Strategy:
    # 1. Generate candidate distance matrices via optimization in log-distance space.
    # 2. Embed via MDS.
    # 3. Polish in coordinate space using L-BFGS-B with smooth diameter.
    
    # Pre-compute indices for distance vector
    i_idx, j_idx = np.triu_indices(8, k=1)
    
    while time.time() < deadline:
        # Phase 1: Distance space optimization
        # Start from regular octagon distances
        pts_init = _regular_octagon()
        diff = pts_init[:, np.newaxis, :] - pts_init[np.newaxis, :, :]
        dists_init = np.sqrt(np.sum(diff**2, axis=2))
        y0 = np.log(dists_init[i_idx, j_idx])
        
        # Add small noise
        y0_noisy = y0 + rng.normal(0, 0.1, 28)
        
        try:
            # Optimize in distance space
            res_dist = minimize(_neg_smooth_log_score_dist, y0_noisy, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res_dist.fun < 1e9:
                y_opt = res_dist.x
                d_vals = np.exp(y_opt)
                
                # Reconstruct D
                D = np.zeros((8, 8))
                D[i_idx, j_idx] = d_vals
                D[j_idx, i_idx] = d_vals
                
                # Check if this distance matrix is Euclidean (PSD Gram matrix)
                D2 = D ** 2
                J = np.eye(8) - np.ones((8, 8)) / 8
                G = -0.5 * J @ D2 @ J
                eigvals = np.linalg.eigvalsh(G)
                
                if np.min(eigvals) > -1e-4:
                    # Embed
                    pts_embedded = _mds_2d(D)
                    
                    # Phase 2: Coordinate space polish
                    # Use L-BFGS-B for better convergence
                    x0 = pts_embedded.flatten()
                    
                    # Try a few betas for smoothing
                    best_local_score = 0
                    best_local_pts = pts_embedded
                    
                    for beta in [50.0, 100.0, 200.0]:
                        try:
                            res_coord = minimize(_neg_smooth_log_score_coords, x0, method='L-BFGS-B',
                                                 args=(beta,), options={'maxiter': 200, 'gtol': 1e-6})
                            if res_coord.fun < 1e9:
                                pts_polished = res_coord.x.reshape(8, 2)
                                s_polished = _score(pts_polished)
                                if s_polished > best_local_score:
                                    best_local_score = s_polished
                                    best_local_pts = pts_polished
                        except Exception:
                            pass
                    
                    if best_local_score > best_score:
                        best_score = best_local_score
                        best_points = best_local_pts
                        
        except Exception:
            pass
            
        # Alternative: Direct coordinate optimization from diverse starts
        # This is a fallback if distance space fails or to explore other basins
        if time.time() < deadline:
            # Generate a diverse initial point
            r = rng.random()
            if r < 0.3:
                init_pts = _regular_octagon() + rng.normal(0, 0.1, (8, 2))
            elif r < 0.6:
                init_pts = _two_concentric_squares(rng.uniform(0.8, 1.2), rng.uniform(0.5, 0.9))
            else:
                init_pts = _random_initial(rng)
            
            x0 = init_pts.flatten()
            try:
                # Use Nelder-Mead for robustness, then L-BFGS-B for polish
                res_nm = minimize(_neg_smooth_log_score_coords, x0, method='Nelder-Mead',
                                  args=(100.0,), options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
                
                if res_nm.fun < 1e9:
                    x_nm = res_nm.x
                    # Polish with L-BFGS-B
                    res_lbfgs = minimize(_neg_smooth_log_score_coords, x_nm, method='L-BFGS-B',
                                         args=(200.0,), options={'maxiter': 100, 'gtol': 1e-7})
                    
                    if res_lbfgs.fun < 1e9:
                        pts_final = res_lbfgs.x.reshape(8, 2)
                        s_final = _score(pts_final)
                        if s_final > best_score:
                            best_score = s_final
                            best_points = pts_final
            except Exception:
                pass

    # Final check: ensure points are distinct and valid
    # Scale to diameter 2? The evaluator scales, so we can return as is.
    # But to be safe, let's return the points that gave the best score.
    
    return best_points.tolist()

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_score(np.array(pts))}")
