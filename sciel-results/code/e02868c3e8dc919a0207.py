import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice of points inside the unit circle."""
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    k = int(1.0 / s) + 2
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    points += np.random.normal(0, s * 0.01, size=points.shape)
    return points

def _radial_ring_init(n_points):
    """Generate points in concentric rings with hexagonal-like spacing."""
    # Estimate ring radii and points per ring
    # Area of annulus between r_k and r_{k+1} should accommodate points
    # For hex packing, density ~ 2/(sqrt(3)*s^2) per unit area
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    points = [[0.0, 0.0]]
    remaining = n_points - 1
    
    ring_idx = 1
    while remaining > 0:
        # Ring radius: approximately ring_idx * s * sqrt(3)/2
        r = ring_idx * s * np.sqrt(3) / 2
        if r > 0.95:
            break
        
        # Number of points in this ring: circumference / (s * sqrt(3)/2)
        n_in_ring = max(6, int(2 * np.pi * r / (s * np.sqrt(3) / 2)))
        
        # Don't exceed remaining
        n_in_ring = min(n_in_ring, remaining)
        
        # Place points evenly around the ring with slight radial jitter
        angles = 2 * np.pi * np.arange(n_in_ring) / n_in_ring
        # Offset alternating rings by pi/n for hex-like packing
        if ring_idx % 2 == 0:
            angles += np.pi / n_in_ring
            
        for angle in angles:
            # Add small radial noise
            r_noise = r + np.random.normal(0, s * 0.02)
            r_noise = max(0.01, min(0.99, r_noise))
            x = r_noise * np.cos(angle)
            y = r_noise * np.sin(angle)
            points.append([x, y])
        
        remaining -= n_in_ring
        ring_idx += 1
    
    points = np.array(points[:n_points])
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    return min(min_pair, min_wall)

def _force_directed_relax(points, n_steps=50, k_repel=1.0, k_wall=0.5):
    """Simple force-directed relaxation to resolve overlaps."""
    pts = points.copy()
    n = len(pts)
    
    # Estimate target distance
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3)))
    
    for step in range(n_steps):
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion (only for overlapping or close pairs)
        D = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        L = np.linalg.norm(D, axis=2)
        
        # Only consider pairs within 2*s
        mask = (L < 2 * s) & (L > 1e-10)
        
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    dist = L[i, j]
                    # Repulsive force: stronger when closer
                    f = k_repel * (1.0 / dist - 1.0 / (2 * s))
                    if f > 0:
                        direction = D[i, j] / dist
                        forces[i] += f * direction
                        forces[j] -= f * direction
        
        # Wall attraction (pull points inward if too close to boundary)
        L_wall = np.linalg.norm(pts, axis=1)
        for i in range(n):
            if L_wall[i] > 0.9:
                f = k_wall * (L_wall[i] - 0.9)
                direction = -pts[i] / L_wall[i]
                forces[i] += f * direction
        
        # Apply forces with damping
        pts += 0.1 * forces
        
        # Keep points inside unit circle
        L_wall = np.linalg.norm(pts, axis=1)
        mask_out = L_wall > 1.0
        if np.any(mask_out):
            pts[mask_out] /= L_wall[mask_out, np.newaxis]
    
    return pts

def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    We want to maximize the soft-min of distances, so we minimize its negative.
    """
    C = x.reshape(N, 2)
    
    # 1. Pairwise distances
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    # Avoid division by zero
    L_safe = np.where(L > 1e-12, L, 1.0)
    
    x_pairs = L_safe / 2.0
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    x_pairs_flat = x_pairs[i_idx, j_idx]
    
    # Combine all x values
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    # Compute weights using log-sum-exp trick for stability
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    # Value: S = min_x - (1/beta) * log(Z)
    val = min_x - (1.0 / beta) * np.log(Z)
    
    # Gradient of S
    grad = np.zeros_like(C)
    
    # Pair contributions
    P_pairs = P[:len(i_idx)]
    
    D_flat = D[i_idx, j_idx]
    L_flat = L[i_idx, j_idx]
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)
    
    U = D_flat / L_flat_safe[:, np.newaxis]
    
    W = P_pairs * 0.5
    
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    # Wall contributions
    P_walls = P[len(i_idx):]
    
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    grad += P_walls[:, np.newaxis] * U_walls
    
    return -val, -grad

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for 150s limit
    
    # Try both initialization methods and pick the better one
    np.random.seed(42)
    points_hex = _hex_lattice_init(N)
    np.random.seed(42)
    points_ring = _radial_ring_init(N)
    
    # Force-directed relaxation on both
    points_hex_relaxed = _force_directed_relax(points_hex, n_steps=30)
    points_ring_relaxed = _force_directed_relax(points_ring, n_steps=30)
    
    score_hex = _compute_true_score(points_hex_relaxed)
    score_ring = _compute_true_score(points_ring_relaxed)
    
    if score_ring > score_hex:
        points = points_ring_relaxed
        best_score = score_ring
    else:
        points = points_hex_relaxed
        best_score = score_hex
    
    best_points = points.copy()
    
    beta = 10.0
    max_beta = 2000.0
    beta_factor = 1.3
    
    current_points = points.copy()
    
    while time.time() - start_time < time_limit:
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-13})
            
            new_points = res.x.reshape(N, 2)
            new_score = _compute_true_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
            
            if np.all(np.isfinite(new_points)):
                current_points = new_points
            
        except Exception:
            pass
        
        beta *= beta_factor
        if beta > max_beta:
            # Final high-precision polish
            beta = max_beta * 5
            try:
                res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                               options={'maxiter': 1000, 'gtol': 1e-9, 'ftol': 1e-15})
                new_points = res.x.reshape(N, 2)
                new_score = _compute_true_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            except Exception:
                pass
            break

    return best_points
