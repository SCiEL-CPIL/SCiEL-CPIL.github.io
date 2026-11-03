import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Efficient pairwise distance computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def get_hexagonal_init(r_approx=0.125):
    """
    Generate initial points based on a hexagonal lattice.
    This provides a good starting point for 50 circles.
    """
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    # Hexagonal lattice generation
    # Spacing s = 2 * r
    s = 2.0 * r_approx
    sqrt3 = np.sqrt(3)
    
    # Generate points in rings of hexagons
    # Ring 0: 1 point (center)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Ring 4: 24 points
    # Total: 1 + 6 + 12 + 18 + 24 = 61 points. We need 50.
    # We will take the first 50 points from a spiral-like hex generation.
    
    # Better: Just generate a large hex grid and pick the 50 closest to origin
    # that fit in a circle of radius ~0.9
    max_r = 0.9
    candidates = []
    # Hex grid coordinates (q, r)
    # Axial to Cartesian:
    # x = s * (q + r/2)
    # y = s * (sqrt(3)/2 * r)
    
    # Range for q, r to cover radius max_r
    # Approximate range
    R = int(max_r / s) + 2
    
    for r in range(-R, R+1):
        for q in range(-R, R+1):
            x = s * (q + r/2.0)
            y = s * (sqrt3/2.0 * r)
            dist = np.sqrt(x*x + y*y)
            if dist <= max_r:
                candidates.append((dist, x, y))
                
    # Sort by distance from origin
    candidates.sort(key=lambda x: x[0])
    
    # Take first N
    for i in range(N):
        if i < len(candidates):
            _, x, y = candidates[i]
            points.append([x, y])
        else:
            # Fallback if not enough candidates (should not happen)
            points.append([0.0, 0.0])
            
    return np.array(points[:N])

def soft_min_objective_and_grad(points_flat, beta=50.0):
    """
    Objective: minimize -soft_min(margins)
    Margins are:
    - Boundary: 1 - ||p_i||
    - Pairwise: ||p_i - p_j|| / 2
    
    Using log-sum-exp for smooth min.
    """
    points = points_flat.reshape(-1, 2)
    
    # Compute margins
    norms = np.linalg.norm(points, axis=1)
    f = 1.0 - norms  # N values
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    g = dists / 2.0  # N x N values
    
    # Combine all margins
    # To save memory and time, we can handle them separately or concatenate
    # Concatenating N + N^2 values is fine for N=50 (50 + 2500 = 2550)
    all_margins = np.concatenate([f.ravel(), g.ravel()])
    
    # Soft min calculation
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize soft_min, so minimize -soft_min
    # Let's compute the value and gradient
    
    # Shift for numerical stability
    max_margin = np.max(all_margins)
    shifted = beta * (max_margin - all_margins)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        return 0.0, np.zeros_like(points_flat)
        
    log_sum_exp = np.log(sum_exp)
    soft_min_val = max_margin - (1.0 / beta) * log_sum_exp
    obj = -soft_min_val
    
    # Probabilities (weights)
    p = exp_shifted / sum_exp
    
    # Gradient of soft_min w.r.t. points
    # d(soft_min)/d(x_k) = sum_j p_j * d(x_j)/d(x_k)
    # Here x are the margins.
    # d(f_i)/d(p_i) = -p_i / ||p_i||
    # d(g_ij)/d(p_i) = (p_i - p_j) / (2 * ||p_i - p_j||)
    
    grad_points = np.zeros_like(points)
    
    # 1. Boundary margins (indices 0 to N-1)
    # p[0:N] are weights for f
    for i in range(N):
        w = p[i]
        if w < 1e-10:
            continue
        norm_i = norms[i]
        if norm_i < 1e-10:
            continue
        # d(f_i)/d(p_i) = -p_i / norm_i
        grad_points[i] += w * (-points[i] / norm_i)
        
    # 2. Pairwise margins (indices N to N+N^2-1)
    # p[N:] are weights for g
    # g_ij is at index N + i*N + j
    # We iterate over pairs i < j to avoid double counting?
    # No, the gradient formula sums over all j.
    # Let's use the structure:
    # grad_points[i] += sum_j p[g_ij] * d(g_ij)/d(p_i)
    # d(g_ij)/d(p_i) = (p_i - p_j) / (2 * dist_ij)
    
    # To be efficient, we can loop i and j.
    # For N=50, 2500 iterations is fast.
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            idx = N + i * N + j
            w = p[idx]
            if w < 1e-10:
                continue
            dist_ij = dists[i, j]
            if dist_ij < 1e-10:
                continue
            vec = points[i] - points[j]
            # d(g_ij)/d(p_i) = vec / (2 * dist_ij)
            grad_points[i] += w * (vec / (2.0 * dist_ij))
            
    # Objective is -soft_min, so gradient is -grad_points
    grad_flat = -grad_points.ravel()
    
    return obj, grad_flat

def construct():
    deadline = time.time() + DEADLINE
    
    np.random.seed(42)
    
    best_r = 0.0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Hexagonal lattice
    hex_init = get_hexagonal_init(r_approx=0.125)
    inits.append(hex_init)
    
    # 2. Perturbed hex
    hex_pert = hex_init + np.random.normal(0, 0.01, hex_init.shape)
    inits.append(hex_pert)
    
    # 3. Ring structure (from previous knowledge, flattened)
    # Ring 0: 1, Ring 1: 6, Ring 2: 12, Ring 3: 18, Ring 4: 13
    ring_radii = [0.0, 0.25, 0.50, 0.70, 0.87]
    ring_counts = [1, 6, 12, 18, 13]
    ring_angles = [0.0, 0.0, np.pi/12, 0.0, np.pi/26]
    ring_points = []
    idx = 0
    for r_idx in range(5):
        r = ring_radii[r_idx]
        a0 = ring_angles[r_idx]
        n = ring_counts[r_idx]
        for k in range(n):
            a = a0 + 2 * np.pi * k / n
            ring_points.append([r * np.cos(a), r * np.sin(a)])
    inits.append(np.array(ring_points))
    
    # 4. Random inside circle
    random_points = np.random.uniform(-0.8, 0.8, (N, 2))
    # Project to unit circle if needed
    norms = np.linalg.norm(random_points, axis=1)
    mask = norms > 0.8
    random_points[mask] *= 0.8 / norms[mask]
    inits.append(random_points)
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        time_left = deadline - time.time() - 5
        if time_left < 5:
            break
            
        # Run optimization
        start_time = time.time()
        
        def fun_and_grad(x):
            if time.time() - start_time > time_left * 0.8:
                raise TimeoutError()
            obj, grad = soft_min_objective_and_grad(x, beta=50.0)
            if not np.all(np.isfinite(grad)):
                grad = np.zeros_like(x)
            return obj, grad
            
        try:
            result = minimize(fun_and_grad, init_pts.ravel(), method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                      'maxfun': 10000})
            if result.x is not None and np.all(np.isfinite(result.x)):
                opt_pts = result.x.reshape(-1, 2)
                r = compute_radius(opt_pts)
                if r > best_r:
                    best_r = r
                    best_points = opt_pts.copy()
        except Exception:
            # If timeout or error, evaluate the last known good state?
            # L-BFGS-B usually returns the last iterate on exception if we catch it,
            # but it's safer to just skip.
            pass
        
        # If we have a good solution, try to refine with higher beta?
        # Or just continue to next init.
        
    # Fallback
    if best_points is None or best_r < 0.1:
        best_points = inits[0]
        best_r = compute_radius(best_points)
        
    # Ensure validity
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.999:
            best_points[i] *= 0.999 / norms[i]
            
    return best_points

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")

