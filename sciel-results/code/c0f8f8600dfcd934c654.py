import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_radius(centers):
    """
    Compute the exact radius for the given centers.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    return r

def smooth_min(values, temperature):
    """
    Differentiable approximation of min using log-sum-exp (negative).
    min(x) = -logsumexp(-x)
    """
    if temperature <= 0:
        return np.min(values)
    # For numerical stability
    a = -values / temperature
    a_max = np.max(a)
    sum_exp = np.sum(np.exp(a - a_max))
    return -temperature * (np.log(sum_exp) + a_max)

def objective_smooth(centers_flat, temp):
    """
    Objective: -smooth_min(all distances)
    """
    centers = centers_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    b_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, b_dists])
    
    # We want to maximize the minimum distance, so minimize -min(dist)
    # Using smooth min
    val = -smooth_min(all_dists, temp)
    return val

def gradient_smooth(centers_flat, temp):
    """
    Gradient of -smooth_min(all distances)
    """
    centers = centers_flat.reshape(N, 2)
    grad = np.zeros_like(centers)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Boundary
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    b_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    all_dists = np.concatenate([pair_dists, b_dists])
    
    # Weights for smooth min:
    # d(-logsumexp(-x/T))/dx = (1/T) * exp(-x/T) / sum(exp(-x/T))
    # Let a = -x/T. weights = exp(a - max(a)) / sum(exp(a - max(a)))
    a = -all_dists / temp
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    sum_exp = np.sum(exp_a)
    weights = exp_a / sum_exp
    
    # Gradient of -smooth_min is sum(weights * grad(dist_k))
    # Note: smooth_min(x) = -logsumexp(-x/T).
    # d/dx_k (-smooth_min(x)) = d/dx_k (logsumexp(-x/T)) = (1/T) * exp(-x_k/T) / sum(...)
    # So grad(-smooth_min) = sum_k [ (1/T) * w_k * grad_k(dist_k) ]
    # Wait, w_k defined above is exp(a_k)/sum.
    # grad = (1/T) * sum_k w_k * grad(dist_k)
    
    coeff = weights / temp
    
    # Pairwise gradients
    # dist_ij = ||c_i - c_j||
    # grad_{c_i} dist_ij = (c_i - c_j) / dist_ij
    # grad_{c_j} dist_ij = (c_j - c_i) / dist_ij
    
    # coeff[:N*(N-1)/2] corresponds to pair_dists
    pair_coeff = coeff[:len(pair_dists)]
    
    # We need to accumulate gradients for each point.
    # This is O(N^2) to loop through pairs. N=50 is small.
    for idx in range(len(pair_dists)):
        i, j = iu[0][idx], iu[1][idx]
        w = pair_coeff[idx]
        if w < 1e-12: continue
        
        d = dist[i, j]
        v = (centers[i] - centers[j]) / d
        
        grad[i] += w * v
        grad[j] -= w * v
        
    # Boundary gradients
    b_coeff = coeff[len(pair_dists):]
    
    # d_left: x_i + 0.5. grad = (1, 0)
    grad[:, 0] += b_coeff[:N]
    # d_right: 0.5 - x_i. grad = (-1, 0)
    grad[:, 0] -= b_coeff[N:2*N]
    # d_bottom: y_i + 0.5. grad = (0, 1)
    grad[:, 1] += b_coeff[2*N:3*N]
    # d_top: 0.5 - y_i. grad = (0, -1)
    grad[:, 1] -= b_coeff[3*N:4*N]
    
    return grad.flatten()

def hexagonal_init(n):
    """
    Generate a hexagonal grid initialization.
    """
    # Approximate number of rows/cols
    # Density of hex packing is higher.
    # Let's just generate a large hex grid and pick the first N closest to center.
    # Or simpler: create a grid and shear it.
    
    # Standard square grid
    m = int(np.ceil(np.sqrt(n)))
    x = (np.arange(m) + 0.5) / m - 0.5
    y = (np.arange(m) + 0.5) / m - 0.5
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Apply hexagonal shear
    # For every row, shift x by half the spacing if row is odd.
    # Spacing is 1/m.
    # We need to identify rows.
    # This is messy with ravel. Let's do it structurally.
    
    pts = []
    # Hex packing:
    # Row 0: x = -0.5 + 0.5/m + k * (1/m)
    # Row 1: x = -0.5 + 1/m + k * (1/m)  (shifted by 0.5/m)
    # y spacing is slightly less than 1/m to maximize density? 
    # For unit square, we just want a good starting spread.
    
    # Let's use a simpler approach: generate points in a hex lattice covering the square.
    # Lattice vectors: a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
    # Scale to fit.
    
    # Actually, a simple jittered grid with hex-like offsets is often sufficient.
    # Let's stick to a structured grid but with hex offsets.
    
    rows = int(np.ceil(np.sqrt(n * 2 / 3))) # Approx
    cols = int(np.ceil(np.sqrt(n * 3 / 2))) # Approx
    
    # Just generate a grid of size ceil(sqrt(N)) x ceil(sqrt(N)) and hexify it.
    m = int(np.ceil(np.sqrt(n)))
    centers = []
    for r in range(m):
        for c in range(m):
            if len(centers) >= n:
                break
            # Base position
            px = (c + 0.5) / m - 0.5
            py = (r + 0.5) / m - 0.5
            # Hex offset: shift odd rows by 0.5/m in x
            if r % 2 == 1:
                px += 0.5 / m
            
            # Clip to square? The task says centers must be in [-0.5, 0.5].
            # If we shift, some might go out.
            # Let's just use the standard grid for init, it's safe.
            # The "hex" inspiration is about the *packing* structure.
            # A square grid is a valid init. The optimization will move them.
            centers.append([px, py])
            
    return np.array(centers[:n])

def _refine(centers, start_time, deadline):
    """
    Perform smooth refinement with annealing of temperature.
    """
    if time.time() > deadline - 5.0:
        return centers

    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Start with high temperature (smooth) and anneal to low (sharp)
    temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    current_centers = centers
    current_radius = compute_radius(current_centers)
    
    for temp in temps:
        if time.time() > deadline - 5.0:
            break
            
        try:
            # L-BFGS-B on the smooth objective
            res = minimize(
                lambda x: objective_smooth(x, temp),
                current_centers.flatten(),
                jac=lambda x: gradient_smooth(x, temp),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if np.all(np.isfinite(new_centers)):
                r_new = compute_radius(new_centers)
                if r_new > current_radius:
                    current_radius = r_new
                    current_centers = new_centers
        except Exception:
            pass
            
    return current_centers

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial hex-like grid (actually just a standard grid, but we'll use hex logic if easy)
    # For now, standard grid is robust.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    best_centers = initial_centers.copy()
    best_radius = compute_radius(initial_centers)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Refine initial grid
    c = _refine(initial_centers, start_time, deadline)
    r = compute_radius(c)
    if r > best_radius:
        best_radius = r
        best_centers = c.copy()
        
    # Phase 2: Simulated Annealing / Random Restarts
    # We perturb the best solution and refine.
    # This helps escape local minima of the smooth landscape.
    
    max_restarts = 20
    current_temp_sa = 0.1 # Scale of perturbations
    
    for i in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Perturb best centers
        noise = rng.normal(0, current_temp_sa, size=best_centers.shape)
        c = best_centers + noise
        # Clamp to bounds to ensure validity
        c = np.clip(c, -0.49, 0.49)
        
        # Refine
        c = _refine(c, start_time, deadline)
        r = compute_radius(c)
        
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()
            # If improved, we can keep temp or reduce it? 
            # Standard SA: if improved, reduce temp.
            current_temp_sa *= 0.9
        else:
            # If not improved, accept with some probability? 
            # Or just restart from a different random point?
            # Let's try a random restart every 5th iteration if stuck.
            if i % 5 == 4:
                c = rng.uniform(-0.4, 0.4, size=(N, 2))
                c = _refine(c, start_time, deadline)
                r = compute_radius(c)
                if r > best_radius:
                    best_radius = r
                    best_centers = c.copy()
            
            # Decay perturbation size to focus search
            current_temp_sa *= 0.8
            
    return best_centers
