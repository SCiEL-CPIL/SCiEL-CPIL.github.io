"""
Rewritten program for packing 200 equal circles in a unit square.
Strategy:
1. Initialize with a hexagonal lattice.
2. Minimize a smooth repulsive energy function to spread points uniformly.
3. Refine using local search to maximize the exact minimum distance.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    # Distance to boundaries
    # x boundaries: 0.5 - |x|
    # y boundaries: 0.5 - |y|
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    
    # Minimum pair distance
    # Use pdist for efficiency
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    
    return min(min_pair_dist / 2.0, min_boundary)

def _energy_and_grad(params):
    """
    Compute smooth repulsive energy and its gradient.
    Energy = sum_{i<j} 1/d_ij^2 + sum_i sum_wall 1/d_iw^2
    """
    c = params.reshape(N, 2)
    
    # Calculate pairwise distances squared
    # To avoid division by zero, add small epsilon
    # But we want it to be smooth, so we assume no exact collisions in the optimization path
    
    # Pairwise terms
    # We need to compute sum 1/d^2 and gradient
    # Efficient way:
    # d2[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
    x = c[:, 0]
    y = c[:, 1]
    
    # Vectorized pairwise squared distances
    # dx[i, j] = x[i] - x[j]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    
    # Avoid division by zero for i=j (d2=0)
    # We only sum over i < j
    # Set diagonal to 1 to avoid NaNs, we will mask it later
    np.fill_diagonal(d2, 1.0)
    
    # Inverse squared distances
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # Energy is sum of upper triangle
    E_pair = np.triu(inv_d2, 1).sum()
    
    # Gradient of pairwise terms
    # d/dx_i (1/d_ij^2) = -2 * (x_i - x_j) / d_ij^4
    # grad_x[i] = sum_{j != i} -2 * (x_i - x_j) / d_ij^4
    # Note: d_ij^4 = (d2_ij)^2
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    # Use matrix multiplication style accumulation
    # term = -2 * dx * inv_d2^2
    # But we need to sum over j for each i
    # inv_d4 = inv_d2^2
    inv_d4 = inv_d2**2
    np.fill_diagonal(inv_d4, 0.0)
    
    # grad_x[i] = sum_j (-2 * dx[i,j] * inv_d4[i,j])
    # This is effectively (-2 * dx @ inv_d4).sum(axis=1)? No.
    # (dx * inv_d4).sum(axis=1) gives sum_j dx[i,j] * inv_d4[i,j]
    grad_x -= 2.0 * (dx * inv_d4).sum(axis=1)
    grad_y -= 2.0 * (dy * inv_d4).sum(axis=1)
    
    # Boundary terms
    # E_wall = sum_i [ 1/(0.5-x_i)^2 + 1/(0.5+x_i)^2 + 1/(0.5-y_i)^2 + 1/(0.5+y_i)^2 ]
    # Let's calculate these directly
    dx_w = 0.5 - x
    dx_w_neg = 0.5 + x
    dy_w = 0.5 - y
    dy_w_neg = 0.5 + y
    
    # Add small epsilon to prevent blowup if too close to boundary
    eps = 1e-8
    dx_w = np.clip(dx_w, eps, None)
    dx_w_neg = np.clip(dx_w_neg, eps, None)
    dy_w = np.clip(dy_w, eps, None)
    dy_w_neg = np.clip(dy_w_neg, eps, None)
    
    E_wall = (1.0/dx_w**2 + 1.0/dx_w_neg**2 + 1.0/dy_w**2 + 1.0/dy_w_neg**2).sum()
    
    # Gradients for walls
    # d/dx (1/(0.5-x)^2) = 2/(0.5-x)^3
    # d/dx (1/(0.5+x)^2) = -2/(0.5+x)^3
    grad_x += 2.0 / dx_w**3 - 2.0 / dx_w_neg**3
    grad_y += 2.0 / dy_w**3 - 2.0 / dy_w_neg**3
    
    total_E = E_pair + E_wall
    total_grad = np.concatenate([grad_x, grad_y])
    
    return total_E, total_grad

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    # Approximate spacing for 200 circles in 1x1 square
    # Area per circle ~ 1/200 ~ 0.005. Diameter ~ sqrt(0.005) ~ 0.07
    # Hexagonal packing is denser.
    
    best_c = None
    best_r = 0
    
    # Try a range of spacings
    for d in np.linspace(0.06, 0.08, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            # Ensure strictly inside
            c = np.clip(c, -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _local_polish(c, max_iters=5000, time_limit=10.0):
    """
    Local search using simultaneous relaxation of conflicting pairs and boundary violations.
    Optimized to only process tight pairs.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    step = 1e-3 # Initial step size
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_boundary_dist = np.min(bdists)
        
        current_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        tol = 1e-6
        
        forces = np.zeros((N, 2))
        
        # 1. Boundary forces
        d_left = 0.5 - c[:, 0]
        d_right = c[:, 0] + 0.5
        d_down = 0.5 - c[:, 1]
        d_up = c[:, 1] + 0.5
        
        min_d_vec = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        
        tight_boundary_idx = np.where(min_d_vec < min_boundary_dist + tol)[0]
        
        for i in tight_boundary_idx:
            if d_left[i] == min_d_vec[i]:
                forces[i, 0] += 1.0
            elif d_right[i] == min_d_vec[i]:
                forces[i, 0] -= 1.0
            elif d_down[i] == min_d_vec[i]:
                forces[i, 1] += 1.0
            else:
                forces[i, 1] -= 1.0

        # 2. Pair forces
        tight_pairs = np.where(dist_matrix < (min_pair_dist + tol))
        i_idx = tight_pairs[0]
        j_idx = tight_pairs[1]
        mask = i_idx < j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        
        if len(i_idx) > 0:
            vecs = c[i_idx] - c[j_idx]
            norms = np.linalg.norm(vecs, axis=1)
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            unit_vecs = vecs / safe_norms[:, np.newaxis]
            
            np.add.at(forces, i_idx, unit_vecs)
            np.add.at(forces, j_idx, -unit_vecs)
        
        # Apply forces
        for i in range(N):
            norm_f = np.linalg.norm(forces[i])
            if norm_f > 1e-10:
                new_c_i = c[i] + (forces[i] / norm_f) * step
                if np.all(np.abs(new_c_i) < 0.499):
                    c[i] = new_c_i
        
        new_r = _compute_radius(c)
        if new_r > current_r:
            current_r = new_r
        else:
            step *= 0.9
            if step < 1e-7:
                break

    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Initialize
    c0 = _initialize_hexagonal()
    r0 = _compute_radius(c0)
    best_c = c0.copy()
    best_r = r0
    
    # 2. Global Optimization using smooth energy
    # We use L-BFGS-B with the analytical gradient
    if time.time() < deadline - 30:
        try:
            # Bounds to keep points strictly inside
            bounds = [(-0.499, 0.499)] * (2 * N)
            
            # We might want to try multiple random restarts if time permits, 
            # but the hexagonal start is usually very good.
            # Let's do one long optimization from the hexagonal start.
            
            res = minimize(
                _energy_and_grad,
                c0.flatten(),
                method='L-BFGS-B',
                jac=True, # We provide gradient in _energy_and_grad
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
            r_opt = _compute_radius(c_opt)
            
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
                
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    # 3. Local Polish
    remaining_time = deadline - time.time() - 5
    if remaining_time > 1:
        c_polished, r_polished = _local_polish(best_c, max_iters=10000, time_limit=remaining_time)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
            
    # Final safety check
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
