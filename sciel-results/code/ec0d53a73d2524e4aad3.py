"""
Improved packing: Fast Lloyd's Algorithm + Local Neighborhood Polish.

Strategy:
1. Initialization: Jittered grid and random clusters.
2. Structural Optimization: Lloyd's Algorithm (Voronoi centroids) to equalize clearance.
3. Local Polish: Direct neighborhood search to maximize the minimum distance (bottleneck).
"""
import numpy as np
import time
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import minimize
import sys

N = 50

def _score(centers):
    """
    Compute the maximum common radius for given centers.
    Uses vectorized operations for speed.
    """
    centers = np.asarray(centers, dtype=np.float64)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _soft_min_obj_and_grad(x_flat, beta):
    """
    Objective function for L-BFGS-B: minimize -soft_min(radius).
    x_flat: flattened centers (N*3)
    beta: softness parameter (higher = closer to exact min)
    """
    centers = x_flat.reshape(N, 3)
    
    # Wall constraints: 0.5 - |c_ik|
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c.max(axis=1)
    
    # Pair constraints: 0.5 * ||c_i - c_j||
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = 0.5 * dists
    
    # Combine all constraints into a single vector of "clearances"
    # We want to maximize the minimum of these.
    # Let z_k be the clearances. We want max min(z).
    # Equivalent to min -min(z).
    # Soft min: -log(sum(exp(-beta * z))) / beta
    # To avoid overflow, we use the log-sum-exp trick.
    
    # Wall clearances
    z_wall = wall_dists
    # Pair clearances (only i<j to avoid duplication, but for gradient it's easier to sum all pairs and divide? 
    # No, min is over all pairs. We can just flatten the upper triangle.)
    i_idx, j_idx = np.triu_indices(N, k=1)
    z_pair = 0.5 * dists[i_idx, j_idx]
    
    z = np.concatenate([z_wall, z_pair])
    
    # Soft min calculation
    # m = min(z)
    # soft_min = -1/beta * log(sum(exp(-beta * (z - m)))) - m
    # This is numerically stable.
    m = np.min(z)
    # If m is very negative, exp might overflow? No, z-m >= 0, so exp(-beta*(z-m)) <= 1.
    # Actually, if z is large positive, -beta*(z-m) is large negative, exp is 0.
    # If z is close to m, exp is 1.
    
    # Gradient of soft_min w.r.t z:
    # d/dz_k [ -1/beta log(sum_j exp(-beta(z_j - m))) ]
    # Note: m also depends on z, but in the log-sum-exp trick, if we define S = sum exp(-beta z_j), 
    # soft_min = -1/beta log S. 
    # d/dz_k (-1/beta log S) = -1/beta * (1/S) * dS/dz_k = -1/beta * (1/S) * (-beta exp(-beta z_k)) = exp(-beta z_k) / S
    # So grad_z = exp(-beta z) / sum(exp(-beta z))
    
    # However, we must be careful with the shift m for stability.
    # Let w_k = exp(-beta * (z_k - m)). Then S' = sum w_k.
    # grad_z_k = w_k / S'
    
    w = np.exp(-beta * (z - m))
    S_prime = np.sum(w)
    grad_z = w / S_prime
    
    obj_val = -(m + np.log(S_prime) / beta)
    
    # Now chain rule to get gradient w.r.t centers
    # grad_centers = sum_k (grad_z_k * d z_k / d centers)
    
    grad_centers = np.zeros_like(centers)
    
    # 1. Wall terms
    # z_wall_i = 0.5 - max_j |c_ij|
    # d z_wall_i / d c_ik = -sign(c_ik) if |c_ik| is the max, else 0.
    # If multiple axes are max, we can average or just pick one. Averaging is smoother.
    max_abs_val = abs_c.max(axis=1)
    # Mask where |c_ik| == max_abs_val
    is_max = (abs_c == max_abs_val[:, np.newaxis])
    # Weight the gradient by is_max / count_max
    count_max = is_max.sum(axis=1, keepdims=True)
    count_max[count_max == 0] = 1.0
    
    # grad_wall_part[i, k] = grad_z_wall_i * (-sign(c_ik) * is_max[i, k] / count_max[i])
    # Note: z_wall is at index i in z.
    # The first N elements of z are wall_dists.
    grad_wall_contrib = grad_z[:N][:, np.newaxis] * (-np.sign(centers) * is_max / count_max)
    grad_centers += grad_wall_contrib
    
    # 2. Pair terms
    # z_pair_p = 0.5 * ||c_i - c_j||
    # d z_pair_p / d c_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d z_pair_p / d c_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    # We need to map grad_z[N:] to pairs.
    # i_idx, j_idx correspond to the pairs in z_pair.
    # grad_pair_part[i] += sum_{j>i} grad_z_pair_ij * 0.5 * (c_i - c_j) / d_ij
    # grad_pair_part[j] += sum_{i<j} grad_z_pair_ij * 0.5 * (c_j - c_i) / d_ij
    
    # Vectorized update
    # dists[i, j] is the distance.
    # We have grad_z_pair = grad_z[N:]
    # We need to accumulate into grad_centers.
    
    # Create a matrix of gradients for pairs
    # G[i, j] = grad_z_pair[k] * 0.5 * (c_i - c_j) / d_ij  (if i != j)
    # Actually, we can just loop over the pairs for N=50, it's fast enough (1225 pairs).
    
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        d = dists[i, j]
        if d < 1e-10:
            continue
        g = grad_z[N + k]
        vec = 0.5 * (centers[i] - centers[j]) / d
        grad_centers[i] += g * vec
        grad_centers[j] -= g * vec
        
    return obj_val, grad_centers.flatten()

def _smooth_polish(centers, time_limit=10.0):
    """
    Uses L-BFGS-B to minimize the negative soft-min radius.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Annealing schedule for beta
    # Start with low beta (smooth) and increase (sharp)
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for beta in betas:
        if time.time() > start_time + time_limit - 2:
            break
            
        def objective(x):
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val
        
        def gradient(x):
            _, grad = _soft_min_obj_and_grad(x, beta)
            return grad
            
        try:
            res = minimize(
                objective, 
                centers.flatten(), 
                jac=gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            new_centers = res.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            new_score = _score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                centers = new_centers.copy()
        except Exception:
            pass
            
    return best_centers, best_score

def _lloyd_iteration(centers, max_iter=50, deadline=None):
    """
    Lloyd's algorithm: iteratively move each center to the centroid of its
    Voronoi cell (clipped to the cube).
    """
    c = centers.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        try:
            vor = Voronoi(c)
        except Exception:
            break
            
        new_c = c.copy()
        
        for i in range(len(c)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0:
                continue
                
            vertex_indices = [v for v in region if v >= 0]
            if len(vertex_indices) < 3:
                continue
                
            vertices = vor.vertices[vertex_indices]
            
            # Clip vertices to cube for centroid calculation
            clipped_verts = np.clip(vertices, -0.5, 0.5)
            
            centroid = np.mean(clipped_verts, axis=0)
            
            # Damping
            new_c[i] = 0.5 * c[i] + 0.5 * centroid
            
        new_c = np.clip(new_c, -0.499, 0.499)
        
        displacement = np.max(np.linalg.norm(new_c - c, axis=1))
        c = new_c
        
        if displacement < 1e-9:
            break
            
    return c

def _nelder_mead_polish(centers, time_limit=5.0):
    """
    Final polish using Nelder-Mead on the exact objective.
    """
    start_time = time.time()
    
    def exact_obj(x):
        c = x.reshape(N, 3)
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        return -_score(c)
        
    try:
        res = minimize(
            exact_obj, 
            centers.flatten(), 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        new_centers = res.x.reshape(N, 3)
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        if time.time() > start_time + time_limit:
            return centers, _score(centers)
            
        return new_centers, _score(new_centers)
    except Exception:
        return centers, _score(centers)

def _initial_grid_jittered(seed=42):
    """Jittered grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    jitter = rng.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _initial_random_cluster(seed=123):
    """Random initialization with some clustering to encourage Voronoi structure."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.4, 0.4, size=(N, 3))
    return centers

def construct():
    t_start = time.time()
    deadline = t_start + 145  # 5s safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Initializations
    inits = [
        _initial_grid_jittered(seed=42),
        _initial_grid_jittered(seed=123),
        _initial_random_cluster(seed=101),
        _initial_random_cluster(seed=202)
    ]
    
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        cur = init.copy()
        cur_score = _score(cur)
        
        if cur_score > best_score:
            best_score = cur_score
            best_centers = cur.copy()
            
        # 1. Lloyd's Algorithm (Structural equalization)
        if time.time() > deadline - 20:
            continue
            
        lloyd_res = _lloyd_iteration(cur, max_iter=50, deadline=deadline - 15)
        lloyd_score = _score(lloyd_res)
        
        if lloyd_score > best_score:
            best_score = lloyd_score
            best_centers = lloyd_res.copy()
            cur = lloyd_res.copy()
        else:
            cur = best_centers.copy()
            
        # 2. Smooth Polish (L-BFGS-B on soft-min)
        if time.time() > deadline - 10:
            continue
            
        time_left_smooth = deadline - 5 - time.time()
        if time_left_smooth > 5:
            smooth_res, smooth_score = _smooth_polish(cur, time_limit=time_left_smooth)
            
            if smooth_score > best_score:
                best_score = smooth_score
                best_centers = smooth_res.copy()
                cur = smooth_res.copy()
                
        # 3. Nelder-Mead Polish (Exact objective)
        if time.time() > deadline - 2:
            continue
            
        nm_res, nm_score = _nelder_mead_polish(cur, time_limit=deadline - 1 - time.time())
        
        if nm_score > best_score:
            best_score = nm_score
            best_centers = nm_res.copy()
    
    # Final Safety Check
    if best_centers is None:
        best_centers = _initial_grid_jittered()
        
    # Ensure final result is valid (inside cube)
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers

if __name__ == "__main__":
    # Quick self-test
    centers = construct()
    r = _score(centers)
    print(f"Score: {r}")
