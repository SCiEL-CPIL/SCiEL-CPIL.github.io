"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper to compute the true radius (for final validation and tracking)
    def compute_true_radius(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    # 1. Initial Placement: Jittered Grid
    # We want to spread spheres out. A 5x5x4 grid is 100 points.
    # Let's use a 5x5x4 grid to fill the cube reasonably.
    nx, ny, nz = 5, 5, 4
    if nx * ny * nz < N:
        # Fallback to cube root if N changes, but N is fixed 100
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        nx = ny = nz = m
        
    # Generate grid points
    x_coords = (np.arange(nx) + 0.5) / nx - 0.5
    y_coords = (np.arange(ny) + 0.5) / ny - 0.5
    z_coords = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
    
    # If we have more points than N, take the first N. 
    # If fewer, we'd need to add, but 5*5*4=100 is exact.
    centers = grid[:N].copy()
    
    # Add small random jitter to break symmetries
    np.random.seed(42) # Deterministic for reproducibility in this run
    centers += np.random.normal(0, 0.01, size=(N, 3))
    
    # Clip to bounds to ensure validity at start
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = compute_true_radius(best_centers)
    
    # 2. Gradient Ascent on Smooth Minimum Surrogate
    # Objective: Maximize r
    # Surrogate: r_smooth = -logsumexp(-dists / beta) * beta
    # Where dists are the set of constraints (pairwise/2 and wall distances).
    # Gradient of r_smooth w.r.t. positions can be computed analytically.
    
    # To speed up, we only compute the gradient of the "active" constraints?
    # No, smooth min uses all constraints but weights them by softmax.
    # This is O(N^2) per step. With N=100, N^2=10,000.
    # 10,000 * 3 (coords) = 30,000 mults per step.
    # We can do ~1000 steps in 140s? 
    # Let's estimate: Python loop overhead is high. 
    # We should vectorize the gradient calculation as much as possible.
    
    # Let's implement a step function that computes the gradient for all points.
    
    def smooth_min_gradient(c, beta):
        # c: (N, 3)
        # Returns: grad (N, 3) such that c += lr * grad increases smooth_min
        
        # 1. Compute all pairwise distances (N, N)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Constraints are dists / 2. Let d_ij = |c_i - c_j| / 2.
        # Gradient of d_ij w.r.t c_i is (c_i - c_j) / |c_i - c_j| = (c_i - c_j) / (2 * d_ij)
        # Wait, d_ij = 0.5 * |c_i - c_j|. 
        # d(d_ij)/d(c_i) = 0.5 * (c_i - c_j) / |c_i - c_j|
        
        # 2. Compute wall distances
        # w_ik = 0.5 - |c_ik|
        # Gradient w.r.t c_ik:
        # if c_ik > 0, d/dx (0.5 - x) = -1
        # if c_ik < 0, d/dx (0.5 - (-x)) = 1
        # So grad_w = -sign(c)
        
        # Collect all constraint values into a single array?
        # Size: N*(N-1)/2 + 3*N. ~ 4950 + 300 = 5250 constraints.
        # This is manageable.
        
        # However, building this array and its gradient every step might be slow in Python if not careful.
        # Let's try to compute the gradient directly using vectorized operations.
        
        # Let f_k be the constraint values.
        # r_smooth = -beta * log( sum_k exp(-f_k / beta) )
        # dr_smooth / df_k = exp(-f_k / beta) / sum_j exp(-f_j / beta)
        # Let p_k = exp(-f_k / beta) / Z. Then dr_smooth / df_k = p_k.
        
        # We need sum_k (dr_smooth/df_k) * (df_k/dc)
        
        # Let's compute the pair part first.
        # f_ij = |c_i - c_j| / 2
        # grad_c_i f_ij = 0.5 * (c_i - c_j) / |c_i - c_j|
        
        # We can compute the weights w_ij = exp(-f_ij / beta) / Z_total
        # But Z_total includes wall terms too.
        
        # For simplicity and speed, let's split the gradient into pair and wall parts,
        # but we need the global normalization constant Z.
        
        # Step 2a: Compute all f values
        f_pairs = dists / 2.0
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        f_pairs_vals = f_pairs[mask] # Size N*(N-1)/2? No, N*(N-1) because dists is N x N.
        # Actually dists is N x N. We need unique pairs.
        # Let's just use the upper triangle.
        iu = np.triu_indices(N, k=1)
        f_pair_unique = f_pairs[iu[0], iu[1]]
        
        f_walls = 0.5 - np.abs(c) # (N, 3)
        f_wall_flat = f_walls.flatten()
        
        all_f = np.concatenate([f_pair_unique, f_wall_flat])
        
        # Compute softmax weights
        # To avoid overflow, subtract max
        max_f = np.max(all_f)
        exp_f = np.exp(-(all_f - max_f) / beta)
        Z = np.sum(exp_f)
        p = exp_f / Z
        
        # Split p back
        n_pairs = len(f_pair_unique)
        p_pairs = p[:n_pairs]
        p_walls = p[n_pairs:]
        
        # Reconstruct p_pairs into (N, N) matrix (symmetric)
        P_pairs = np.zeros((N, N))
        P_pairs[iu[0], iu[1]] = p_pairs
        P_pairs[iu[1], iu[0]] = p_pairs
        
        # Gradient of pair part:
        # grad_c_i = sum_j (P_pairs[i, j] * 0.5 * (c_i - c_j) / |c_i - c_j|)
        # Note: dists[i, j] is |c_i - c_j|.
        # We need to handle division by zero (diagonal is inf, so weight is 0, but let's be safe).
        
        # Vectorized calculation:
        # diff is (N, N, 3)
        # dists is (N, N)
        # We want for each i: sum_j P[i, j] * 0.5 * diff[i, j] / dists[i, j]
        
        # diff / dists[:, :, np.newaxis] gives unit vectors.
        # Where dists is inf (diagonal), this is 0/inf = 0? No, 0/inf is 0 in numpy? 
        # Actually diff[i,i] is 0. 0/inf is 0.
        # But we have P[i,i]=0. So it's fine.
        
        unit_vecs = diff / dists[:, :, np.newaxis]
        
        # grad_pairs = 0.5 * sum_j P[i, j] * unit_vecs[i, j]
        # This is a matrix multiplication?
        # grad_pairs[i] = 0.5 * (P[i, :] dot unit_vecs[i, :])
        
        # P is (N, N), unit_vecs is (N, N, 3).
        # We can do: np.einsum('ij,ijk->ik', P_pairs, unit_vecs)
        
        grad_pairs = 0.5 * np.einsum('ij,ijk->ik', P_pairs, unit_vecs)
        
        # Gradient of wall part:
        # f_w_ik = 0.5 - |c_ik|
        # grad f_w_ik w.r.t c_ik = -sign(c_ik)
        # grad_walls = sum_k p_walls[i, k] * (-sign(c_ik))
        
        signs = np.sign(c)
        # p_walls is (N, 3) reshaped? 
        # p_walls is flat. Reshape to (N, 3).
        P_walls = p_walls.reshape(N, 3)
        
        grad_walls = P_walls * (-signs)
        
        total_grad = grad_pairs + grad_walls
        
        return total_grad

    # Optimization Loop
    beta = 0.01  # Temperature for smooth min
    lr = 0.005   # Learning rate
    
    # We will run this for a fixed number of steps or until time limit.
    # Let's try 2000 steps.
    
    for step in range(2000):
        if time.time() > deadline - 10:
            break
            
        grad = smooth_min_gradient(centers, beta)
        
        # Update centers
        centers += lr * grad
        
        # Clip to bounds to maintain validity
        # If we clip, the gradient at the boundary is effectively zeroed out for that component,
        # which is consistent with the constraint being active.
        centers = np.clip(centers, -0.5, 0.5)
        
        # Occasionally re-evaluate true radius to track best
        # Doing it every step is expensive (O(N^2) norm).
        # But smooth_min_gradient already computes dists. 
        # We can modify it to return the true radius as well? 
        # Or just compute it every 10 steps.
        
        if step % 10 == 0:
            current_radius = compute_true_radius(centers)
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                
        # Anneal beta? 
        # Lowering beta makes the smooth min sharper, focusing on the minimum constraint.
        # This helps in the final stages.
        if step > 1000:
            beta *= 0.99
            lr *= 0.99
            
    # Final check
    final_radius = compute_true_radius(best_centers)
    if final_radius > best_radius:
        best_radius = final_radius
        
    return best_centers
# EVOLVE-BLOCK-END
