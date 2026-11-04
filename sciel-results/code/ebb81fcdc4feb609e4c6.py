import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for overhead/return
    
    # 1. Initialization
    # Use a 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    # Add small noise to break symmetry
    np.random.seed(42)
    centers = grid + np.random.uniform(-0.01, 0.01, size=(N, 3))
    # Clip to be strictly inside
    centers = np.clip(centers, -0.49, 0.49)
    
    # 2. Optimization Parameters
    # Soft-min temperature beta
    # Start low, increase to sharpen the minimum
    beta_start = 10.0
    beta_end = 500.0
    steps = 2000
    lr = 0.005
    momentum = 0.9
    
    # Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros_like(centers)
    v = np.zeros_like(centers)
    
    best_centers = centers.copy()
    best_score = -np.inf
    
    # Pre-compute pair indices for speed?
    # Pairs: i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Animate beta
        progress = step / steps
        # Exponential ramp
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Compute pairwise distances
        # diff[i] = centers[i] - centers[j]
        diff = centers[i_idx] - centers[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        dists_safe = np.maximum(dists, 1e-8)
        
        # Compute wall distances
        # For each point i and axis k, d = 0.5 - |c_ik|
        # Shape: (N, 3)
        wall_dists = 0.5 - np.abs(centers)
        
        # Combine all distances for soft-min
        # We need weights for all terms
        # Pairwise terms: exp(-beta * dists)
        # Wall terms: exp(-beta * wall_dists)
        
        # To avoid underflow/overflow, we can shift by max
        all_dists = np.concatenate([dists, wall_dists.flatten()])
        
        # Soft min calculation
        # f = -1/beta * log(sum(exp(-beta * d)))
        # grad f = sum( w_k * grad d_k ) where w_k = exp(-beta * d_k) / sum(exp(-beta * d))
        
        # For numerical stability, subtract max
        max_d = np.max(all_dists)
        # Actually, we want to maximize the min, so we are minimizing -soft_min.
        # The gradient of the soft_min is what we need.
        # Let's compute the weights.
        
        # exp terms
        exp_pair = np.exp(-beta * (dists - max_d))
        exp_wall = np.exp(-beta * (wall_dists - max_d).flatten())
        
        sum_exp = np.sum(exp_pair) + np.sum(exp_wall)
        
        w_pair = exp_pair / sum_exp
        w_wall = exp_wall / sum_exp
        
        # Gradient of soft_min with respect to centers
        # grad_soft_min = sum_k w_k * grad d_k
        
        grad = np.zeros_like(centers)
        
        # Pairwise contributions
        # grad d_ij wrt c_i = (c_i - c_j) / d_ij
        # grad d_ij wrt c_j = (c_j - c_i) / d_ij
        
        # w_pair is shape (N_pairs,)
        # diff is shape (N_pairs, 3)
        # unit_diff = diff / dists_safe[:, None]
        unit_diff = diff / dists_safe[:, None]
        
        # Contribution to c_i: sum w * unit_diff
        # Contribution to c_j: sum w * (-unit_diff)
        
        # Use np.add.at for accumulation
        np.add.at(grad, i_idx, w_pair[:, None] * unit_diff)
        np.add.at(grad, j_idx, -w_pair[:, None] * unit_diff)
        
        # Wall contributions
        # d_ik = 0.5 - |c_ik|
        # grad d_ik wrt c_ik = -sign(c_ik)
        
        # w_wall is shape (N*3,)
        # centers is shape (N, 3)
        signs = np.sign(centers).flatten()
        # If center is 0, sign is 0, which is fine (gradient 0)
        
        grad_wall = -w_wall * signs
        grad += grad_wall.reshape(N, 3)
        
        # We want to MAXIMIZE soft_min, so we want to move in direction of grad.
        # Adam step for maximization
        # Standard Adam minimizes. So we use -grad for the update direction if we use standard Adam.
        # Or just update centers += lr * ...
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        
        # Update: centers = centers + lr * m_hat / (sqrt(v_hat) + eps)
        centers = centers + lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Clip to cube to ensure validity
        centers = np.clip(centers, -0.5, 0.5)
        
        # Evaluate exact score (hard min)
        # This is expensive, so do it every 10 steps
        if step % 10 == 0:
            # Compute min pairwise distance
            min_pair = np.min(dists)
            # Compute min wall distance
            min_wall = np.min(wall_dists)
            r = min(min_pair, min_wall) / 2.0
            
            if r > best_score:
                best_score = r
                best_centers = centers.copy()
                
    # Final polish: ensure no overlaps and inside
    # If best_score is -inf, use current
    if best_score == -np.inf:
        best_score = 0
        best_centers = centers
        
    # One final check and cleanup
    # Compute radius
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = best_centers[i_idx] - best_centers[j_idx]
    dists = np.linalg.norm(diff, axis=1)
    min_pair = np.min(dists)
    min_wall = np.min(0.5 - np.abs(best_centers))
    r = min(min_pair, min_wall) / 2.0
    
    # If r is very small or invalid, return the grid?
    # But the optimization should have improved it.
    
    return best_centers
# EVOLVE-BLOCK-END
