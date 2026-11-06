import numpy as np
import time

N = 50

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    X = grid[:N].copy()
    
    # Optimize using gradient ascent on soft-min distance
    # Objective: Maximize min(distances)
    # Approximation: Maximize - (1/beta) * log(sum(exp(-beta * d)))
    
    beta = 50.0  # Sharpness of min approximation
    alpha = 0.01  # Initial step size
    
    # Precompute indices for pairs
    # We need to handle all pairs (i,j) i<j and all wall distances
    
    def compute_grad(X, beta):
        # X is (N, 2)
        # Calculate all pairwise distances
        # d_ij = ||x_i - x_j||
        # d_i_wall = min(0.5 - |x_ix|, 0.5 - |x_iy|)
        
        # Vectorized pairwise distances
        # dist_matrix[i,j] = ||x_i - x_j||
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf) # Ignore self
        
        # Wall distances
        # For each point, distance to closest wall
        # d_wall_i = min(0.5 - abs(x_i), 0.5 - abs(y_i))
        # We need the gradient of this min.
        # Let's just compute the soft-min over all constraints.
        
        # Constraints:
        # 1. Pairwise: d_ij for i < j
        # 2. Walls: d_ix, d_iy (distance to left/right, top/bottom)
        
        # It is easier to just iterate if N is small, but vectorization is better.
        # Let's construct a flat array of all distances.
        
        # Pairwise indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Wall distances
        # d_left = 0.5 - X[:,0]
        # d_right = 0.5 + X[:,0]
        # d_bottom = 0.5 - X[:,1]
        # d_top = 0.5 + X[:,1]
        wall_dists = np.concatenate([
            0.5 - X[:, 0], 0.5 + X[:, 0],
            0.5 - X[:, 1], 0.5 + X[:, 1]
        ])
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # Gradient w.r.t X
        # d M / d X = -1/beta * 1/S * sum (-beta * exp(-beta * d_k) * d d_k / d X)
        #            = 1/S * sum (exp(-beta * d_k) * d d_k / d X)
        
        weights = np.exp(-beta * all_dists)
        S = np.sum(weights)
        
        grad = np.zeros_like(X)
        
        # Pairwise contribution
        # d d_ij / d x_i = (x_i - x_j) / d_ij
        # d d_ij / d x_j = (x_j - x_i) / d_ij
        # Note: if d_ij is 0, gradient is undefined, but weights will be high.
        # Add small epsilon to distances to avoid division by zero.
        
        pair_dists_safe = np.maximum(pair_dists, 1e-8)
        w_pairs = weights[:len(pair_dists)]
        
        # For each pair (i,j), contribution to grad[i] is w/S * (x_i - x_j)/d
        # Contribution to grad[j] is w/S * (x_j - x_i)/d
        
        # We can use np.add.at for accumulation
        # But vectorized loop over pairs might be slow if N is large, but N=50 is small (1225 pairs)
        
        # Let's do it in a loop for clarity and correctness, as N is small
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            w = w_pairs[k] / S
            d = pair_dists_safe[k]
            diff_vec = X[i] - X[j]
            grad_vec = (w / d) * diff_vec
            grad[i] += grad_vec
            grad[j] -= grad_vec
            
        # Wall contribution
        # d (0.5 - x_i) / d x_i = -1
        # d (0.5 + x_i) / d x_i = 1
        # d (0.5 - y_i) / d y_i = -1
        # d (0.5 + y_i) / d y_i = 1
        
        w_walls = weights[len(pair_dists):]
        w_wall_S = w_walls / S
        
        # Indices in w_walls:
        # 0..N-1: left (x)
        # N..2N-1: right (x)
        # 2N..3N-1: bottom (y)
        # 3N..4N-1: top (y)
        
        grad[:, 0] += -w_wall_S[:N]       # Left wall
        grad[:, 0] += w_wall_S[N:2*N]     # Right wall
        grad[:, 1] += -w_wall_S[2*N:3*N]  # Bottom wall
        grad[:, 1] += w_wall_S[3*N:4*N]   # Top wall
        
        return grad

    best_X = X.copy()
    best_score = -np.inf
    
    # Run optimization
    for it in range(2000):
        if time.time() > deadline:
            break
            
        grad = compute_grad(X, beta)
        X = X + alpha * grad
        
        # Clamp to square
        X = np.clip(X, -0.5, 0.5)
        
        # Calculate true score (min distance)
        # This is expensive, so do it periodically
        if it % 10 == 0:
            # Calculate min distance
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            
            min_wall = np.min(np.minimum(0.5 - np.abs(X[:,0]), 0.5 - np.abs(X[:,1])))
            
            score = min(min_pair, min_wall)
            
            if score > best_score:
                best_score = score
                best_X = X.copy()
                
        # Decay step size
        alpha *= 0.99
        
        # Increase beta to sharpen min
        beta *= 1.01
        
    return best_X

# EVOLVE-BLOCK-END
