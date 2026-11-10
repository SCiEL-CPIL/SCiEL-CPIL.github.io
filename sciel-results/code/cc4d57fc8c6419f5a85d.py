"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization: Start with a jittered grid to break symmetries
    # A simple cubic grid has many equivalent local minima. Jittering helps.
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random jitter to break symmetry and avoid degenerate flat regions
    # Jitter should be small relative to grid spacing to maintain validity
    grid_spacing = 1.0 / m
    centers += np.random.normal(0, 0.1 * grid_spacing, size=(N, 3))
    
    # Ensure initial validity (clip to bounds, though jitter is small)
    centers = np.clip(centers, -0.5, 0.5)
    
    # Helper to compute the exact radius for the final result
    def compute_exact_radius(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    best_centers = centers.copy()
    best_radius = compute_exact_radius(best_centers)
    
    # 2. Optimization: Gradient Ascent with Soft-Min Surrogate
    
    # The objective is to maximize R = min( min_{i<j} d_ij/2, min_i w_i )
    # We use a soft-min approximation: R_soft = -1/k * log(sum(exp(-k * x_i)))
    # where x_i are the candidate radii (pair distances/2 and wall distances).
    # As k increases, R_soft approaches the true min.
    # We anneal k from a small value (smooth, global view) to a large value (sharp, local refinement).
    
    # Parameters
    k_start = 10.0
    k_end = 1000.0
    num_iter = 2000  # Fixed number of iterations, time-checked
    lr_start = 0.05
    lr_end = 0.001
    
    # Precompute indices for efficiency? 
    # We need pairwise distances and wall distances.
    
    for it in range(num_iter):
        if time.time() > deadline:
            break
            
        # Annealing schedule
        t = it / num_iter
        # Exponential decay for k (smooth -> sharp)
        k = k_start * (k_end / k_start) ** t
        # Linear decay for learning rate
        lr = lr_start + (lr_end - lr_start) * t
        
        # Compute distances
        # c: (N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        
        # We want to maximize the minimum of:
        # 1. d_ij / 2 for all i < j
        # 2. w_ik for all i, k
        # Let's flatten these into a single vector of "constraints" X
        # However, building a huge vector every step is slow.
        # Alternative: Compute gradients directly.
        
        # Gradient of soft-min:
        # Let f(x) = -1/k * log(sum(exp(-k*x)))
        # df/dx_i = (1/sum(exp(-k*x))) * exp(-k*x_i) = softmax(-k*x)_i
        # We want to maximize f, so we move in direction of +grad f.
        
        # The "x" values are:
        # x_pairs = dists / 2 (upper triangle)
        # x_walls = wall_dists (flattened)
        
        # Instead of flattening, we can accumulate gradients.
        # grad_centers[i] = sum_{j!=i} (softmax_weight_ij * d(d_ij/2)/d(c_i)) 
        #                  + sum_{k} (softmax_weight_ik * d(w_ik)/d(c_i))
        
        # d(d_ij/2)/d(c_i) = (c_i - c_j) / d_ij
        # d(w_ik)/d(c_i) = -sign(c_ik)
        
        # Let's compute the weights.
        # To avoid memory blowup and slow exp, we can work with the full matrix.
        
        # Pairwise part:
        # We need softmax over the upper triangle of (dists/2).
        # Let M = dists / 2.
        # Mask lower triangle and diagonal.
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        M_masked = np.where(mask, M, -np.inf)
        
        # Compute softmax weights for pairs
        # To ensure numerical stability, subtract max
        M_max = np.max(M_masked)
        exp_M = np.exp(-k * (M_masked - M_max))
        sum_exp_M = np.sum(exp_M)
        if sum_exp_M == 0:
            w_pairs = np.zeros((N, N))
        else:
            w_pairs = exp_M / sum_exp_M
            # w_pairs[i,j] is the weight for constraint |c_i - c_j|/2
            # Note: This weight represents the sensitivity of the soft-min to that specific pair.
            
        # Wall part:
        # W = wall_dists (N, 3)
        W_max = np.max(W)
        exp_W = np.exp(-k * (W - W_max))
        sum_exp_W = np.sum(exp_W)
        if sum_exp_W == 0:
            w_walls = np.zeros((N, 3))
        else:
            w_walls = exp_W / sum_exp_W
            # w_walls[i, k] is the weight for constraint (0.5 - |c_ik|)
            
        # Now compute gradients
        # grad_c = 0
        # For each pair (i, j) with i < j:
        # Contribution to grad_c[i] += w_pairs[i, j] * (c_i - c_j) / d_ij
        # Contribution to grad_c[j] += w_pairs[j, i] * (c_j - c_i) / d_ij  (symmetry)
        
        # Vectorized gradient computation for pairs:
        # grad_pair[i] = sum_j w_pairs[i, j] * (c_i - c_j) / d_ij
        # We have diff = c_i - c_j.
        # term = w_pairs * (diff / dists)  # (N, N, 3)
        # We need to sum over j for each i.
        
        # Handle division by zero (diagonal is inf, so 1/inf = 0, but dists has 0 on diag. 
        # We masked with -inf for exp, but dists still has 0. 
        # Let's replace 0 in dists with 1 for division, mask handles the rest.
        dists_safe = np.where(dists == 0, 1.0, dists)
        
        # term[i, j] = w_pairs[i, j] * (c_i - c_j) / d_ij
        # Note: w_pairs is symmetric? No, we computed softmax on upper triangle.
        # Actually, the soft-min is over the set of all distances. The set is symmetric.
        # So the weight for (i,j) should be the same as (j,i) if we consider the set.
        # My mask was upper triangle. 
        # Let's make w_pairs symmetric to be safe and simple.
        # If I use full matrix softmax, it's easier.
        
        # Let's restart the weight calculation with full matrix symmetry.
        # X_pairs = dists / 2
        # Mask diagonal.
        X_pairs = M
        X_pairs[~mask] = -np.inf  # Keep only upper triangle? 
        # Actually, the objective is min over ALL i<j. 
        # If I use full matrix, I double count. 
        # Let's stick to upper triangle but make the gradient accumulation correct.
        
        # grad_c[i] += sum_{j>i} w_ij * (c_i - c_j)/d_ij  - sum_{j<i} w_ji * (c_j - c_i)/d_ji
        # Since w_ij = w_ji (if we symmetrize) and (c_j - c_i)/d_ji = -(c_i - c_j)/d_ij
        # grad_c[i] = sum_{j != i} w_ij * (c_i - c_j)/d_ij
        
        # So we need w_ij to be the weight of the constraint.
        # If I compute softmax on the upper triangle, w_ij is defined for i<j.
        # I can create a symmetric weight matrix W_sym where W_sym[i,j] = w_pairs[i,j] if i<j else w_pairs[j,i].
        
        W_sym = np.zeros((N, N))
        W_sym[np.triu_indices(N, 1)] = w_pairs[np.triu_indices(N, 1)]
        W_sym[np.tril_indices(N, -1)] = W_sym.T[np.tril_indices(N, -1)]
        # Wait, simpler:
        W_sym = w_pairs + w_pairs.T
        # But w_pairs has 0 on lower triangle. So W_sym[i,j] = w_pairs[i,j] (if i<j) + w_pairs[j,i] (which is 0 if j>i, i.e., i<j).
        # So for i<j, W_sym[i,j] = w_pairs[i,j].
        # For i>j, W_sym[i,j] = w_pairs[j,i].
        # This is correct.
        
        # Now compute grad from pairs:
        # G_pair[i] = sum_j W_sym[i,j] * (c_i - c_j) / d_ij
        # term[i,j,k] = W_sym[i,j] * diff[i,j,k] / dists[i,j]
        
        # dists is (N,N). diff is (N,N,3).
        # We need to divide diff by dists (broadcasting along last axis).
        unit_vecs = diff / dists_safe[:, :, np.newaxis]
        # Multiply by weights
        weighted_vecs = W_sym[:, :, np.newaxis] * unit_vecs
        # Sum over j (axis 1)
        grad_pairs = np.sum(weighted_vecs, axis=1)
        
        # Wall part:
        # grad_c[i,k] += w_walls[i,k] * d(0.5 - |c_ik|)/d(c_ik)
        # d(0.5 - |x|)/dx = -sign(x)
        # So grad_walls[i,k] = w_walls[i,k] * (-sign(c_ik))
        
        grad_walls = w_walls * (-np.sign(centers))
        
        # Total gradient
        grad = grad_pairs + grad_walls
        
        # Update
        # We are maximizing, so move in +grad direction.
        new_centers = centers + lr * grad
        
        # Handle bounds:
        # If a center goes out of bounds, the wall distance becomes negative, 
        # which would lower the score. 
        # The gradient from walls should push it back in.
        # However, if it goes far out, the sign might flip or magnitude might be weird.
        # It's safer to clip or reject. 
        # Given the "soft" nature, clipping is a practical necessity for validity.
        # But clipping can cause sticking to walls.
        # Let's clip to [-0.5, 0.5].
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        # Compute exact radius for this new configuration
        new_radius = compute_exact_radius(new_centers)
        
        # Accept if better
        if new_radius > best_radius:
            best_radius = new_radius
            best_centers = new_centers.copy()
            centers = new_centers.copy()
        else:
            # If not better, we might still want to move to escape local minima?
            # Gradient ascent on the soft-min is monotonic for the soft objective, 
            # but not necessarily for the hard min.
            # If the hard min decreases, we revert to the previous best centers 
            # to ensure we don't degrade.
            # However, sticking to "best" only is hill-climbing.
            # To escape, we rely on the changing k (annealing).
            # As k increases, the soft-min becomes sharper, focusing on the true min.
            # If we get stuck, we might need a perturbation.
            # For now, let's just keep the best valid state.
            # If we don't update centers, the gradient will be the same next step?
            # No, k changes, so the weights change, so the gradient direction changes.
            # So even if we revert centers, the next gradient step will be different.
            
            # Actually, if we revert centers to the *previous* step's centers (not best), 
            # we might oscillate. 
            # Let's keep `centers` as the current working state.
            # If the move is bad, we revert `centers` to the state before the move.
            # But we didn't save the old state. 
            # Let's just not update `centers` if the move is bad?
            # No, that means we retry the same gradient? No, k changes.
            # So if we don't update centers, we are at the old position, but with a new k.
            # This is fine.
            
            pass # centers remains unchanged (old position)
            
    # Final check
    return best_centers
# EVOLVE-BLOCK-END
