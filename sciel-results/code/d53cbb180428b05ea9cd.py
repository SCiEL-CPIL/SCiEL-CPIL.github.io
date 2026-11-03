"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_radius(pos):
    """Compute the actual packing radius from positions."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Distance to boundary
    dist_x = 0.5 - np.abs(pos[:, 0])
    dist_y = 0.5 - np.abs(pos[:, 1])
    dist_boundary = np.minimum(dist_x, dist_y)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    # Using broadcasting for pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Mask diagonal
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = float(np.min(dist_sq))
    min_pair = math.sqrt(min_pair_sq)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_hex_lattice(spacing):
    """Generate a fine hexagonal lattice of candidate points in [-0.5, 0.5]^2."""
    points = []
    dy = spacing * math.sqrt(3) / 2.0
    max_coord = 0.5
    n_rows = int(np.ceil(2 * max_coord / dy)) + 2
    n_cols = int(np.ceil(2 * max_coord / spacing)) + 2
    
    for row in range(n_rows):
        y = -max_coord + row * dy
        if y > max_coord:
            break
        x_offset = 0.0 if row % 2 == 0 else spacing / 2.0
        for col in range(n_cols):
            x = -max_coord + col * spacing + x_offset
            if x > max_coord:
                break
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append([x, y])
    
    return np.array(points) if points else np.array([]).reshape(0, 2)

def _greedy_select(candidates, n_select):
    """
    Greedy best-fit: select n_select points from candidates to maximize
    the minimum distance to already-selected points and boundaries.
    """
    if len(candidates) < n_select:
        return candidates.copy()
    
    selected = []
    # Precompute boundary distances for all candidates
    dist_x = 0.5 - np.abs(candidates[:, 0])
    dist_y = 0.5 - np.abs(candidates[:, 1])
    boundary_dist = np.minimum(dist_x, dist_y)
    
    # Track the current "score" for each candidate
    scores = boundary_dist.copy()
    
    for _ in range(n_select):
        # Find the candidate with the highest score
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        selected.append(candidates[best_idx])
        
        # Update scores
        new_point = candidates[best_idx]
        diff = candidates - new_point
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)
        dist[best_idx] = np.inf  # Exclude self
        
        # Update scores: score_i = min(old_score_i, dist_i_to_new)
        scores = np.minimum(scores, dist)
    
    return np.array(selected) if selected else np.array([]).reshape(0, 2)

def _soft_min_and_grad(pos, beta):
    """
    Compute the soft-minimum of all pairwise half-distances and boundary distances,
    and its gradient with respect to positions.
    
    Returns:
        value: The soft-min value (approximation of min radius)
        grad: Gradient of the negative soft-min (for maximization) 
              Actually, we want to maximize r. Let f = -soft_min. Minimize f.
              Or let g = soft_min. Maximize g.
              We will return grad_g such that pos += lr * grad_g improves g.
    """
    n = len(pos)
    # 1. Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Set diagonal to a large number so they don't contribute to min
    np.fill_diagonal(dist_sq, 1e10)
    
    # We care about half-distances for circles
    # Let d_ij = dist(i,j) / 2
    # We want min(d_ij, boundary_dist_i)
    
    # To avoid sqrt for gradient calculation complexity, we can work with squared distances?
    # No, soft-min of sqrt(x) is different. Let's stick to distances.
    # However, computing sqrt of 200x200 matrix is fine.
    dists = np.sqrt(dist_sq)
    half_dists = dists / 2.0
    
    # 2. Boundary distances
    dist_x = 0.5 - np.abs(pos[:, 0])
    dist_y = 0.5 - np.abs(pos[:, 1])
    boundary_dists = np.minimum(dist_x, dist_y)
    
    # Combine all "constraints" into a single array of distances
    # The objective is min( all half_dists, all boundary_dists )
    
    # For gradient, we need to be careful.
    # Let S be the set of all distances.
    # SoftMin(S) = -1/beta * log( sum(exp(-beta * s_k)) )
    # d(SoftMin)/dx = (1/beta) * sum( exp(-beta * s_k) * ds_k/dx ) / sum( exp(-beta * s_k) )
    # This is a weighted average of the gradients of the individual distances.
    
    # To be efficient, we can compute the weights w_k = exp(-beta * s_k) / sum(exp(-beta * s_k))
    # Then grad = sum(w_k * grad_s_k)
    
    # Let's compute the unnormalized weights for pairwise and boundary separately to save memory?
    # No, 200x200 is small. Let's just concatenate.
    
    # Reshape half_dists to a flat vector, excluding diagonal
    mask = ~np.eye(n, dtype=bool)
    flat_half_dists = half_dists[mask]
    
    # Gradients for pairwise half-distances:
    # d/dx_i (dist(i,j)/2) = (x_i - x_j) / (2 * dist(i,j))
    # We need to accumulate these gradients for each point i.
    
    # Let's compute the weights first.
    # Note: boundary_dists are 1D array of size n.
    # flat_half_dists is 1D array of size n*(n-1).
    
    all_dists = np.concatenate([flat_half_dists, boundary_dists])
    
    # Shift to avoid overflow in exp?
    # max_dist = np.max(all_dists)
    # weights = np.exp(-beta * (all_dists - max_dist))
    # This is numerically stable.
    
    max_d = np.max(all_dists)
    # If beta is large, exp(-beta * d) can underflow for d >> min_d. That's good.
    # We want to weight the smallest distances most heavily.
    
    # Calculate weights
    # Using log-sum-exp trick for stability is better, but for gradient we need normalized weights.
    # Let's just compute exp(-beta * all_dists).
    # To prevent underflow issues affecting the relative weights of the *smallest* values,
    # we can subtract the minimum distance.
    min_d = np.min(all_dists)
    shifted_dists = all_dists - min_d
    weights = np.exp(-beta * shifted_dists)
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        # Fallback if all underflow (should not happen if min_d is subtracted)
        weights = np.ones_like(all_dists)
        sum_weights = len(all_dists)
    
    weights = weights / sum_weights
    
    # Now compute the gradient of the soft-min.
    # grad_softmin = sum_k w_k * grad_s_k
    
    grad_pos = np.zeros_like(pos)
    
    # 1. Pairwise part
    # flat_half_dists corresponds to the upper/lower triangle of the dist matrix.
    # It's easier to iterate or use matrix operations.
    # Let's use matrix operations.
    
    # We have weights for pairwise: w_pair (size n*(n-1))
    # We have weights for boundary: w_bound (size n)
    
    w_pair = weights[:len(flat_half_dists)]
    w_bound = weights[len(flat_half_dists):]
    
    # Reshape w_pair back to matrix
    w_mat = np.zeros((n, n))
    w_mat[mask] = w_pair
    
    # We need to compute for each i:
    # sum_j w_ij * (x_i - x_j) / (2 * dist_ij)
    # Note: w_ij is associated with distance d_ij.
    # In our flat array, d_ij and d_ji are both present?
    # Yes, mask includes both (i,j) and (j,i).
    # So w_mat[i,j] is the weight for distance between i and j.
    
    # Let's compute the vector field.
    # diff[i,j] = x_i - x_j
    # unit_vec[i,j] = diff[i,j] / dist_ij
    # grad_pair[i] = sum_j w_mat[i,j] * 0.5 * unit_vec[i,j]
    
    # dists is the full matrix of distances.
    # Avoid division by zero on diagonal (weights are 0 there anyway, but let's be safe)
    safe_dists = np.where(dists > 1e-8, dists, 1.0)
    
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # Weighted sum
    # grad_pair[i] = sum_j w_mat[i,j] * 0.5 * unit_vecs[i,j]
    # w_mat is (n,n), unit_vecs is (n,n,2)
    # We can do: (w_mat[..., np.newaxis] * unit_vecs).sum(axis=1) * 0.5
    
    grad_pair = (w_mat[:, :, np.newaxis] * unit_vecs).sum(axis=1) * 0.5
    
    # 2. Boundary part
    # d/dx_i (0.5 - |x_ix|) = -sign(x_ix) if x_ix != 0
    # d/dy_i (0.5 - |x_iy|) = -sign(x_iy) if x_iy != 0
    
    grad_bound = np.zeros_like(pos)
    grad_bound[:, 0] = -np.sign(pos[:, 0])
    grad_bound[:, 1] = -np.sign(pos[:, 1])
    
    # Handle zeros in sign (gradient is undefined, but 0 is a reasonable choice or random)
    # If pos is exactly 0, sign is 0.
    
    # Weighted sum for boundary
    grad_bound_weighted = w_bound[:, np.newaxis] * grad_bound
    
    # Total gradient
    total_grad = grad_pair + grad_bound_weighted
    
    # The value is the soft-min
    # SoftMin = min_d - 1/beta * log(sum_weights * exp(-beta * min_d)?) 
    # Actually, SoftMin = -1/beta * log( sum(exp(-beta * s_k)) )
    # = -1/beta * log( exp(-beta * min_d) * sum(exp(-beta * (s_k - min_d))) )
    # = min_d + 1/beta * log(sum_weights_unnorm)  ... wait.
    # sum(exp(-beta * s_k)) = exp(-beta * min_d) * sum(exp(-beta * (s_k - min_d)))
    # log(sum) = -beta * min_d + log(sum_exp_shifted)
    # SoftMin = -1/beta * (-beta * min_d + log(sum_exp_shifted))
    # = min_d - 1/beta * log(sum_exp_shifted)
    
    # We computed weights = exp(-beta * shifted) / sum(exp(-beta * shifted))
    # So sum(exp(-beta * shifted)) = 1 / min(weights) ? No.
    # sum_weights_unnorm = sum(exp(-beta * shifted_dists))
    # We have w = exp(...) / sum_unnorm.
    # So sum_unnorm = exp(-beta * shifted_min) / w_min ? No.
    # Let's just compute the value directly from the original formula for stability.
    # value = min_d - (1.0 / beta) * math.log(sum_weights_unnorm)
    # We need sum_weights_unnorm.
    # sum_weights_unnorm = np.sum(np.exp(-beta * shifted_dists))
    
    # Re-calculating sum for accuracy
    sum_unnorm = np.sum(np.exp(-beta * shifted_dists))
    value = min_d - (1.0 / beta) * math.log(sum_unnorm)
    
    return value, total_grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_pos = None
    best_r = 0.0
    
    # 1. Initialization
    # Use a fine hex grid for good initial structure
    spacing = 0.02
    candidates = _generate_hex_lattice(spacing)
    if len(candidates) == 0:
        # Fallback to random
        pos = np.random.uniform(-0.45, 0.45, (N, 2))
    else:
        pos = _greedy_select(candidates, N)
    
    if len(pos) < N:
        # Pad with random if necessary
        missing = N - len(pos)
        rand_pos = np.random.uniform(-0.45, 0.45, (missing, 2))
        pos = np.vstack([pos, rand_pos])
    
    # Ensure valid shape
    pos = pos[:N]
    
    current_r = _compute_radius(pos)
    if current_r > best_r:
        best_r = current_r
        best_pos = pos.copy()
    
    # 2. Gradient Descent with Annealing
    # Start with low beta (smooth) and increase to high beta (sharp)
    # This helps escape local minima and refine the solution.
    
    # Parameters
    lr_start = 0.005
    lr_end = 0.0005
    beta_start = 50.0
    beta_end = 500.0
    n_iter = 800  # Total iterations
    
    # Check time
    time_per_iter_est = 0.05 # Estimated 50ms per iter
    max_iters_allowed = int((deadline - time.time()) / time_per_iter_est)
    n_iter = min(n_iter, max_iters_allowed)
    
    if n_iter > 0:
        for i in range(n_iter):
            if time.time() > deadline - 10:
                break
            
            # Annealing schedule
            t = i / max(1, n_iter - 1)
            # Linear annealing for beta
            beta = beta_start + (beta_end - beta_start) * t
            # Linear annealing for learning rate (decrease over time)
            lr = lr_start + (lr_end - lr_start) * t
            
            # Compute gradient
            try:
                val, grad = _soft_min_and_grad(pos, beta)
            except Exception:
                break
            
            # Update positions
            pos = pos + lr * grad
            
            # Clip to square
            pos = np.clip(pos, -0.4999, 0.4999)
            
            # Check improvement
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
            
            # If we get stuck (no improvement for a while), maybe restart or jitter?
            # For now, rely on annealing.
    
    # 3. Final Polish
    # A few iterations of exact local search to fix any tiny overlaps or boundary issues.
    if time.time() < deadline - 2:
        # Simple coordinate descent
        pos = best_pos.copy()
        current_r = _compute_radius(pos)
        
        # Try to move points to increase the minimum distance
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                # Try moving point i in 4 directions + 4 diagonals
                for dx in [-0.001, 0, 0.001]:
                    for dy in [-0.001, 0, 0.001]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pos = pos.copy()
                        new_pos[i, 0] += dx
                        new_pos[i, 1] += dy
                        new_pos = np.clip(new_pos, -0.4999, 0.4999)
                        
                        new_r = _compute_radius(new_pos)
                        if new_r > current_r + 1e-12:
                            pos = new_pos
                            current_r = new_r
                            improved = True
                            if current_r > best_r:
                                best_r = current_r
                                best_pos = pos.copy()
                            break
                    if improved:
                        break
            if not improved:
                break
    
    # Final safety check
    if best_pos is None:
        # Absolute fallback: grid
        side = int(np.ceil(math.sqrt(N)))
        best_pos = np.zeros((N, 2))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                best_pos[count, 0] = -0.49 + 0.98 * i / (side - 1) if side > 1 else 0
                best_pos[count, 1] = -0.49 + 0.98 * j / (side - 1) if side > 1 else 0
                count += 1
            if count >= N:
                break
    
    return best_pos
# EVOLVE-BLOCK-END
