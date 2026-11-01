"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START

def _score(c):
    """Compute the largest common radius for centres c (shape (N, 3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # Wall distances
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist) / 2.0
    
    return min(min_wall, min_dist)

def _softmin_val_grad(c, beta):
    """
    Compute value and gradient of the soft-minimum of wall and pair distances.
    Objective: maximize soft_min = -1/beta * log( sum exp(-beta * d) )
    where d is the set of all constraints (wall and pair/2).
    
    Returns:
        val: float, the soft-min value
        grad: array (N, 3), gradient pointing in direction of increasing val
    """
    # 1. Wall constraints
    # d_wall[i, k] = 0.5 - |c[i, k]|
    # We need exp(-beta * d_wall)
    # Note: c is clipped to [-0.5, 0.5] in the main loop, so 0.5 - |c| >= 0
    
    # To avoid overflow/underflow in exp, we can compute relative to a max, 
    # but for simplicity and given beta is moderate, direct exp might work if we scale.
    # However, exp(-beta * 0) = 1. If d is small, exp is large.
    # Let's compute the log-sum-exp properly.
    
    # Wall terms
    abs_c = np.abs(c)
    d_wall = 0.5 - abs_c  # (N, 3)
    
    # Pair terms
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    d_pair = 0.5 * np.sqrt(dist_sq)  # (N, N)
    
    # Combine all distances into a flat array for log-sum-exp
    # We don't want to allocate a huge array if possible, but N=100 is small.
    # d_wall has 300 elements, d_pair has 9900 elements.
    d_flat = np.concatenate([d_wall.ravel(), d_pair.ravel()])
    
    # LogSumExp: LSE(x) = log(sum(exp(x)))
    # We want to maximize -1/beta * LSE(-beta * d)
    # Let y = -beta * d
    y = -beta * d_flat
    
    # Stable LogSumExp
    max_y = np.max(y)
    lse = max_y + np.log(np.sum(np.exp(y - max_y)))
    
    val = -lse / beta
    
    # Gradient:
    # val = -1/beta * log( sum exp(-beta * d) )
    # d(val)/dc = -1/beta * ( 1 / sum(exp(-beta*d)) ) * sum( exp(-beta*d_j) * d(-beta*d_j)/dc )
    # d(val)/dc = -1/beta * ( 1 / S ) * sum( exp(-beta*d_j) * (-beta) * d(d_j)/dc )
    # d(val)/dc = sum( exp(-beta*d_j) * d(d_j)/dc ) / S
    # where S = sum(exp(-beta*d_j))
    
    # This is a weighted average of the gradients of the individual distances.
    # weights w_j = exp(-beta * d_j) / S
    
    weights = np.exp(y - max_y)  # Unnormalized weights
    S = np.sum(weights)
    weights /= S  # Normalized weights, sum to 1
    
    # Now compute gradient as sum( w_j * grad(d_j) )
    grad = np.zeros_like(c)
    
    # 1. Wall part
    # d(d_wall[i,k])/dc[i,k] = -sign(c[i,k])
    # We need to map weights back to (N, 3)
    w_wall = weights[:N*3].reshape(N, 3)
    
    # grad_wall[i,k] = w_wall[i,k] * (-sign(c[i,k]))
    # But we want to INCREASE the distance, so we move in direction of -grad(d)/dc?
    # Wait. val is the soft MIN. We want to maximize val.
    # d(val)/dc = sum w_j * d(d_j)/dc.
    # So we should move in direction of +grad.
    # d(d_wall)/dc = -sign(c). So grad = w * (-sign(c)).
    # If c > 0, sign=1, grad = -w. Move in -x direction (away from +wall). Correct.
    # If c < 0, sign=-1, grad = +w. Move in +x direction (away from -wall). Correct.
    
    signs = np.sign(c)
    # Handle 0
    signs[signs == 0] = 1.0
    grad += w_wall * (-signs)
    
    # 2. Pair part
    # d(d_pair[i,j])/dc[i] = (c[i] - c[j]) / |c[i] - c[j]|
    # d(d_pair[i,j])/dc[j] = (c[j] - c[i]) / |c[i] - c[j]| = -d(d_pair[i,j])/dc[i]
    
    w_pair = weights[N*3:].reshape(N, N)
    
    # We can vectorize this.
    # grad[i] += sum_j w_pair[i,j] * (c[i] - c[j]) / dist[i,j]
    # grad[j] += sum_i w_pair[i,j] * (c[j] - c[i]) / dist[i,j]
    
    # dist is already computed as d_pair * 2? No, d_pair = 0.5 * dist.
    # So dist = 2 * d_pair.
    # We need to be careful with division by zero. Diagonal is inf, so weight 0.
    
    # Let's compute the contribution for each i.
    # For each i, sum over j != i of w_pair[i,j] * (c[i]-c[j]) / (2*d_pair[i,j])
    # Note: (c[i]-c[j]) / |c[i]-c[j]| is the unit vector.
    
    # diff is (N, N, 3).
    # unit_diff = diff / dist[..., np.newaxis]
    # But dist is 0 on diagonal. We set d_pair diag to inf, so w_pair diag is 0.
    # So we can safely divide if we handle the 0/0 case.
    
    # dist = 2 * d_pair
    # dist_diag was set to inf in d_pair, so it is inf here.
    # diff diag is 0.
    # 0 / inf = 0. This is fine in numpy? 
    # Actually, we set dist_sq diag to inf, so dist diag is inf.
    # diff diag is 0.
    # unit_diff = diff / dist[..., None]
    # 0 / inf = 0.
    
    dist = 2.0 * d_pair
    # Replace 0 with 1 to avoid warnings, though result is 0 anyway due to weight 0.
    # Actually, if dist is inf, 0/inf is 0.
    # If dist is 0 (which shouldn't happen for i!=j unless coincident), 0/0 is nan.
    # But if coincident, d_pair is 0, weight is huge, but this is a singularity.
    # In SA, coincident points are rare and resolved quickly.
    # To be safe, we can use np.where(dist > 1e-10, diff/dist, 0)
    
    # Let's compute unit vectors
    # dist shape (N, N)
    # diff shape (N, N, 3)
    
    # To avoid memory blowup, we can do this in a loop or vectorized carefully.
    # N=100, so (100, 100, 3) is small.
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_diff = diff / dist[..., np.newaxis]
        
    # Set diagonal to 0 explicitly to be safe
    np.fill_diagonal(unit_diff[:, :, 0], 0)
    np.fill_diagonal(unit_diff[:, :, 1], 0)
    np.fill_diagonal(unit_diff[:, :, 2], 0)
    
    # grad[i] = sum_j w_pair[i,j] * unit_diff[i,j]
    # This is a matrix multiplication: (N, N) @ (N, N, 3) -> (N, 3)
    grad += np.einsum('ij,ijk->ik', w_pair, unit_diff)
    
    # The gradient for j is handled by the symmetry?
    # No, the gradient of the scalar objective with respect to c[j] involves terms where j is the second index.
    # Our derivation:
    # grad_c = sum_k w_k * grad(d_k)
    # For pair (i, j), d_k = d_pair[i, j].
    # grad_c[i] += w_pair[i, j] * (c[i]-c[j])/dist
    # grad_c[j] += w_pair[i, j] * (c[j]-c[i])/dist
    #
    # In the einsum 'ij,ijk->ik', we computed sum_j w_pair[i,j] * unit_diff[i,j].
    # This covers the contribution to c[i] from all pairs (i, j).
    # What about the contribution to c[j] from pair (i, j)?
    # That term is w_pair[i, j] * (c[j]-c[i])/dist = - w_pair[i, j] * unit_diff[i, j].
    # This term contributes to grad[j].
    # In the sum over all pairs, the contribution to grad[j] is sum_i w_pair[i, j] * (-unit_diff[i, j]).
    # Note that w_pair is symmetric? No, d_pair[i,j] = d_pair[j,i], so w_pair[i,j] = w_pair[j,i].
    # And unit_diff[j, i] = -unit_diff[i, j].
    # So sum_i w_pair[i, j] * unit_diff[j, i] = sum_i w_pair[j, i] * (-unit_diff[i, j]) = - sum_i w_pair[i, j] * unit_diff[i, j].
    #
    # So if we compute G = w_pair @ unit_diff (where G[i] = sum_j w_pair[i,j] unit_diff[i,j]),
    # then the full gradient is G + G^T? No.
    # grad[i] = sum_j w_pair[i,j] unit_diff[i,j]  (This is G[i])
    # The term for c[j] from pair (i,j) is -w_pair[i,j] unit_diff[i,j].
    # Summing over i for a fixed j: - sum_i w_pair[i,j] unit_diff[i,j].
    # This is exactly -G[j] if we consider the symmetry?
    # Let's check: G[j] = sum_i w_pair[j,i] unit_diff[j,i] = sum_i w_pair[i,j] (-unit_diff[i,j]) = - sum_i w_pair[i,j] unit_diff[i,j].
    # So the contribution to grad[j] from the "second index" role is - (sum_i w_pair[i,j] unit_diff[i,j]).
    # But wait, the "first index" role for j is G[j] = sum_i w_pair[j,i] unit_diff[j,i].
    # Since w_pair is symmetric and unit_diff is antisymmetric, G[j] = - (sum_i w_pair[i,j] unit_diff[i,j]).
    # So the total gradient for c[j] is G[j] (from being i) + (-G[j]) (from being j)?
    # No.
    # Let's restart the gradient sum.
    # Objective sum over all pairs (i,j) with i<j? Or all i!=j?
    # In my d_flat, I included ALL i!=j pairs (upper and lower triangle).
    # d_pair is symmetric.
    # So the sum is over all i!=j.
    # Term for pair (i,j) in the sum is w_pair[i,j] * d(d_pair[i,j]).
    # d(d_pair[i,j])/dc[i] = u_ij
    # d(d_pair[i,j])/dc[j] = -u_ij
    #
    # Total grad[i] = sum_{j!=i} w_pair[i,j] u_ij + sum_{k!=i} w_pair[k,i] u_ki
    # Note u_ki = -u_ik. And w_pair[k,i] = w_pair[i,k].
    # So second term = sum_{k!=i} w_pair[i,k] (-u_ik) = - sum_{k!=i} w_pair[i,k] u_ik.
    # So Total grad[i] = sum_{j!=i} w_pair[i,j] u_ij - sum_{k!=i} w_pair[i,k] u_ik = 0?
    # This implies the pair gradient is zero? That can't be right.
    #
    # Ah, the objective is sum over ALL pairs in the log-sum-exp.
    # If I sum over all i,j != i, I am double counting the pairs if I think of them as unordered.
    # But the log-sum-exp is over the set of values. The set of values for d_pair includes d_12 and d_21 which are equal.
    # So the weight for d_12 is w, and for d_21 is w.
    # The gradient contribution from d_12 to c1 is w * u_12.
    # The gradient contribution from d_21 to c1 is w * u_21 = w * (-u_12).
    # Sum = 0.
    #
    # This means if I include both (i,j) and (j,i) in the sum, the gradients cancel out for the pair terms!
    # This is a critical error in the formulation if I want to move spheres apart.
    #
    # Solution: Only sum over i < j (unique pairs).
    # Let's redo d_flat to only include i < j.
    
    # Recalculate d_flat and weights for unique pairs
    # This requires changing the structure.
    
    # Let's return and fix the function to use unique pairs.
    return val, grad

def _softmin_val_grad_unique(c, beta):
    """
    Compute value and gradient of the soft-minimum of wall and UNIQUE pair distances.
    """
    # 1. Wall constraints (N*3)
    abs_c = np.abs(c)
    d_wall = 0.5 - abs_c  # (N, 3)
    
    # 2. Pair constraints (N*(N-1)/2)
    # We need to extract upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    # d_pair_vals = 0.5 * norm(c[i_idx] - c[j_idx])
    diff = c[i_idx] - c[j_idx]  # (M, 3) where M = N*(N-1)/2
    dist = np.sqrt(np.sum(diff**2, axis=1))
    d_pair = 0.5 * dist
    
    # Combine
    d_flat = np.concatenate([d_wall.ravel(), d_pair])
    y = -beta * d_flat
    
    max_y = np.max(y)
    lse = max_y + np.log(np.sum(np.exp(y - max_y)))
    val = -lse / beta
    
    weights = np.exp(y - max_y)
    S = np.sum(weights)
    weights /= S
    
    grad = np.zeros_like(c)
    
    # 1. Wall part
    w_wall = weights[:N*3].reshape(N, 3)
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    grad += w_wall * (-signs)
    
    # 2. Pair part
    w_pair = weights[N*3:]  # (M,)
    
    # grad[i_idx[k]] += w_pair[k] * (c[i] - c[j]) / dist
    # grad[j_idx[k]] += w_pair[k] * (c[j] - c[i]) / dist
    
    # unit_vec = diff / dist
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vec = diff / dist[:, np.newaxis]
    # Handle dist=0
    unit_vec[dist < 1e-10] = 0
    
    # Accumulate gradients
    # Use np.add.at for unbuffered accumulation
    np.add.at(grad, i_idx, w_pair[:, np.newaxis] * unit_vec)
    np.add.at(grad, j_idx, -w_pair[:, np.newaxis] * unit_vec)
    
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initialization
    # Start with a grid to have a decent starting point
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    # Add small random noise to break symmetry
    c += np.random.normal(0, 0.01, size=c.shape)
    c = np.clip(c, -0.499, 0.499)
    
    best_c = c.copy()
    best_score = _score(best_c)
    
    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 0.0001
    beta_start = 50.0
    beta_end = 500.0
    
    num_steps = 2000  # Adjust based on time
    
    # We will iterate, updating T and beta
    # We need to estimate how long one step takes.
    # _softmin_val_grad_unique takes ~5ms?
    # 2000 steps * 5ms = 10s. Plenty of time.
    # Let's do more steps or slower cooling.
    
    # Adaptive step count
    # Let's run until deadline - 20s
    step_count = 0
    last_check = start_time
    
    while time.time() < deadline - 20.0:
        step_count += 1
        progress = step_count / 2000.0  # Normalize for first 2000 steps, then keep going
        if progress > 1.0:
            progress = 1.0
            
        # Cooling schedules
        T = T_start * (T_end / T_start) ** progress
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute current val and grad
        val, grad = _softmin_val_grad_unique(c, beta)
        
        # Propose new position
        # Step size should be related to T and gradient magnitude
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-10:
            grad_norm = grad / g_norm
        else:
            grad_norm = np.zeros_like(c)
            
        # Step size: larger when T is high
        step_size = 0.005 * (T / T_start) ** 0.5 + 0.001
        # Add random noise
        noise = np.random.normal(0, step_size * 0.5, size=c.shape)
        
        c_new = c + step_size * grad_norm + noise
        c_new = np.clip(c_new, -0.499, 0.499)
        
        # Compute new val
        val_new, _ = _softmin_val_grad_unique(c_new, beta)
        
        # Acceptance criterion
        delta_val = val_new - val
        if delta_val > 0:
            accept = True
        else:
            # Accept with probability exp(delta_val / T)
            # Note: val is the soft-min (negative or small positive).
            # If val_new < val, delta_val is negative.
            # We want to maximize val.
            # Standard SA minimizes energy. Here we maximize val.
            # So "energy" = -val.
            # delta_energy = -delta_val.
            # Accept if exp(-delta_energy / T) > random => exp(delta_val / T) > random.
            # If delta_val is negative, exp is < 1.
            
            # To avoid underflow, if delta_val / T < -700, prob is 0.
            if delta_val / T > -700:
                prob = np.exp(delta_val / T)
            else:
                prob = 0.0
                
            accept = np.random.rand() < prob
            
        if accept:
            c = c_new
            if val_new > best_score:
                # Check true score periodically or if val improves significantly
                # val is a lower bound on true score?
                # Soft-min is <= true min.
                # So if val_new > best_score, true score might be even higher.
                # But we should compute true score to update best_score correctly.
                # Computing true score is expensive (O(N^2) sqrt).
                # Let's do it every 10 steps or if val_new is significantly better.
                if step_count % 10 == 0 or val_new > best_score + 0.001:
                    true_score = _score(c)
                    if true_score > best_score:
                        best_score = true_score
                        best_c = c.copy()
        else:
            # Reject, keep c
            pass
            
        # Periodic check of true score for best_c
        if step_count % 50 == 0:
            true_score = _score(best_c)
            if true_score > best_score:
                best_score = true_score
            # Also check current c
            true_score_c = _score(c)
            if true_score_c > best_score:
                best_score = true_score_c
                best_c = c.copy()
                
    # Final Polish
    # Use high beta to sharpen the minimum
    if time.time() < deadline - 5.0:
        c = best_c.copy()
        beta = 1000.0
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            val, grad = _softmin_val_grad_unique(c, beta)
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-12:
                break
            grad_norm = grad / g_norm
            # Small steps
            for lr in [0.01, 0.005, 0.001, 0.0005]:
                c_new = c + lr * grad_norm
                c_new = np.clip(c_new, -0.499, 0.499)
                val_new, _ = _softmin_val_grad_unique(c_new, beta)
                if val_new > val:
                    c = c_new
                    val = val_new
                    break
                    
        final_score = _score(c)
        if final_score > best_score:
            best_score = final_score
            best_c = c.copy()
            
    return best_c

# EVOLVE-BLOCK-END
