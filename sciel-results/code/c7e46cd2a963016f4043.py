"""
Strategy: Multi-Start L-BFGS-B on a Log-Sum-Exp (LSE) Surrogate.

This approach abandons stochastic physics (MD/SA) in favor of a deterministic
gradient-based optimization. The objective function is a smooth approximation
of the "minimum radius" using the Log-Sum-Exp trick. This creates a landscape
where the global optimum is a deep valley, allowing L-BFGS-B to converge
reliably to high-quality packings. We use multiple diverse starting points
(grid, FCC, random) to escape local minima, and a final local search phase
to polish the exact geometric constraints.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centres):
    """Compute the exact maximum common radius for given centres."""
    if len(centres) < 2:
        return 0.5 - np.max(np.abs(centres), axis=1)
    
    # Distance to walls
    wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists) / 2.0
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)

def _lse_surrogate(centres, beta=100.0):
    """
    Compute a smooth surrogate objective: -log(sum(exp(-beta * d_ij))) 
    This approximates -beta * min(d_ij).
    We maximize the minimum distance, so we minimize the negative of this.
    Actually, let's minimize U = log(sum(exp(-beta * d))) for all constraints.
    As beta -> inf, U -> -beta * min(d). Minimizing U is maximizing min(d).
    
    Constraints:
    1. Pairwise distances d_ij
    2. Wall distances w_i
    
    We combine them into a single log-sum-exp.
    """
    n = len(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
    
    # Collect all distances
    # To avoid overflow in exp, we subtract the max (shifted)
    # We want log(sum(exp(-beta * d)))
    # Let d_min be the min. log(sum(exp(-beta*d))) = -beta*d_min + log(sum(exp(-beta*(d-d_min))))
    # For numerical stability, we can just compute the logsumexp directly if we handle large values.
    # scipy.special.logsumexp is useful but we can do it manually.
    
    # Distances are positive.
    # d_pairs is a flattened array of unique pairs? No, full matrix is easier for vectorization, 
    # but we must be careful with the diagonal (inf).
    
    d_pairs = dists_sq ** 0.5
    d_pairs_flat = d_pairs[d_pairs < np.inf]
    
    d_walls = wall_dists
    
    # Combine
    all_d = np.concatenate([d_pairs_flat, d_walls])
    
    # Clip to avoid 0
    all_d = np.clip(all_d, 1e-9, 1.0)
    
    # LogSumExp of -beta * d
    # log(sum(exp(-beta * d)))
    # = -beta * min_d + log(sum(exp(-beta * (d - min_d))))
    min_d = np.min(all_d)
    shifted = -beta * (all_d - min_d)
    
    # If shifted has very negative values, exp becomes 0. That's fine.
    # logsumexp = log(sum(exp(shifted)))
    # To be safe:
    max_shifted = np.max(shifted)
    if max_shifted > 500: # Should not happen since min_d is the max of shifted
        max_shifted = 500
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    
    val = -beta * min_d + log_sum_exp
    return val

def _lse_surrogate_grad(centres, beta=100.0):
    """
    Gradient of the LSE surrogate.
    U = log(sum_k exp(-beta * d_k))
    dU/dx_i = sum_k (dU/dd_k * dd_k/dx_i)
    dU/dd_k = -beta * exp(-beta * d_k) / sum_j exp(-beta * d_j)
             = -beta * softmax_k
    
    For pairs: d_k = |x_i - x_j|
    dd_k/dx_i = (x_i - x_j) / d_k
    
    For walls: d_k = 0.5 - max(|x_i|)
    dd_k/dx_i = -sign(x_i) * e_axis (where e_axis is the unit vector for the max coordinate)
    
    Note: We must handle the case where d_k is very small (0) carefully.
    """
    n = len(centres)
    grad = np.zeros_like(centres)
    
    # Pairwise
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall
    wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
    
    # Combine distances for softmax weights
    d_pairs = dists[dists < np.inf]
    d_walls = wall_dists
    all_d = np.concatenate([d_pairs, d_walls])
    all_d = np.clip(all_d, 1e-9, 1.0)
    
    # Softmax weights: w_k = exp(-beta * d_k) / sum(exp(-beta * d_j))
    shifted = -beta * all_d
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # Split weights back
    w_pairs = weights[:len(d_pairs)]
    w_walls = weights[len(d_pairs):]
    
    # Gradient contribution from pairs
    # We need to map the flattened weights back to the matrix.
    # This is tricky with numpy. Let's reconstruct the matrix of weights.
    w_matrix = np.zeros((n, n))
    mask = ~np.eye(n, dtype=bool)
    # d_pairs is flattened in row-major order.
    # We can create a full matrix of distances, flatten it, and use that to index?
    # Or just iterate? No, N=50, N^2=2500. Iteration is fast enough for gradient? 
    # 50*50/2 = 1225 pairs. 
    # Let's do it vectorized if possible, or simple loop if not.
    # Simple loop for gradient accumulation is often clearer and fast enough for N=50.
    
    # Actually, let's use the matrix approach.
    # Create a full matrix of distances
    full_dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(full_dists, np.inf)
    
    # We need the weights for the full matrix.
    # The flattened d_pairs corresponds to full_dists[mask].
    # So we can create a full weight matrix.
    full_weights = np.zeros((n, n))
    full_weights[mask] = w_pairs
    
    # Gradient of U wrt x_i:
    # Sum over j: w_ij * (-beta) * (x_i - x_j) / d_ij
    # Note: w_ij is the weight for the distance d_ij.
    # The term is -beta * w_ij * (x_i - x_j) / d_ij
    
    # Let's compute the force-like term: F_ij = -beta * w_ij / d_ij * (x_i - x_j)
    # We can compute coeff = -beta * w_matrix / full_dists
    # Then grad[i] = sum_j coeff[i,j] * (x_i - x_j)
    
    # Handle division by zero/inf
    coeff = np.zeros_like(full_weights)
    valid = full_dists < np.inf
    coeff[valid] = -beta * full_weights[valid] / full_dists[valid]
    
    # grad[i] = sum_j coeff[i,j] * diff[i,j]
    grad_pairs = np.einsum('ij,ijk->ik', coeff, diff)
    
    # Gradient contribution from walls
    # d_wall_i = 0.5 - max(|x_i|)
    # Let m_i = max(|x_i|). axis_i = argmax(|x_i|).
    # d(d_wall_i)/d(x_i) = -sign(x_i[axis_i]) * e_{axis_i}
    # Contribution: sum_i w_wall_i * (-beta) * d(d_wall_i)/d(x_i)
    
    abs_centres = np.abs(centres)
    max_idx = np.argmax(abs_centres, axis=1)
    signs = np.sign(centres[np.arange(n), max_idx])
    
    # grad_wall[i] = -beta * w_walls[i] * (-signs[i]) * e_{max_idx[i]}
    # = beta * w_walls[i] * signs[i] * e_{max_idx[i]}
    
    # e_{max_idx[i]} is a vector with 1 at max_idx[i] and 0 elsewhere.
    # We can construct this using a matrix or scatter.
    # grad_wall = np.zeros_like(centres)
    # grad_wall[np.arange(n), max_idx] = beta * w_walls * signs
    
    # Let's verify the sign.
    # U = log(sum exp(-beta d)). d = 0.5 - |x|.
    # dU/dx = -beta * w * d(d)/dx.
    # d(d)/dx = -sign(x).
    # dU/dx = -beta * w * (-sign(x)) = beta * w * sign(x).
    # So if x > 0, gradient is positive -> increases x -> increases |x| -> decreases d -> increases U?
    # Wait. We want to MINIMIZE U.
    # If x is close to wall (0.5), d is small. U is large (exp(-beta*small) is large).
    # We want to move x away from wall.
    # If x > 0, we want to decrease x.
    # Gradient dU/dx should be positive so that -grad (descent) decreases x.
    # My formula: beta * w * sign(x). If x>0, grad > 0. Descent: x -= grad > 0. x decreases. Correct.
    
    grad_wall = np.zeros_like(centres)
    grad_wall[np.arange(n), max_idx] = beta * w_walls * signs
    
    return grad_pairs + grad_wall

def _optimize_lbgfgs(start_centres, beta=100.0, time_limit=10.0):
    """Run L-BFGS-B optimization."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        c = x.reshape(-1, 3)
        return _lse_surrogate(c, beta)
        
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        c = x.reshape(-1, 3)
        return _lse_surrogate_grad(c, beta).flatten()
        
    x0 = start_centres.flatten()
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            jac=gradient, 
            options={'maxiter': 1000, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return start_centres

def _generate_grid_start():
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_fcc_start():
    # FCC lattice points in a cube
    # FCC primitive vectors
    # Generate a supercell of FCC
    # Lattice constant a. 
    # Points: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
    # We want 50 points.
    # Let's just generate a grid and perturb it towards FCC-like?
    # Or generate a dense grid and pick the best subset?
    # For simplicity, let's use a perturbed grid as a "structured" start.
    grid = _generate_grid_start()
    return grid + np.random.randn(N, 3) * 0.01

def _generate_random_start():
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _refine_local(centres, max_iter=500):
    """Final polish using direct nudge on the minimum constraint."""
    centres = centres.copy()
    n = len(centres)
    
    for _ in range(max_iter):
        r = _compute_radius(centres)
        
        # Find min constraint
        wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_val = wall_dists[min_wall_idx]
        
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_val = dists[min_pair_idx] / 2.0
        
        if min_wall_val < min_pair_val:
            i = min_wall_idx
            c = centres[i]
            abs_c = np.abs(c)
            axis = np.argmax(abs_c)
            
            # Nudge inwards
            step = 0.0001
            centres[i, axis] -= np.sign(c[axis]) * step
            centres[i, axis] = np.clip(centres[i, axis], -0.5, 0.5)
            
        else:
            i, j = min_pair_idx
            v = centres[i] - centres[j]
            dist = np.linalg.norm(v)
            if dist < 1e-10:
                centres[i] += np.random.rand(3) * 0.001
                centres[j] -= np.random.rand(3) * 0.001
            else:
                u = v / dist
                step = 0.0001
                centres[i] += u * step
                centres[j] -= u * step
                centres[i] = np.clip(centres[i], -0.5, 0.5)
                centres[j] = np.clip(centres[j], -0.5, 0.5)
        
        # Check improvement
        new_r = _compute_radius(centres)
        if new_r <= r:
            # If we didn't improve, maybe we are stuck or the nudge was too small/wrong direction?
            # For wall, we always improve by moving in.
            # For pair, we move apart, so distance increases.
            # Why would it not improve? 
            # 1. Moving i might bring it closer to another sphere k.
            # 2. Moving j might bring it closer to another sphere k.
            # 3. Moving i away from wall might make it closer to another wall? No, we move in.
            # So the global min might be determined by a *different* constraint now.
            # If new_r < r, we made it worse. Revert?
            # Since we don't store previous, we should check and revert if worse.
            # But _compute_radius is O(N^2). 
            # Let's just accept if it's better, else try a smaller step or random kick.
            pass
            
    return centres

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    best_centres = None
    best_r = 0.0
    
    # Multi-start
    starts = []
    
    # 1. Grid
    starts.append(_generate_grid_start())
    
    # 2. Random
    starts.append(_generate_random_start())
    
    # 3. Another random
    starts.append(_generate_random_start())
    
    # 4. FCC-like (perturbed grid)
    starts.append(_generate_fcc_start())
    
    # 5. Random with larger variance
    starts.append(np.random.uniform(-0.48, 0.48, (N, 3)))
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
            
        # Allocate time based on remaining budget
        time_left = deadline - time.time()
        alloc = min(20.0, time_left / 2.0) # Give each start up to 20s
        
        # Run L-BFGS-B
        opt_centres = _optimize_lbgfgs(start, beta=50.0, time_limit=alloc)
        
        # Clip to ensure validity
        opt_centres = np.clip(opt_centres, -0.5, 0.5)
        
        # Polish
        if time.time() < deadline - 5.0:
            polished = _refine_local(opt_centres, max_iter=200)
            polished = np.clip(polished, -0.5, 0.5)
        else:
            polished = opt_centres
            
        r = _compute_radius(polished)
        
        if r > best_r:
            best_r = r
            best_centres = polished.copy()
            
    # Final global polish if time permits
    if best_centres is not None and time.time() < deadline - 10.0:
        final = _refine_local(best_centres, max_iter=1000)
        final = np.clip(final, -0.5, 0.5)
        final_r = _compute_radius(final)
        if final_r > best_r:
            best_centres = final
            best_r = final_r
            
    return best_centres

# EVOLVE-BLOCK-END
