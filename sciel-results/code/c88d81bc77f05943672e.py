"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_bcc():
    """Body-Centered Cubic initialization."""
    m = int(np.ceil((N / 2) ** (1.0 / 3.0) - 1e-9))
    if m < 1: m = 1
    # Generate BCC points in a grid of size m
    # Points: (i, j, k) and (i+0.5, j+0.5, k+0.5)
    # Scale to fit in [-0.5, 0.5]
    # Spacing s = 1.0 / m
    s = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corners of the sub-cube
                points.append(((i + 0.25) * s - 0.5, (j + 0.25) * s - 0.5, (k + 0.25) * s - 0.5))
                # Center of the sub-cube
                points.append(((i + 0.75) * s - 0.5, (j + 0.75) * s - 0.5, (k + 0.75) * s - 0.5))
    
    pts = np.array(points)
    # Shuffle and take N
    np.random.shuffle(pts)
    return pts[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

def _softmin_val(c, beta=100.0):
    """
    Differentiable approximation of the negative minimum distance.
    We want to maximize min_dist. So we minimize -min_dist.
    Using log-sum-exp: -min(x) = (1/beta) * log(sum(exp(-beta*x)))
    Here x are the constraints (distances).
    Constraints:
    1. Pairwise distances d_ij
    2. Wall distances w_ik
    
    We construct a vector of all constraints and apply softmin.
    """
    # Pairwise
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    # We want distances, but log-sum-exp works better with squared distances?
    # Actually, min distance is sqrt(min dist2).
    # Let's just use the values directly.
    # dist = sqrt(dist2). But sqrt is not differentiable at 0, though we are far from 0.
    # However, calculating full dist matrix is O(N^2).
    
    # To save memory and time, we can sample or just use full matrix since N=100 is small.
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    
    # Wall distances
    # d_wall_i = min_k (0.5 - |c_ik|)
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c # shape (N, 3)
    min_wall_per_sphere = np.min(wall_dists, axis=1) # shape (N,)
    
    # Combine all constraints into one vector
    # We want to minimize the soft-min of these values.
    # softmin(x) = (1/beta) * log(sum(exp(-beta * x)))
    # This approximates min(x).
    
    # Flatten dist (excluding diagonal)
    mask = ~np.eye(N, dtype=bool)
    dist_flat = dist[mask]
    
    # Combine
    all_vals = np.concatenate([dist_flat, min_wall_per_sphere])
    
    # Numerical stability: shift by max
    max_val = np.max(all_vals)
    # We want min, so we look at exp(-beta * (val - max_val)) ?
    # Actually, standard log-sum-exp for sum(exp(x)) uses max.
    # For min, min(x) = -max(-x).
    # So softmin(x) = -softmax(-x).
    # softmax(x) = log(sum(exp(x - max(x))))
    
    neg_vals = -all_vals
    max_neg = np.max(neg_vals)
    # exp(neg_vals - max_neg)
    # If beta is large, this is stable.
    # Let's just compute directly with scaling
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    
    # log(sum(exp(scaled - max_scaled))) + max_scaled
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    
    val = log_sum_exp / beta
    
    return val

def _softmin_grad(c, beta=100.0):
    """
    Gradient of the soft-min objective.
    f(c) = (1/beta) * log(sum(exp(-beta * x_k(c))))
    df/dc = (1/beta) * (1/S) * sum(exp(-beta * x_k) * (-beta) * dx_k/dc)
          = - (1/S) * sum(exp(-beta * x_k) * dx_k/dc)
    where S = sum(exp(-beta * x_k))
    """
    # Recalculate terms
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall_per_sphere = np.min(wall_dists, axis=1)
    
    mask = ~np.eye(N, dtype=bool)
    dist_flat = dist[mask]
    
    all_vals = np.concatenate([dist_flat, min_wall_per_sphere])
    
    # Calculate weights w_k = exp(-beta * x_k) / S
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    S = np.sum(exp_scaled)
    weights = exp_scaled / S
    
    # Gradient is - sum(w_k * grad(x_k))
    grad = np.zeros_like(c)
    
    # 1. Pairwise part
    # x_ij = |c_i - c_j|
    # grad_c_i (x_ij) = (c_i - c_j) / |c_i - c_j|
    # grad_c_j (x_ij) = (c_j - c_i) / |c_i - c_j| = - grad_c_i
    
    # We need to map flat weights back to pairs
    # It's easier to iterate or use vectorized accumulation
    # N=100, so N^2 = 10000. Iteration is fine.
    
    # Reconstruct dist matrix for gradient direction
    # dist is already computed
    
    # Weights for pairs
    # indices of dist_flat correspond to upper triangle? No, mask is full off-diagonal.
    # Let's just loop over i < j to be safe and clear, N=100 is small.
    
    # Actually, let's vectorize the pairwise part properly.
    # Create a matrix of weights for each pair (i,j)
    # The flat array `dist_flat` is row-major order of the masked matrix.
    # It's complex to map back. Let's just use the matrix form.
    
    # Recreate weight matrix for pairs
    # w_ij = exp(-beta * dist_ij) / S_pair_part? No, S is global sum.
    # This coupling makes it hard to vectorize simply without knowing S_pair vs S_wall.
    # Let's just compute the full gradient via finite differences? No, too slow.
    # Let's compute the weights matrix explicitly.
    
    # To avoid re-computing exp for every pair, let's just assume the wall part is small or handle it separately?
    # No, must be exact.
    
    # Let's compute the contribution of each pair (i,j) to the gradient.
    # grad = - sum_{i<j} w_ij * (u_ij e_i - u_ij e_j) - sum_i w_wall_i * grad_wall_i
    # where u_ij = (c_i - c_j) / dist_ij
    
    # We need w_ij for all pairs.
    # Let's create a matrix W_pair of shape (N, N)
    # and W_wall of shape (N,)
    
    # We have all_vals.
    # First N*(N-1)/2 elements? No, mask is N*N - N elements.
    # Let's just re-calculate the weights in matrix form to be safe.
    
    # Re-calculate exp values in matrix form
    # This duplicates work but is simpler and robust for N=100.
    
    # Pairwise
    # dist matrix is ready.
    # We need to zero out diagonal.
    dist_for_exp = dist.copy()
    np.fill_diagonal(dist_for_exp, np.inf)
    
    exp_pair = np.exp(-beta * dist_for_exp)
    # Set diagonal to 0
    np.fill_diagonal(exp_pair, 0.0)
    
    # Wall
    # min_wall_per_sphere is ready.
    # But we need the gradient of min_wall.
    # min_k (0.5 - |c_ik|)
    # The gradient is determined by the minimizing axis k.
    # If multiple axes tie, average gradients? Or just pick one?
    # For soft-min, the weight is distributed based on exp values.
    # Wait, my softmin formulation treated min_wall_per_sphere as a single value per sphere.
    # The gradient of min(a,b,c) is the gradient of the minimum.
    # If I use softmin on the vector [d_11, d_12, d_13, d_21...], I need to expand the wall terms.
    
    # Let's refine the objective to be on the full vector of all constraints.
    # Constraints:
    # 1. d_ij for i<j
    # 2. 0.5 - c_ik and c_ik + 0.5 for all i,k?
    # No, distance to wall is min(0.5-c, 0.5+c).
    # Let's use the 6 wall planes explicitly for each sphere.
    # This gives 6N wall constraints.
    
    # Let's restart the gradient calculation with a clearer structure.
    
    # Objective: f(c) = softmin( [d_ij (i<j), w_ik (i=1..N, k=1..3, planes)] )
    # This is cleaner.
    
    # 1. Pairwise distances
    # We already have `dist` matrix.
    # Upper triangle only to avoid double counting?
    # If I sum over all i!=j, I double count.
    # The softmin is symmetric. It's fine to include all i!=j or just i<j.
    # Including all i!=j doubles the weight, which effectively changes beta.
    # Let's stick to i<j for pairs to be consistent with standard distance definition.
    
    # Let's build the full list of constraint values and their gradients.
    
    # This is getting complicated to vectorize perfectly in one shot.
    # Given N=100, a semi-vectorized approach or even a loop for gradient might be acceptable if fast enough.
    # 10000 pairs.
    
    # Let's try a simpler gradient:
    # Just use the weights from the `all_vals` calculation but map them back.
    
    # Let's assume the wall part is handled by the min_wall_per_sphere value.
    # The gradient of min_wall_i is the unit vector pointing to the nearest wall.
    # If we use the softmin over [d_flat, min_wall], the weight for min_wall_i is w_wall_i.
    # The gradient contribution is -w_wall_i * grad(min_wall_i).
    
    # Let's compute w_pair_matrix and w_wall_vector.
    # This requires knowing how the flat array maps.
    
    # Alternative: Use scipy.optimize.minimize with a function that computes the score and gradient via finite differences?
    # No, analytical is better.
    
    # Let's implement the gradient by iterating over pairs, which is O(N^2) = 10k ops. Very fast.
    
    grad = np.zeros_like(c)
    
    # Recalculate weights to avoid inconsistency
    # We need the global S.
    # Let's compute S from the previous step's logic.
    # Actually, let's just re-calculate everything inside _softmin_grad to be self-contained.
    
    # 1. Distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, np.inf)
    
    # 2. Wall distances (min per sphere)
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall_per_sphere = np.min(wall_dists, axis=1)
    
    # 3. Flatten for weight calculation
    # To be consistent with _softmin_val, we used dist_flat (all off-diag) and min_wall.
    # Let's stick to that.
    mask = ~np.eye(N, dtype=bool)
    dist_flat = dist[mask]
    all_vals = np.concatenate([dist_flat, min_wall_per_sphere])
    
    scaled = -beta * all_vals
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    S = np.sum(exp_scaled)
    weights = exp_scaled / S
    
    # 4. Apply weights
    # Pairwise part
    # weights[0 : N*(N-1)] correspond to dist_flat
    # We need to add -w_ij * (c_i - c_j)/dist_ij to grad[i] and grad[j]
    
    # It is inefficient to iterate 10k times in Python.
    # Let's vectorize.
    # Create a matrix W of shape (N, N) from the flat weights.
    # The mask is row-major.
    # We can create a zero matrix and fill it.
    
    # Actually, let's just use the fact that the gradient of the sum is the sum of gradients.
    # grad_pair = - sum_{i!=j} w_ij * u_ij e_i ...
    
    # Let's create a weight matrix W_pair
    W_pair = np.zeros((N, N))
    # Fill W_pair with weights
    # The indices in dist_flat are [0, 1, ..., N-1] for row 0 (skipping 0), etc.
    # This mapping is annoying.
    
    # Let's just use the full matrix approach for weights.
    # Re-calculate exp_pair_matrix
    dist_for_exp = dist.copy()
    np.fill_diagonal(dist_for_exp, np.inf)
    exp_pair = np.exp(-beta * dist_for_exp)
    np.fill_diagonal(exp_pair, 0.0)
    
    # We need to normalize by the GLOBAL S, not just the pair sum.
    # S = sum(exp_pair) + sum(exp_wall)
    
    # Calculate exp_wall
    # min_wall_per_sphere is a scalar per sphere.
    # exp_wall = exp(-beta * min_wall_per_sphere)
    exp_wall = np.exp(-beta * min_wall_per_sphere)
    
    S_pair = np.sum(exp_pair)
    S_wall = np.sum(exp_wall)
    S_total = S_pair + S_wall
    
    # Normalize
    W_pair = exp_pair / S_total
    W_wall = exp_wall / S_total
    
    # Now compute gradient
    # Pairwise:
    # grad[i] += - sum_j W_pair[i,j] * (c[i] - c[j]) / dist[i,j]
    
    # Vectorized:
    # u = (c[:, None, :] - c[None, :, :]) / dist[:, :, None]
    # Handle dist=inf (diagonal) -> u=0
    with np.errstate(divide='ignore', invalid='ignore'):
        u = diff / dist[:, :, None]
    # Set diagonal to 0
    np.fill_diagonal(u[:, :, 0], 0.0)
    np.fill_diagonal(u[:, :, 1], 0.0)
    np.fill_diagonal(u[:, :, 2], 0.0)
    
    # grad_pair[i] = - sum_j W_pair[i,j] * u[i,j]
    grad_pair = -np.sum(W_pair[:, :, None] * u, axis=1)
    
    # Wall:
    # grad_wall[i] = - W_wall[i] * grad(min_wall_i)
    # min_wall_i = min_k (0.5 - |c_ik|)
    # grad is -sign(c_ik) for the minimizing k.
    
    # Find minimizing axis for each sphere
    # wall_dists is (N, 3)
    min_idx = np.argmin(wall_dists, axis=1) # shape (N,)
    
    # Create gradient matrix for walls
    # grad_wall_mat = zeros (N, 3)
    # For each i, grad_wall_mat[i, min_idx[i]] = -sign(c[i, min_idx[i]])
    # If c[i, k] > 0, dist is 0.5 - c, grad is -1.
    # If c[i, k] < 0, dist is 0.5 + c, grad is +1.
    # So grad is -sign(c).
    
    signs = np.sign(c)
    # If c is 0, sign is 0. Gradient is undefined, but measure zero.
    
    # grad_wall_vec = zeros (N, 3)
    grad_wall_vec = np.zeros_like(c)
    # Scatter
    grad_wall_vec[np.arange(N), min_idx] = -signs[np.arange(N), min_idx]
    
    # Combine
    grad_wall_contrib = -W_wall[:, None] * grad_wall_vec
    
    grad = grad_pair + grad_wall_contrib
    
    return grad

def _polish(c, beta=100.0):
    """Polish the configuration using L-BFGS-B on the soft-min objective."""
    x0 = c.flatten()
    
    def func(x):
        c_x = x.reshape(N, 3)
        return _softmin_val(c_x, beta)
        
    def jac(x):
        c_x = x.reshape(N, 3)
        g = _softmin_grad(c_x, beta)
        return g.flatten()
        
    try:
        res = minimize(func, x0, jac=jac, method='L-BFGS-B', 
                       bounds=[(-0.5, 0.5)] * (N*3),
                       options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.x.reshape(N, 3)
    except:
        return c

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initialization
    c = _init_bcc()
    best_c = c.copy()
    best_score = _score(c)
    
    # Also try grid and random
    c_grid = _init_grid()
    s_grid = _score(c_grid)
    if s_grid > best_score:
        best_score = s_grid
        best_c = c_grid.copy()
        
    for _ in range(3):
        if time.time() > deadline: break
        c_rand = _init_random()
        s_rand = _score(c_rand)
        if s_rand > best_score:
            best_score = s_rand
            best_c = c_rand.copy()
            
    c = best_c.copy()
    
    # Simulated Annealing
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.995
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() > deadline - 10.0: # Leave time for polish
            break
            
        # Random move
        # Pick a few spheres to move
        n_moves = 3
        indices = np.random.choice(N, n_moves, replace=False)
        
        c_new = c.copy()
        for idx in indices:
            # Gaussian step
            step = np.random.randn(3) * 0.05
            c_new[idx] += step
            # Clamp
            c_new[idx] = np.clip(c_new[idx], -0.499, 0.499)
            
        s_old = best_score if i == 0 else _score(c) # Optimization: keep current score
        # Actually, we should track current score
        if i == 0:
            s_cur = _score(c)
        else:
            s_cur = s_prev
            
        s_new = _score(c_new)
        
        # Accept if better or with probability exp(-delta/T)
        if s_new > s_cur:
            c = c_new
            s_prev = s_new
            if s_new > best_score:
                best_score = s_new
                best_c = c.copy()
        else:
            delta = s_new - s_cur
            if np.random.rand() < np.exp(-delta / T):
                c = c_new
                s_prev = s_new
            else:
                s_prev = s_cur # Keep old score for next iter? No, s_cur is current state score.
                
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final Polish
    if time.time() < deadline - 5.0:
        c_polished = _polish(c)
        s_polished = _score(c_polished)
        if s_polished > best_score:
            best_score = s_polished
            best_c = c_polished.copy()
            
        # Polish best_c too if it's different
        if not np.allclose(c, best_c):
            c_polished2 = _polish(best_c)
            s_polished2 = _score(c_polished2)
            if s_polished2 > best_score:
                best_score = s_polished2
                best_c = c_polished2.copy()

    return best_c
# EVOLVE-BLOCK-END
