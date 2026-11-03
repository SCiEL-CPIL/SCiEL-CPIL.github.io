"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def compute_radius(centers):
    """
    Computes the exact maximum radius for the given circle centers.
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 2):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0

    # Wall distances
    # d = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    dx = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    dy = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    wall_dists = np.minimum(dx, dy)
    min_wall = np.min(wall_dists)

    # Pair distances
    tree = cKDTree(centers)
    # k=2 to get self and nearest neighbor
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    r = min(min_wall, min_pair / 2.0)
    return r

def generate_hex_grid():
    """
    Generates a hexagonal grid packing. Tries a few configurations.
    """
    best_centers = None
    best_r = 0.0
    
    # Try different number of rows
    for n_rows in range(12, 20):
        n_cols = int(np.ceil(N / n_rows))
        
        # Hexagonal spacing
        # We want to fit n_cols circles in width 1, n_rows in height 1.
        # Width span: (n_cols - 1) * s + 2r <= 1
        # Height span: (n_rows - 1) * s * sqrt(3)/2 + 2r <= 1
        # s = 2r
        # (n_cols - 1) 2r + 2r <= 1 => 2r n_cols <= 1 => r <= 1/(2 n_cols)
        # (n_rows - 1) 2r sqrt(3)/2 + 2r <= 1 => r ( (n_rows-1)sqrt(3) + 2 ) <= 1
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                # Offset alternating rows
                offset = 0.0 if i % 2 == 0 else s / 2.0
                x = -0.5 + r + j * s + offset
                y = -0.5 + r + i * s * np.sqrt(3) / 2.0
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        if len(centers) < N:
            continue
            
        centers = np.array(centers[:N])
        
        # Center the grid to ensure it's inside
        cx = (centers[:, 0].max() + centers[:, 0].min()) / 2.0
        cy = (centers[:, 1].max() + centers[:, 1].min()) / 2.0
        centers[:, 0] -= cx
        centers[:, 1] -= cy
        
        # Check if it fits
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            # If it doesn't fit, scale it down slightly? Or skip.
            # Let's just compute radius, it will be 0 if out of bounds
            pass

        r_actual = compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
            
    if best_centers is None:
        # Fallback: Square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

def soft_objective_and_grad(x_flat, k):
    """
    Objective: -softmin(constraints)
    x_flat: (2N,) array of centers
    """
    centers = x_flat.reshape(N, 2)
    
    # Wall constraints
    # d_L = 0.5 - x, d_R = 0.5 + x, d_B = 0.5 - y, d_T = 0.5 + y
    dx = np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5)
    dy = np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    # Note: using min here is non-differentiable, but the outer softmin
    # will handle the selection. However, for gradient, we need to be careful.
    # Actually, let's use the 4 separate wall distances for the softmin set.
    wall_dists = np.concatenate([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ])
    
    # Pair constraints
    # We need all pairwise distances / 2
    # This is O(N^2).
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Upper triangle
    iu = np.triu_indices(N, 1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all constraints
    Y = np.concatenate([wall_dists, pair_dists])
    
    # Softmin
    # To avoid overflow/underflow, subtract min
    Y_min = np.min(Y)
    # If Y_min is very small, exp(-k*Y) might be large.
    # We want sum(exp(-k * (Y - Y_min)))
    exp_terms = np.exp(-k * (Y - Y_min))
    sum_exp = np.sum(exp_terms)
    
    softmin_val = Y_min - np.log(sum_exp) / k
    obj_val = -softmin_val
    
    # Gradient
    # d(softmin)/dY_j = exp_terms[j] / sum_exp
    weights = exp_terms / sum_exp
    
    grad_Y = weights # This is d(softmin)/dY_j
    # We want grad of -softmin, so -grad_Y
    
    # Accumulate gradient for centers
    grad_centers = np.zeros((N, 2))
    
    # Wall parts
    # Y[0:N] are 0.5 - x_i
    # dY/dx_i = -1
    grad_centers[:, 0] -= weights[0:N] * (-1.0) # - (w * -1) = +w? 
    # Wait. obj = -softmin.
    # d(obj)/dx = - d(softmin)/dx = - sum_j (d(softmin)/dY_j * dY_j/dx)
    # d(softmin)/dY_j = weights[j]
    # Y_j = 0.5 - x_i. dY_j/dx_i = -1.
    # So term is - w_j * (-1) = + w_j.
    
    # Let's re-verify signs.
    # f = - S. S = softmin(Y).
    # df/dx = - dS/dx = - sum_j (w_j * dY_j/dx).
    # For Y_j = 0.5 - x_i, dY_j/dx_i = -1.
    # df/dx_i = - w_j * (-1) = w_j.
    # So if a point is close to left wall (small Y), weight is high, gradient pushes x to the right (positive). Correct.
    
    grad_centers[:, 0] += weights[0:N]
    
    # Y[N:2N] are 0.5 + x_i
    # dY/dx_i = 1
    # df/dx_i = - w_j * 1 = -w_j.
    grad_centers[:, 0] -= weights[N:2*N]
    
    # Y[2N:3N] are 0.5 - y_i
    grad_centers[:, 1] += weights[2*N:3*N]
    
    # Y[3N:4N] are 0.5 + y_i
    grad_centers[:, 1] -= weights[3*N:4*N]
    
    # Pair parts
    # Y[4N:] are d_ij / 2
    # w_pair = weights[4*N:]
    # dY_k/dx_i = (x_i - x_j) / (2 * d_ij) ?
    # Y_k = 0.5 * ||x_i - x_j||.
    # dY_k/dx_i = 0.5 * (x_i - x_j) / ||x_i - x_j|| = (x_i - x_j) / (2 * d_ij) * d_ij/d_ij?
    # No. ||x_i - x_j|| = d_ij.
    # d/dx_i (0.5 * d_ij) = 0.5 * (x_i - x_j) / d_ij.
    
    # We need to accumulate this for all pairs.
    # This is O(N^2).
    # We have the weights w_pair corresponding to the upper triangle indices.
    
    # We can use a matrix approach or loop. Loop is slower.
    # Let's use the diff matrix.
    # diff[i,j] = x_i - x_j.
    # dists[i,j] = ||diff[i,j]||.
    # We only care about i < j.
    
    # Create a matrix of weights for pairs
    # W_pair is 1D array of length N(N-1)/2.
    # We need to map this back to the (N, N) matrix.
    
    # Actually, we can compute the gradient contribution directly.
    # grad_x_i += sum_{j != i} w_{ij} * 0.5 * (x_i - x_j) / d_{ij}
    
    # Let's construct the full weight matrix W (N x N), symmetric, 0 diagonal.
    W_full = np.zeros((N, N))
    W_full[iu] = weights[4*N:]
    W_full[iu[1], iu[0]] = weights[4*N:]
    
    # dists_full is N x N.
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-9, dists, 1.0)
    
    # Factor F[i,j] = W[i,j] * 0.5 / dists[i,j]
    F = W_full * 0.5 / dists_safe
    np.fill_diagonal(F, 0.0)
    
    # grad_x_i = sum_j F[i,j] * (x_i - x_j)
    # diff[i,j] = x_i - x_j
    # grad_x = sum_j F[i,j] * diff[i,j]
    grad_pair = np.sum(F[:, :, np.newaxis] * diff, axis=1)
    
    # Total gradient
    grad_centers += grad_pair
    
    # Flatten and negate for objective gradient?
    # No, minimize takes gradient of the function to be minimized.
    # We are minimizing obj_val = -softmin.
    # So we return grad of obj_val.
    # My derivation: df/dx = - sum w_j dY_j/dx.
    # I calculated grad_centers as the sum of terms like w_j.
    # Let's re-check the sign for pairs.
    # df/dx_i = - sum_{j} w_{ij} * (x_i - x_j) / (2 d_{ij})
    # My code: grad_pair = sum F * diff. F = w * 0.5 / d.
    # So grad_pair = sum w * 0.5 / d * (x_i - x_j).
    # This is + sum ...
    # But df/dx should be - sum ...
    # So I need to negate grad_pair?
    # Let's check wall again.
    # df/dx_i (left wall) = w. (Pushes right).
    # If x_i is close to -0.5, Y is small, w is large.
    # Gradient is positive. x increases. Moves away from wall. Correct.
    
    # For pair:
    # If x_i is close to x_j, Y is small, w is large.
    # x_i - x_j is small vector pointing from j to i.
    # df/dx_i = - w * (x_i - x_j) / (2d).
    # If x_i > x_j, x_i - x_j is positive. Gradient is negative.
    # x_i decreases. Moves away from x_j. Correct.
    
    # So grad_centers should be:
    # Wall: +w for left/bottom, -w for right/top.
    # Pair: - sum w * (x_i - x_j) / (2d).
    
    # My code for walls:
    # grad[:, 0] += w_left (positive). Correct.
    # grad[:, 0] -= w_right (negative). Correct.
    
    # My code for pairs:
    # grad_pair = sum F * diff. (Positive sum).
    # I added it: grad_centers += grad_pair.
    # This is WRONG. It should be subtracted.
    
    grad_centers -= grad_pair
    
    return obj_val, grad_centers.flatten()

def construct():
    start_time = time.time()
    
    # 1. Initialization
    centers = generate_hex_grid()
    best_radius = compute_radius(centers)
    best_centers = centers.copy()
    
    if _time_left() < 5.0:
        return best_centers
        
    # 2. Optimization
    # Start with small k, increase
    k_start = 50.0
    k_end = 500.0
    
    # We can run a few rounds
    for round_idx in range(5):
        if _time_left() < 10.0:
            break
            
        k = k_start + (k_end - k_start) * (round_idx / 4.0)
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                soft_objective_and_grad, 
                x0, 
                args=(k,),
                method='L-BFGS-B', 
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_centers = res.x.reshape(N, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_radius = compute_radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
                
        except Exception:
            pass

    # 3. Final Polish (Optional, if time)
    if _time_left() > 5.0:
        # Try a few random perturbations + L-BFGS-B
        for _ in range(5):
            if _time_left() < 5.0:
                break
            perturbed = best_centers + np.random.normal(0, 0.001, (N, 2))
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            x0 = perturbed.flatten()
            try:
                res = minimize(
                    soft_objective_and_grad, 
                    x0, 
                    args=(k_end,),
                    method='L-BFGS-B', 
                    jac=True,
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                new_centers = res.x.reshape(N, 2)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_radius = compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
            except Exception:
                pass

    return best_centers

# EVOLVE-BLOCK-END
