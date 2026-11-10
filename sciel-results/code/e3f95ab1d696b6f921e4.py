"""Combined solver: smooth log-sum-exp surrogate optimization with force-directed refinement.
Uses a differentiable soft-min objective to maximize the minimum sphere radius via L-BFGS-B,
then polishes with force-directed local search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100


def _min_radius(centers):
    """Compute the maximum common radius for given centers inside [-0.5, 0.5]^3."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    r = min(min_pair / 2.0, min_wall)
    return max(r, 0.0)


def _smooth_objective(x, beta):
    """Log-sum-exp based smooth surrogate for minimizing the 'closeness' of spheres.
    Lower objective = better packing. We minimize this."""
    c = x.reshape(N, 3)
    
    # Pairwise squared distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dists = np.sqrt(dist_sq)
    
    # Wall distances (distance from each center to nearest wall)
    wall = 0.5 - np.abs(c)
    
    # For the objective, we want to maximize min(min_pair_dist/2, min_wall)
    # Equivalently minimize -min(...), or use a soft-min surrogate.
    # Soft-min of values v_i is: -(1/beta) * log(sum(exp(-beta * v_i)))
    # Here our "values" are the constraints: d_ij/2 for pairs, wall_i for walls.
    # We want to maximize the minimum, so we minimize the soft-min of (1/d_ij, 1/wall_i) ... 
    # Actually, let's use: minimize sum of exp(-beta * value) which pushes all values up.
    # Or more directly: the objective to minimize is the soft-min of the constraint values.
    # soft_min(v) = -(1/beta) * log(sum(exp(-beta*v)))
    # As beta->inf, soft_min -> min(v). We want to maximize min(v), so minimize -soft_min(v) = (1/beta)*log(sum(exp(-beta*v)))
    
    # Constraint values: d_ij/2 for each pair, wall_i for each sphere
    pair_vals = dists / 2.0
    np.fill_diagonal(pair_vals, np.inf)  # exclude self
    
    # For efficiency, only consider pairs that are somewhat close, or use all
    # With 100 spheres, 4950 pairs + 100 walls = 5050 values
    # This is manageable for log-sum-exp
    
    exp_pairs = np.exp(-beta * pair_vals)
    np.fill_diagonal(exp_pairs, 0.0)
    
    exp_walls = np.exp(-beta * wall)
    
    S = np.sum(exp_pairs) + np.sum(exp_walls)
    
    if S < 1e-300:
        val = 0.0
    else:
        val = (1.0 / beta) * np.log(S)
    
    return val


def _smooth_gradient(x, beta):
    """Gradient of the smooth objective with respect to x."""
    c = x.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1e-10)
    dists = np.sqrt(dist_sq)
    
    wall = 0.5 - np.abs(c)
    
    pair_vals = dists / 2.0
    np.fill_diagonal(pair_vals, np.inf)
    
    exp_pairs = np.exp(-beta * pair_vals)
    np.fill_diagonal(exp_pairs, 0.0)
    
    exp_walls = np.exp(-beta * wall)
    
    S = np.sum(exp_pairs) + np.sum(exp_walls)
    
    if S < 1e-300:
        return np.zeros_like(x)
    
    # d/dx of (1/beta)*log(S) = (1/(beta*S)) * dS/dx
    # dS/dx = sum over pairs of d(exp(-beta*d_ij/2))/dx + sum over walls of d(exp(-beta*wall_i))/dx
    
    # For pair (i,j): d(exp(-beta*d_ij/2))/d(c_i) = exp(-beta*d_ij/2) * (-beta/2) * d(d_ij)/d(c_i)
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # So contribution to grad[i] = exp(-beta*d_ij/2) * (-beta/2) * (c_i - c_j)/d_ij
    
    # For wall i: d(exp(-beta*wall_i))/d(c_i) = exp(-beta*wall_i) * (-beta) * d(wall_i)/d(c_i)
    # wall_i = min_k (0.5 - |c_ik|) ... actually wall_i = min over axes of (0.5 - |c_i_axis|)
    # d(wall_i)/d(c_i_k) = -sign(c_i_k) if axis k is the binding one
    
    # Simplified: use the gradient of the sum
    # grad[c_i] = (1/(beta*S)) * [ sum_j exp(-beta*d_ij/2) * (-beta/2) * (c_i - c_j)/d_ij + exp(-beta*wall_i) * (-beta) * dwall/dc_i ]
    #           = (1/S) * [ -0.5 * sum_j exp(-beta*d_ij/2) * (c_i - c_j)/d_ij - exp(-beta*wall_i) * dwall/dc_i ]
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    inv_dists = np.zeros_like(dists)
    mask = dists > 1e-10
    inv_dists[mask] = 1.0 / dists[mask]
    np.fill_diagonal(inv_dists, 0.0)
    
    # weight[i,j] = exp(-beta*d_ij/2) / d_ij
    weights = exp_pairs * inv_dists
    np.fill_diagonal(weights, 0.0)
    
    # grad_pair[i] = -0.5 * sum_j weights[i,j] * (c_i - c_j)
    #              = -0.5 * (sum_j weights[i,j] * c_i - sum_j weights[i,j] * c_j)
    #              = -0.5 * (c_i * sum_j weights[i,j] - (weights @ c)_i)
    
    W_sum = np.sum(weights, axis=1)  # shape (N,)
    Wc = weights.dot(c)  # shape (N, 3)
    
    grad = -0.5 * (c * W_sum[:, np.newaxis] - Wc)
    
    # Wall contributions
    # For each sphere i, wall_i = min_k (0.5 - |c_i_k|)
    # We use a soft version: the gradient points away from the nearest wall
    abs_c = np.abs(c)
    # Find the axis with minimum (0.5 - |c_i_k|) for each i
    wall_per_axis = 0.5 - abs_c  # shape (N, 3)
    min_axis = np.argmin(wall_per_axis, axis=1)  # shape (N,)
    
    # d(wall_i)/d(c_i_k) = -sign(c_i_k) if k == min_axis[i], else 0
    wall_grad = np.zeros_like(c)
    for i in range(N):
        k = min_axis[i]
        wall_grad[i, k] = -np.sign(c[i, k])
        if wall_grad[i, k] == 0:
            wall_grad[i, k] = 1.0  # if c=0, push in positive direction
    
    grad += -exp_walls[:, np.newaxis] * wall_grad
    
    # Scale by 1/S
    grad /= S
    
    return grad.reshape(N * 3)


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Initial placement: a jittered grid
    m = 5  # 5x5x5 = 125, we take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].astype(np.float64)

    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.001, size=centers.shape)

    # Clamp to inside cube with small margin
    centers = np.clip(centers, -0.49, 0.49)

    best_centers = centers.copy()
    best_r = _min_radius(best_centers)

    # Phase 1: Smooth optimization with increasing beta
    betas = [5, 10, 20, 50, 100, 200, 500]
    x = centers.flatten()
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        try:
            def obj(x_curr, b=beta):
                return _smooth_objective(x_curr, b)
            
            def grad(x_curr, b=beta):
                return _smooth_gradient(x_curr, b)
            
            res = minimize(obj, x, jac=grad, method='L-BFGS-B',
                           bounds=[(-0.4999, 0.4999)] * (N * 3),
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if res.fun < 1e10:  # check for numerical issues
                x_new = np.clip(res.x, -0.4999, 0.4999)
                c_new = x_new.reshape(N, 3)
                r_new = _min_radius(c_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_centers = c_new.copy()
                
                x = x_new
        except Exception:
            pass
    
    # Phase 2: Force-directed refinement on the best configuration
    centers = best_centers.copy()
    current_r = best_r
    
    learning_rate = 0.002
    decay = 0.9995
    min_lr = 1e-8
    max_steps = 100000
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        if learning_rate < min_lr:
            break

        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)

        # Repulsive force: stronger when closer
        inv_dist = 1.0 / dists
        inv_dist2 = inv_dist * inv_dist
        force_mag = np.minimum(inv_dist2, 50.0)
        np.fill_diagonal(force_mag, 0.0)

        # Force vector
        forces = diff * (force_mag * inv_dist)[:, :, np.newaxis]
        net_forces = np.sum(forces, axis=1)

        # Boundary repulsion
        for axis in range(3):
            left_dist = centers[:, axis] + 0.5
            right_dist = 0.5 - centers[:, axis]
            left_force = np.where(left_dist < 0.15, 1.0 / (left_dist + 1e-6), 0.0)
            right_force = np.where(right_dist < 0.15, 1.0 / (right_dist + 1e-6), 0.0)
            net_forces[:, axis] += -left_force + right_force

        # Weak centering force
        net_forces -= 0.005 * centers

        # Update
        centers += learning_rate * net_forces
        centers = np.clip(centers, -0.4999, 0.4999)

        # Decay
        learning_rate *= decay

        # Check improvement
        r = _min_radius(centers)
        if r > current_r:
            current_r = r
            best_r = r
            best_centers = centers.copy()

    # Phase 3: Fine polish with very small steps
    centers = best_centers.copy()
    for step in range(30000):
        if time.time() > deadline - 2:
            break

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1e-10)

        inv_dist = 1.0 / dists
        inv_dist2 = inv_dist * inv_dist
        force_mag = np.minimum(inv_dist2, 50.0)
        np.fill_diagonal(force_mag, 0.0)

        forces = diff * (force_mag * inv_dist)[:, :, np.newaxis]
        net_forces = np.sum(forces, axis=1)

        for axis in range(3):
            left_dist = centers[:, axis] + 0.5
            right_dist = 0.5 - centers[:, axis]
            left_force = np.where(left_dist < 0.08, 1.0 / (left_dist + 1e-6), 0.0)
            right_force = np.where(right_dist < 0.08, 1.0 / (right_dist + 1e-6), 0.0)
            net_forces[:, axis] += -left_force + right_force

        net_forces -= 0.002 * centers

        centers += 5e-6 * net_forces
        centers = np.clip(centers, -0.4999, 0.4999)

        r = _min_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    # Final validation
    final_r = _min_radius(best_centers)
    if final_r > best_r:
        best_r = final_r

    return best_centers
