"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit
    
    def min_radius(c):
        """Compute the minimum radius for a given set of centers."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def epigraph_objective(x, beta):
        """
        Epigraph formulation: optimize centers c and radius t simultaneously.
        Minimize -t subject to soft constraints that all distances >= t.
        x = [c_1, c_2, ..., c_N, t]
        """
        n = len(x)
        c = x[:n-1].reshape(-1, 3)
        t = x[n-1]
        
        # Pairwise squared distances / 4
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        pair_dists_sq = d_sq / 4.0
        
        # Wall squared distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        wall_dists_sq = dist_to_walls ** 2
        
        # Soft penalty: sum of exp(-beta * (dist_sq - t^2)) for all constraints
        t_sq = t * t
        all_dists_sq = np.concatenate([pair_dists_sq.flatten(), wall_dists_sq])
        
        # Only penalize constraints that are violated (dist_sq < t_sq)
        violations = np.maximum(0, t_sq - all_dists_sq)
        
        if np.max(violations) < 1e-15:
            return -t
        
        # Soft penalty
        max_v = np.max(violations)
        exp_vals = np.exp(-beta * (violations - max_v))
        sum_exp = np.sum(exp_vals)
        
        if sum_exp < 1e-300:
            penalty = max_v
        else:
            penalty = max_v - np.log(sum_exp) / beta
        
        return -t + 0.1 * penalty
    
    def epigraph_gradient(x, beta):
        """
        Gradient of the epigraph objective.
        """
        n = len(x)
        c = x[:n-1].reshape(-1, 3)
        t = x[n-1]
        
        grad = np.zeros_like(x)
        
        # Pairwise squared distances / 4
        diff = c[:, np.newaxis, :] - c[np.newaxis, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        pair_dists_sq = d_sq / 4.0
        
        # Wall squared distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        wall_dists_sq = dist_to_walls ** 2
        
        t_sq = t * t
        all_dists_sq = np.concatenate([pair_dists_sq.flatten(), wall_dists_sq])
        
        # Only consider violated constraints
        violations = np.maximum(0, t_sq - all_dists_sq)
        
        if np.max(violations) < 1e-15:
            # No violations, gradient is just -1 for t
            grad[n-1] = -1.0
            return grad
        
        max_v = np.max(violations)
        exp_vals = np.exp(-beta * (violations - max_v))
        sum_exp = np.sum(exp_vals)
        
        if sum_exp < 1e-300:
            weights = np.zeros_like(violations)
            weights[np.argmax(violations)] = 1.0
        else:
            weights = exp_vals / sum_exp
        
        # Gradient of penalty with respect to t: 2t * sum(weights)
        grad_t_penalty = 2.0 * t * np.sum(weights)
        
        # Gradient of penalty with respect to centers
        grad_c = np.zeros_like(c)
        
        # Pairwise contribution (vectorized)
        n_pairs = N * (N - 1) // 2
        pair_weights = weights[:n_pairs]
        
        # Create index arrays for pairs
        idx_i, idx_j = np.triu_indices(N, k=1)
        
        # Only process pairs with significant weight
        mask = pair_weights > 1e-10
        if np.any(mask):
            i_idx = idx_i[mask]
            j_idx = idx_j[mask]
            w = pair_weights[mask]
            
            vecs = c[i_idx] - c[j_idx]  # shape (num_active, 3)
            
            # d(pair_dists_sq[i,j])/dc_i = (c_i - c_j) / 2
            # d(pair_dists_sq[i,j])/dc_j = (c_j - c_i) / 2
            # Penalty gradient: -weights * d(violation)/dc
            # violation = t_sq - dist_sq, so d(violation)/dc_i = -d(dist_sq)/dc_i = -(c_i - c_j)/2
            
            # Using np.add.at for accumulation
            for k in range(len(w)):
                grad_c[i_idx[k]] += vecs[k] * w[k] / 2.0
                grad_c[j_idx[k]] -= vecs[k] * w[k] / 2.0
        
        # Wall contribution
        wall_weights = weights[n_pairs:]
        for i in range(N):
            w = wall_weights[i]
            if w > 1e-10:
                for k in range(3):
                    if c[i, k] > 0:
                        # dist = 0.5 - c[i,k], d(dist_sq)/dc = 2*dist*(-1) = -2*dist
                        # violation = t_sq - dist_sq, d(violation)/dc = 2*dist
                        grad_c[i, k] += 2.0 * (0.5 - c[i, k]) * w
                    else:
                        # dist = c[i,k] + 0.5, d(dist_sq)/dc = 2*dist
                        # d(violation)/dc = -2*dist
                        grad_c[i, k] -= 2.0 * (c[i, k] + 0.5) * w
        
        # Assemble gradient
        grad[:n-1] = 0.1 * grad_c.flatten()
        grad[n-1] = -1.0 + 0.1 * grad_t_penalty
        
        return grad
    
    def spectral_init(n_points, seed=0):
        """Generate diverse initial configurations using spectral methods."""
        rng = np.random.RandomState(seed)
        
        # Start with a random configuration
        centers = rng.uniform(-0.4, 0.4, size=(n_points, 3))
        
        # Perform some relaxation iterations
        for _ in range(50):
            # Compute forces
            forces = np.zeros_like(centers)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            min_d = np.min(d)
            target_d = min_d * 1.2
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if d[i, j] < target_d:
                        vec = centers[i] - centers[j]
                        norm = d[i, j]
                        if norm > 1e-9:
                            force = (target_d - d[i, j]) / norm
                            forces[i] += vec * force
                            forces[j] -= vec * force
            
            # Wall repulsion
            for i in range(n_points):
                for k in range(3):
                    if centers[i, k] > 0.3:
                        forces[i, k] -= (centers[i, k] - 0.3) * 2.0
                    elif centers[i, k] < -0.3:
                        forces[i, k] -= (centers[i, k] + 0.3) * 2.0
            
            centers += forces * 0.1
            centers = np.clip(centers, -0.45, 0.45)
        
        return centers
    
    # Generate multiple diverse initial configurations
    n_starts = 8
    initial_configs = []
    for i in range(n_starts):
        if i == 0:
            # Grid initialization
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            initial_configs.append(grid[:N].copy())
        else:
            initial_configs.append(spectral_init(N, seed=i * 100 + 42))
    
    best_centers = None
    best_r = -1.0
    
    # Time budget allocation
    time_per_start = (time_limit - 5.0) / n_starts
    
    for start_idx, init_centers in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        centers = init_centers.copy()
        current_r = min_radius(centers)
        
        # Annealed beta schedule for L-BFGS-B optimization with epigraph formulation
        beta_schedule = [10, 50, 100, 200, 500, 1000]
        
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Initial t estimate
            t_init = max(0.05, min_radius(centers) * 1.05)
            
            # State: [centers..., t]
            x0 = np.concatenate([centers.flatten(), [t_init]])
            
            # Bounds: centers in [-0.5+eps, 0.5-eps], t in [0, 0.5-eps]
            bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3) + [(0.0, 0.5 - 1e-9)]
            
            try:
                result = minimize(
                    epigraph_objective,
                    x0,
                    jac=lambda x: epigraph_gradient(x, beta),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x_opt = result.x
                centers = x_opt[:N*3].reshape(-1, 3)
            except Exception:
                pass
            
            current_r = min_radius(centers)
            
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        # Final polish with very high beta
        if time.time() - start_time < time_limit - 3.0:
            beta_final = 10000
            t_init = max(0.05, min_radius(centers) * 1.05)
            x0 = np.concatenate([centers.flatten(), [t_init]])
            
            try:
                result = minimize(
                    epigraph_objective,
                    x0,
                    jac=lambda x: epigraph_gradient(x, beta_final),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                x_opt = result.x
                centers = x_opt[:N*3].reshape(-1, 3)
            except Exception:
                pass
            
            current_r = min_radius(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
    
    # Ensure we have a valid result
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
