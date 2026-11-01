import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    def generate_hex_grid(spacing, rotation=0.0):
        """Generate hexagonal lattice points inside unit circle with optional rotation."""
        points = []
        max_j = int(2.0 / (spacing * np.sqrt(3.0))) + 3
        max_i = int(2.0 / spacing) + 3
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Apply rotation
                if rotation != 0.0:
                    xr = x * np.cos(rotation) - y * np.sin(rotation)
                    yr = x * np.sin(rotation) + y * np.cos(rotation)
                else:
                    xr, yr = x, y
                if xr * xr + yr * yr <= 1.0:
                    points.append([xr, yr])
        return np.array(points)

    def compute_score(pts):
        """Compute the packing score (minimum radius)."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_bound = np.min(1.0 - radii)
        
        return min(min_pair / 2.0, min_bound)

    def soft_min_and_grad(x, beta):
        """Compute soft-min and its gradient."""
        x_max = np.max(x)
        exp_neg_beta_x = np.exp(-beta * (x - x_max))
        sum_exp = np.sum(exp_neg_beta_x)
        s_val = -1.0 / beta * (np.log(sum_exp) - beta * x_max)
        grad = exp_neg_beta_x / sum_exp
        return s_val, grad

    def objective_and_grad(flat_pts, beta):
        """Objective function and gradient for L-BFGS-B."""
        pts = flat_pts.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_matrix = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(d_matrix, np.inf)
        
        # Upper triangle mask
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_flat = d_matrix[mask]
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        d_bound = 1.0 - radii
        
        # Effective radii: pairwise/2 and boundary
        eff_rad = np.concatenate([d_flat / 2.0, d_bound])
        
        # Soft min
        s_val, grad_eff = soft_min_and_grad(eff_rad, beta)
        obj = -s_val
        
        # Backpropagate gradient
        grad_pair = grad_eff[:len(d_flat)]
        grad_bound = grad_eff[len(d_flat):]
        
        # Reconstruct gradient matrix
        grad_d_matrix = np.zeros((N, N))
        grad_d_matrix[mask] = grad_pair
        
        # Gradient of d_matrix w.r.t. points
        # d(d_ij)/d(p_i) = (p_i - p_j) / d_ij
        safe_d = np.where(d_matrix < 1e-9, 1e-9, d_matrix)
        
        # Force on point i from pairwise constraints
        # F_i = sum_j grad_d_matrix[i,j] * (p_i - p_j) / d_ij
        # But we need to be careful: grad_d_matrix is only upper triangle.
        # The full gradient contribution for pair (i,j) where i<j:
        #   to p_i: grad_pair[k] * (p_i - p_j) / d_ij
        #   to p_j: -grad_pair[k] * (p_i - p_j) / d_ij
        # So we can compute:
        #   F_i += sum_{j>i} grad_pair[i,j] * (p_i - p_j) / d_ij
        #   F_i -= sum_{j<i} grad_pair[j,i] * (p_j - p_i) / d_ij  (note sign)
        
        # More efficiently:
        # F = grad_d_matrix @ (diff / safe_d[:, :, np.newaxis])
        # But this isn't quite right because grad_d_matrix is not symmetric in the way we need.
        
        # Let's compute directly:
        # For each i, F_i = sum_j [grad_d_matrix[i,j] * (p_i - p_j) / d_ij]
        #                 + sum_j [grad_d_matrix[j,i] * (p_j - p_i) / d_ij]  (from the j side)
        # Wait, let's think again.
        # d_ij = ||p_i - p_j||
        # dd_ij/dp_i = (p_i - p_j) / d_ij
        # dd_ij/dp_j = (p_j - p_i) / d_ij
        # 
        # The objective depends on d_ij (for i<j).
        # d(obj)/dp_i = sum_{j>i} (d(obj)/d(d_ij)) * (d(d_ij)/dp_i)
        #             + sum_{j<i} (d(obj)/d(d_ji)) * (d(d_ji)/dp_i)
        # Note d(d_ji)/dp_i = (p_i - p_j) / d_ji = (p_i - p_j) / d_ij
        # And d(obj)/d(d_ji) = grad_pair[j,i]
        #
        # So: d(obj)/dp_i = sum_{j>i} grad_pair[i,j] * (p_i - p_j) / d_ij
        #                   + sum_{j<i} grad_pair[j,i] * (p_i - p_j) / d_ij
        #
        # We can combine:
        # d(obj)/dp_i = sum_j [ (grad_pair[i,j] if i<j else grad_pair[j,i]) * (p_i - p_j) / d_ij ]
        #
        # Let's build a symmetric matrix G where G[i,j] = G[j,i] = grad_pair[min(i,j), max(i,j)]
        G = np.zeros((N, N))
        G[mask] = grad_pair
        G.T[mask] = grad_pair
        
        # Now F_i = sum_j G[i,j] * (p_i - p_j) / d_ij
        # Note: G[i,i] = 0, d_ii is inf, so no contribution.
        # We can compute this as:
        # F = G @ (diff / safe_d[:, :, np.newaxis])
        # But we need to handle the diagonal.
        # Actually, diff[i,i] = 0, so it's fine.
        
        # Compute F_pair
        F_pair = np.zeros_like(pts)
        # Vectorized: F_pair[i] = sum_j G[i,j] * diff[i,j] / safe_d[i,j]
        # This is (G * (1/safe_d)) @ diff
        # But G * (1/safe_d) has inf on diagonal? No, safe_d has 1e-9 on diagonal, but G is 0.
        # So it's fine.
        weight = G / safe_d
        F_pair = weight @ diff  # This gives sum_j weight[i,j] * diff[i,j]
        
        # Boundary gradient
        # d_bound_i = 1 - ||p_i||
        # d(d_bound_i)/dp_i = -p_i / ||p_i||
        safe_radii = np.where(radii < 1e-9, 1e-9, radii)
        F_bound = -grad_bound[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
        
        total_grad = F_pair + F_bound
        
        return obj, total_grad.flatten()

    # 1. Initialization: Try multiple hexagonal lattices with different spacings and rotations
    best_init = None
    best_init_score = -np.inf
    
    spacings = [0.10, 0.105, 0.106, 0.107, 0.11, 0.115]
    rotations = [0.0, np.pi / 12, np.pi / 6, np.pi / 4]
    
    for sp in spacings:
        for rot in rotations:
            pts = generate_hex_grid(sp, rot)
            if len(pts) < N:
                continue
            
            # Select N points closest to center
            dists_to_center = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            idx = np.argsort(dists_to_center)[:N]
            init_pts = pts[idx].copy()
            
            score = compute_score(init_pts)
            if score > best_init_score:
                best_init_score = score
                best_init = init_pts
    
    if best_init is None:
        # Fallback: random
        best_init = np.random.randn(N, 2)
        best_init /= np.linalg.norm(best_init, axis=1, keepdims=True)
        best_init *= 0.9
        best_init_score = compute_score(best_init)

    current_points = best_init.copy()
    current_best_score = best_init_score
    current_best_points = best_init.copy()

    # 2. Optimization with annealed beta schedule
    # Start with low beta (smoother), increase over time (sharper)
    
    max_time = deadline - start_time - 5.0  # Leave 5s for final checks
    beta_start = 50.0
    beta_end = 20000.0
    
    # We'll do multiple rounds of L-BFGS-B with increasing beta
    # Each round uses a fraction of the time budget
    
    num_rounds = 8
    time_per_round = max_time / num_rounds
    
    for round_idx in range(num_rounds):
        if time.time() > deadline - 5.0:
            break
            
        # Anneal beta
        t = round_idx / max(1, num_rounds - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Run L-BFGS-B
        flat_pts = current_points.flatten()
        
        # Time limit for this round
        round_deadline = time.time() + time_per_round * 0.9
        
        try:
            result = minimize(
                objective_and_grad,
                flat_pts,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'maxfun': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                
                # Ensure points are inside unit circle
                radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                out_of_bounds = radii > 1.0
                if np.any(out_of_bounds):
                    new_pts[out_of_bounds] *= (0.9999 / radii[out_of_bounds])[:, np.newaxis]
                
                new_score = compute_score(new_pts)
                
                if new_score > current_best_score:
                    current_best_score = new_score
                    current_best_points = new_pts.copy()
                    current_points = new_pts.copy()
                    
        except Exception:
            pass
        
        # Check time
        if time.time() > round_deadline:
            continue

    # 3. Final polish pass with high beta
    if time.time() < deadline - 5.0:
        beta_final = 50000.0
        flat_pts = current_points.flatten()
        
        try:
            result = minimize(
                objective_and_grad,
                flat_pts,
                args=(beta_final,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 500,
                    'maxfun': 1000,
                    'ftol': 1e-14,
                    'gtol': 1e-14,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                
                # Ensure points are inside unit circle
                radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                out_of_bounds = radii > 1.0
                if np.any(out_of_bounds):
                    new_pts[out_of_bounds] *= (0.9999 / radii[out_of_bounds])[:, np.newaxis]
                
                new_score = compute_score(new_pts)
                
                if new_score > current_best_score:
                    current_best_score = new_score
                    current_best_points = new_pts.copy()
                    
        except Exception:
            pass

    # 4. Return the best configuration found
    return current_best_points
