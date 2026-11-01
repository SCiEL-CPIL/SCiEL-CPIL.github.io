"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _generate_hex_grid(n):
    """
    Generate a hexagonal lattice and select n points that are well-distributed
    and close to the center to provide a good warm start.
    """
    best_points = None
    best_score = -1
    
    for d in np.linspace(0.06, 0.09, 20):
        num_rows = int(1.0 / (d * np.sqrt(3) / 2)) + 2
        num_cols = int(1.0 / d) + 2
        
        points = []
        
        for j in range(-num_rows, num_rows + 1):
            y = j * d * np.sqrt(3) / 2
            if y < -0.5 or y > 0.5:
                continue
            offset = 0.0
            if j % 2 != 0:
                offset = d / 2.0
            
            for i in range(-num_cols, num_cols + 1):
                x = i * d + offset
                if x < -0.5 or x > 0.5:
                    continue
                points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)
        selected = points[idx[:n]]
        
        diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(selected[:, 0]), 0.5 - np.abs(selected[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        score = min(min_pair / 2, min_boundary)
        
        if score > best_score:
            best_score = score
            best_points = selected.copy()
            
    if best_points is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
        
    return best_points

def construct():
    points = _generate_hex_grid(N)
    
    # Initial radius estimate
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    r = min(min_pair / 2, min_boundary)
    
    def energy_and_grad(p_flat, r_target):
        p = p_flat.reshape(-1, 2)
        # Compute distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise violations
        v_pairs = np.maximum(0, 2 * r_target - dists)
        # Gradient of sum v_pairs^2
        # d/dp_i sum_j v_ij^2 = 2 sum_j v_ij * d v_ij / d p_i
        # If v_ij > 0, d v_ij / d p_i = - (p_i - p_j) / d_ij
        # So term is -2 v_ij (p_i - p_j) / d_ij
        
        # We need to sum over j for each i.
        # Let M_ij = 2 * v_ij / d_ij if v_ij > 0 else 0
        # Then grad_i = - sum_j M_ij (p_i - p_j)
        # grad_i = - (sum_j M_ij) p_i + sum_j M_ij p_j
        
        M = np.zeros_like(dists)
        mask = v_pairs > 0
        M[mask] = 2 * v_pairs[mask] / dists[mask]
        
        row_sums = np.sum(M, axis=1)
        weighted_p = M @ p
        
        pair_grad = - (row_sums[:, np.newaxis] * p - weighted_p)
        
        # Boundary violations
        bx = 0.5 - np.abs(p[:, 0])
        by = 0.5 - np.abs(p[:, 1])
        b = np.minimum(bx, by)
        
        v_bound = np.maximum(0, r_target - b)
        
        # Gradient of sum v_bound^2
        # d/dp_i v_bound_i^2 = 2 v_bound_i * d v_bound_i / d p_i
        # If v_bound_i > 0:
        # b_i is min(bx, by).
        # If bx < by, b = bx.
        #   If x > 0, bx = 0.5 - x, db/dx = -1, db/dy = 0.
        #   If x < 0, bx = 0.5 + x, db/dx = 1, db/dy = 0.
        #   So db/dx = -sign(x).
        # If by < bx, b = by.
        #   db/dy = -sign(y).
        # If both equal, gradient is ambiguous, pick either or average.
        
        bound_grad = np.zeros_like(p)
        for i in range(len(p)):
            if v_bound[i] > 0:
                if bx[i] < by[i]:
                    bound_grad[i, 0] = 2 * v_bound[i] * (-np.sign(p[i, 0]))
                elif by[i] < bx[i]:
                    bound_grad[i, 1] = 2 * v_bound[i] * (-np.sign(p[i, 1]))
                else:
                    # Tie, use both
                    bound_grad[i, 0] = 2 * v_bound[i] * (-np.sign(p[i, 0]))
                    bound_grad[i, 1] = 2 * v_bound[i] * (-np.sign(p[i, 1]))
        
        total_grad = pair_grad + bound_grad
        total_energy = np.sum(v_pairs ** 2) + np.sum(v_bound ** 2)
        
        return total_energy, total_grad.flatten()

    # Optimization Loop
    # Start with current r, increase it
    r_increment = 0.0005
    max_outer_iters = 100
    
    current_r = r
    for _ in range(max_outer_iters):
        # Optimize positions for current_r
        p_flat = points.flatten()
        res = minimize(
            lambda x: energy_and_grad(x, current_r)[0],
            p_flat,
            jac=lambda x: energy_and_grad(x, current_r)[1],
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        points = res.x.reshape(-1, 2)
        
        # Check if we can increase r
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        min_boundary = np.min(boundary_dists)
        
        new_r = min(min_pair / 2, min_boundary)
        
        if new_r > current_r + 1e-7:
            current_r = new_r
        else:
            # If we can't increase, try a smaller increment or break
            # For simplicity, break if stuck
            # But maybe we can try to optimize with a slightly higher r to see if it converges?
            # If energy is 0, we can increase. If energy > 0, we are stuck.
            # Let's check the energy
            e, _ = energy_and_grad(points.flatten(), current_r)
            if e < 1e-6:
                 current_r += r_increment
            else:
                 break

    # Final calculation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return points
# EVOLVE-BLOCK-END
