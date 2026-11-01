"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # --- Helper Functions ---

    def hexagonal_grid(n, width=1.0, height=1.0):
        """Generate a hexagonal grid of points within [0,1]^2."""
        points = []
        # Hexagonal spacing
        # For a hexagon with flat top, horizontal distance is s, vertical is s*sqrt(3)
        # Let's just generate a dense grid and filter
        # Or better, generate rows
        # Approximate number of points
        # We want roughly n points.
        # Let's try a grid of 6x6 or 7x7 and select closest to center or random subset?
        # Actually, for packing, a regular hexagonal lattice is best.
        # Let's generate a lattice with spacing s such that we get ~N points.
        # Area of unit square = 1. Area per hexagon ~ s^2 * sqrt(3)/2.
        # N ~ 1 / (s^2 * sqrt(3)/2) => s ~ sqrt(2 / (N * sqrt(3)))
        s = math.sqrt(2.0 / (N * math.sqrt(3)))
        
        # Generate points
        row = 0
        while True:
            y = (row + 0.5) * s * math.sqrt(3)
            if y > 1.0:
                break
            offset = 0 if row % 2 == 0 else s / 2.0
            col = 0
            while True:
                x = (col + 0.5) * s + offset
                if x > 1.0:
                    break
                points.append((x, y))
                col += 1
            row += 1
        
        points = np.array(points)
        if len(points) < N:
            # If we don't have enough, add random points or expand grid
            # For N=32, this should be enough with small s
            # Let's just use random points if insufficient
            extra = np.random.rand(N - len(points), 2)
            points = np.vstack([points, extra])
        
        # Shuffle and take first N
        np.random.shuffle(points)
        return points[:N]

    def compute_exact_radii(centers):
        """Compute exact radii for a set of centers."""
        n = len(centers)
        radii = np.zeros(n)
        for i in range(n):
            # Wall distance
            r_wall = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            min_d = r_wall
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    min_d = min(min_d, d / 2)
            radii[i] = min_d
        return radii

    def soft_min_obj_and_grad(centers_flat, beta=50.0):
        """
        Compute the soft-min objective and gradient.
        Objective: -sum(soft_min(distances_i))
        distances_i includes wall distances and half pairwise distances.
        """
        n = len(centers_flat) // 2
        c = centers_flat.reshape(n, 2)
        
        # Compute pairwise distances
        # D[i,j] = ||c[i] - c[j]||
        # Use broadcasting
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Set diagonal to inf so they don't contribute to min
        np.fill_diagonal(dists, np.inf)
        
        # Candidate radii from other circles: dists / 2
        cand_radii = dists / 2.0
        
        # Wall distances
        # c[i][0], 1-c[i][0], c[i][1], 1-c[i][1]
        wall_dists = np.array([
            c[:, 0], 
            1 - c[:, 0], 
            c[:, 1], 
            1 - c[:, 1]
        ]).T # Shape (n, 4)
        
        # Combine candidates: (n, n-1 + 4)
        # We need to be careful with the shape for log-sum-exp
        # Let's just iterate over i for the soft-min part if vectorization is tricky, 
        # but n=32 is small. However, vectorizing the sum is easy.
        
        # To avoid numerical issues with large beta, we use the log-sum-exp trick
        # softmin(d) = -1/beta * log( sum( exp(-beta * d) ) )
        
        # We want to maximize sum(softmin_i). So we minimize -sum(softmin_i).
        
        obj_val = 0.0
        grad = np.zeros_like(centers_flat)
        
        for i in range(n):
            # Candidates for circle i
            # dists[i, :] has inf at i. We exclude it.
            d_circ = cand_radii[i].copy()
            d_circ = d_circ[np.isfinite(d_circ)] # Remove inf
            d_wall = wall_dists[i]
            
            all_d = np.concatenate([d_circ, d_wall])
            
            # Soft min calculation
            # To be stable, subtract max (or min, depending on sign)
            # We are computing log(sum(exp(-beta * d)))
            # Let m = min(all_d). Then -beta * all_d + beta * m <= 0.
            # exp(-beta * (all_d - m))
            m = np.min(all_d)
            exp_terms = np.exp(-beta * (all_d - m))
            sum_exp = np.sum(exp_terms)
            
            # softmin_val = m - (1/beta) * log(sum_exp)
            softmin_val = m - (1.0 / beta) * np.log(sum_exp)
            
            obj_val -= softmin_val # We minimize negative sum
            
            # Gradient
            # d(softmin)/d(d_k) = exp(-beta * (d_k - m)) / sum_exp
            weights = exp_terms / sum_exp
            
            # Now map these weights back to gradients with respect to coordinates
            # d_k corresponds to either a pairwise distance or a wall distance.
            
            # 1. Wall distances
            # d_wall = [x, 1-x, y, 1-y]
            # grad wrt x: weights[0] * 1 + weights[1] * (-1)
            # grad wrt y: weights[2] * 1 + weights[3] * (-1)
            gw_x = weights[0] - weights[1]
            gw_y = weights[2] - weights[3]
            
            # 2. Pairwise distances
            # d_{ij} = ||c_i - c_j||
            # d(d_{ij})/d(c_i) = (c_i - c_j) / ||c_i - c_j||
            # d(d_{ij})/d(c_j) = (c_j - c_i) / ||c_i - c_j||
            
            # The weights for pairwise distances are in weights[4:] corresponding to d_circ
            # d_circ was constructed from cand_radii[i] which is dists[i]/2
            # So d_k = dists[i, j] / 2 for j != i
            # d(d_k)/d(c_i) = 0.5 * (c_i - c_j) / dists[i, j]
            
            # We need to sum over j
            # Let's reconstruct the indices
            # d_circ has length n-1.
            # The original dists[i] has length n, with inf at i.
            # We removed the inf. The order is preserved for j != i.
            
            # Let's just loop over j for the gradient accumulation for this i
            # This is O(N^2) per objective call, which is fine for N=32.
            
            grad_x_i = gw_x
            grad_y_i = gw_y
            
            # Map weights[4:] to j indices
            # We need to know which j corresponds to which weight
            # d_circ was cand_radii[i] with inf removed.
            # cand_radii[i] is dists[i]/2.
            # Let's just re-iterate j
            w_idx = 4
            for j in range(n):
                if i == j:
                    continue
                w = weights[w_idx]
                w_idx += 1
                
                # d_k = 0.5 * ||c_i - c_j||
                # diff = c_i - c_j
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                # Gradient of d_k wrt c_i is 0.5 * diff / dist
                # Gradient of objective wrt c_i is - (softmin_grad * d(d_k)/dc_i)
                # Wait, obj = - sum(softmin).
                # d(obj)/dc_i = - sum_k (w_k * d(d_k)/dc_i)
                
                grad_vec = 0.5 * diff / dist
                grad_x_i -= w * grad_vec[0]
                grad_y_i -= w * grad_vec[1]
                
                # Gradient wrt c_j
                # d(d_k)/dc_j = -0.5 * diff / dist
                # d(obj)/dc_j = - (w_k * (-0.5 * diff / dist)) = + w_k * 0.5 * diff / dist
                grad_j_idx = j * 2
                grad[grad_j_idx] += w * 0.5 * diff[0] / dist
                grad[grad_j_idx + 1] += w * 0.5 * diff[1] / dist
                
            # Assign gradient for c_i
            grad_i_idx = i * 2
            grad[grad_i_idx] = grad_x_i
            grad[grad_j_idx] = grad_y_i # Typo fix: grad[grad_i_idx + 1]
            grad[grad_i_idx + 1] = grad_y_i

        return obj_val, grad

    # --- Main Optimization Loop ---

    best_sum = -np.inf
    best_centers = None
    best_radii = None

    np.random.seed(42)

    # Generate initial points
    init_centers = hexagonal_grid(N)

    # Multi-start optimization
    n_restarts = 10
    beta = 50.0 # Start with moderate beta

    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        if restart == 0:
            c0 = init_centers.copy()
        else:
            # Perturb the hexagonal grid or use random
            if restart % 2 == 0:
                c0 = init_centers + np.random.randn(N, 2) * 0.05
            else:
                c0 = np.random.rand(N, 2)
            
            # Clip to valid range
            c0 = np.clip(c0, 0.01, 0.99)

        c0_flat = c0.flatten()

        # Use L-BFGS-B
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda x: soft_min_obj_and_grad(x, beta)[0],
                c0_flat,
                jac=lambda x: soft_min_obj_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < soft_min_obj_and_grad(c0_flat, beta)[0]:
                c_new = np.clip(res.x.reshape(N, 2), 1e-6, 1 - 1e-6)
                
                # Evaluate exact objective
                r_new = compute_exact_radii(c_new)
                sum_new = np.sum(r_new)
                
                if sum_new > best_sum:
                    best_sum = sum_new
                    best_centers = c_new.copy()
                    best_radii = r_new.copy()
                    
        except Exception as e:
            pass

    # --- Final Polish ---
    # Use the best result found and do a high-precision local search
    # or run another optimization with higher beta
    
    if best_centers is not None and time.time() < deadline - 5:
        # Increase beta for sharper min approximation
        beta_high = 200.0
        c0_flat = best_centers.flatten()
        
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda x: soft_min_obj_and_grad(x, beta_high)[0],
                c0_flat,
                jac=lambda x: soft_min_obj_and_grad(x, beta_high)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < soft_min_obj_and_grad(c0_flat, beta_high)[0]:
                c_polished = np.clip(res.x.reshape(N, 2), 1e-6, 1 - 1e-6)
                r_polished = compute_exact_radii(c_polished)
                sum_polished = np.sum(r_polished)
                
                if sum_polished > best_sum:
                    best_sum = sum_polished
                    best_centers = c_polished.copy()
                    best_radii = r_polished.copy()
        except Exception:
            pass

    # If no valid result found, fallback to grid
    if best_centers is None:
        centers = []
        side = int(math.ceil(math.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        best_centers = np.array(centers)
        best_radii = compute_exact_radii(best_centers)

    # Build result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i][0], best_centers[i][1], best_radii[i]]
    
    return result
# EVOLVE-BLOCK-END
