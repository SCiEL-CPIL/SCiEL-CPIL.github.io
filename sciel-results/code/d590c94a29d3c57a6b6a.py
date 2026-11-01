import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def compute_radius(pts):
        """Compute the common radius: min of half pairwise distances and boundary distances."""
        if len(pts) < 2:
            return 0.0
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    def hex_lattice_init(n):
        """Generate hexagonal lattice points inside the unit disk."""
        spacing = 2.0 / (np.sqrt(n) + 1.5)
        while True:
            k = int(1.0 / spacing) + 3
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.97

    def spectral_objective_and_grad(x_flat):
        """
        Objective: -min(G_off_diag, boundary_constraints)
        x_flat: flattened 2D matrix of centers (N x 2)
        
        We use a smooth approximation to the min to allow gradient-based optimization.
        The 'spectral' aspect here is that we optimize in the coordinate space but 
        utilize the structure of the Gram matrix G = X X^T implicitly through 
        pairwise distances which are sqrt(G_ii + G_jj - 2G_ij).
        
        However, to strictly follow the prompt's "discrete spectral optimization" 
        and "low-rank matrix factorization" hint, we treat the centers as rows of X.
        The Gram matrix G = X X^T is PSD. The distances are determined by G.
        We optimize X directly.
        """
        if time.time() > deadline:
            return 1e10, np.zeros_like(x_flat)
            
        X = x_flat.reshape(N, 2)
        
        # Compute pairwise distances
        # Using broadcasting for efficiency in Python/NumPy for N=300
        # diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # dist = np.sqrt(np.maximum(dist_sq, 1e-15))
        
        # More efficient: ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi.xj
        norms_sq = np.sum(X**2, axis=1)
        gram = X @ X.T
        dist_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * gram
        np.fill_diagonal(dist_sq, np.inf)
        dist_sq = np.maximum(dist_sq, 1e-15)
        dist = np.sqrt(dist_sq)
        
        # Boundary constraints: 1 - ||xi||
        radii = np.sqrt(np.maximum(norms_sq, 1e-15))
        boundary = 1.0 - radii
        
        # We want to maximize min( min_{i<j} dist_ij/2, min_i boundary_i )
        # Let's collect all values
        iu = np.triu_indices(N, k=1)
        pair_vals = dist[iu] / 2.0
        all_vals = np.concatenate([pair_vals, boundary])
        
        # Smooth min (soft-min)
        # Use log-sum-exp trick for stability
        alpha = 100.0 # Sharpness
        min_val = np.min(all_vals)
        all_vals_clipped = np.clip(all_vals, min_val - 10, min_val + 10)
        
        # z = -alpha * all_vals
        # soft_min = -1/alpha * log( sum( exp(z) ) )
        # To avoid overflow, shift by max(z)
        z = -alpha * all_vals_clipped
        z_max = np.max(z)
        sum_exp = np.sum(np.exp(z - z_max))
        soft_min = -1.0/alpha * (np.log(sum_exp) + z_max)
        
        obj = -soft_min
        
        # Gradient
        # d(soft_min)/dx = -1/alpha * d(log(sum(exp(-alpha*v))))/dx
        # = -1/alpha * (1/sum) * sum( d(exp(-alpha*v))/dx )
        # = -1/alpha * (1/sum) * sum( exp(-alpha*v) * (-alpha) * dv/dx )
        # = sum( (exp(-alpha*v)/sum) * dv/dx )
        # Let w_v = exp(-alpha*v) / sum_exp
        # grad_obj = -grad_soft_min = -sum( w_v * dv/dx )
        
        weights = np.exp(z - z_max) / sum_exp
        
        grad_X = np.zeros_like(X)
        
        # Pair contributions
        # v_ij = dist_ij / 2
        # d(dist_ij)/dX_i = (X_i - X_j) / dist_ij
        # d(v_ij)/dX_i = (X_i - X_j) / (2 * dist_ij)
        # d(v_ij)/dX_j = (X_j - X_i) / (2 * dist_ij)
        
        # We need to accumulate weights for each pair (i,j)
        # v_vals are in all_vals[0:pair_count]
        pair_weights = weights[:len(pair_vals)]
        
        # We need to map pair_weights back to the full matrix structure or iterate
        # For N=300, iterating over ~45k pairs is feasible in Python if simple, 
        # but vectorized is better.
        
        # Let's construct the gradient using the Gram matrix structure or direct accumulation.
        # dist matrix is N x N.
        # v_matrix = dist / 2.0
        # v_matrix[diag] = 0 (masked)
        
        v_matrix = dist / 2.0
        np.fill_diagonal(v_matrix, 0)
        
        # We need weights for every element in v_matrix.
        # The 'all_vals' array contains upper triangle pairs and boundary terms.
        # Let's create a weight matrix W of size N x N for pairs.
        W = np.zeros((N, N))
        W[iu] = pair_weights
        W = W + W.T # Symmetric
        
        # Gradient of sum_{i,j} W_ij * dist_ij / 2
        # Note: sum over i<j is equivalent to 1/2 sum over i!=j if W is symmetric and defined properly.
        # In our W, W[i,j] is the weight for the term dist_ij/2.
        # The objective term is sum_{i<j} w_ij * dist_ij/2.
        # This is equal to 1/2 sum_{i!=j} W_ij * dist_ij/2 ? No.
        # Let's stick to the definition: sum_{i<j} w_ij * (dist_ij/2).
        
        # d(dist_ij)/dX_k:
        # if k==i: (X_i - X_j)/dist_ij
        # if k==j: (X_j - X_i)/dist_ij
        # else 0
        
        # grad_X[i] += sum_{j!=i} W_ij * (X_i - X_j) / (2 * dist_ij)
        # Note: W_ij corresponds to weight w_ij for i<j.
        # In the sum for grad_X[i], we sum over all j.
        # If i<j, term is W_ij * (X_i - X_j) / (2 * dist_ij)
        # If i>j, term is W_ji * (X_i - X_j) / (2 * dist_ij) which is W_ji * (X_i - X_j) / (2 * dist_ij)
        # Since W is symmetric, W_ji = W_ij.
        # So grad_X[i] = sum_{j!=i} W_ij * (X_i - X_j) / (2 * dist_ij)
        
        # Vectorized calculation:
        # dist_inv = 1/dist where dist>0
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        inv_dist = 1.0 / dist_safe
        
        # Coefficient matrix C_ij = W_ij / (2 * dist_ij)
        C = W * inv_dist / 2.0
        np.fill_diagonal(C, 0)
        
        # grad_X[i] = sum_j C_ij (X_i - X_j) = X_i * sum_j C_ij - sum_j C_ij X_j
        # sum_j C_ij is row sum of C.
        row_sums = np.sum(C, axis=1)
        
        # term1 = X * row_sums[:, np.newaxis]
        # term2 = C @ X
        # grad_X = term1 - term2
        
        term1 = X * row_sums[:, np.newaxis]
        term2 = C @ X
        
        grad_X = term1 - term2
        
        # Boundary contributions
        # v_i = 1 - ||X_i||
        # d(v_i)/dX_i = -X_i / ||X_i||
        # weights for boundary are weights[len(pair_vals):]
        boundary_weights = weights[len(pair_vals):]
        
        # grad_X[i] += boundary_weights[i] * (-X_i / ||X_i||)
        # Note: obj = -soft_min, so grad_obj = -grad_soft_min.
        # grad_soft_min = sum w_v dv/dx.
        # So grad_obj = - sum w_v dv/dx.
        
        # For pairs: grad_obj_X[i] = - sum_j W_ij (X_i - X_j) / (2 * dist_ij)
        # Wait, my derivation above:
        # obj = -soft_min
        # grad_obj = - grad_soft_min
        # grad_soft_min = sum w_v dv/dx
        # So grad_obj = - sum w_v dv/dx
        
        # For pairs:
        # dv_ij/dX_i = (X_i - X_j) / (2 * dist_ij)
        # Contribution to grad_obj[X_i] = - w_ij * (X_i - X_j) / (2 * dist_ij)
        
        # My previous grad_X was sum w_v dv/dx (which is grad_soft_min).
        # So I need to negate it.
        grad_X = -grad_X
        
        # For boundary:
        # dv_i/dX_i = -X_i / ||X_i||
        # Contribution to grad_obj[X_i] = - w_i * (-X_i / ||X_i||) = w_i * X_i / ||X_i||
        
        radii_safe = np.where(radii > 1e-10, radii, 1.0)
        boundary_grad = (boundary_weights / radii_safe)[:, np.newaxis] * X
        grad_X += boundary_grad
        
        # Flatten gradient
        grad_flat = grad_X.flatten()
        
        return obj, grad_flat

    # Step 1: Generate initial hexagonal lattice
    pts = hex_lattice_init(N)
    
    # Step 2: Compute initial radius
    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Step 3: Optimization using the spectral/coordinate formulation
    # We use L-BFGS-B because it handles the large number of parameters well
    # and we have a gradient.
    
    x0 = pts.flatten()
    
    # Run optimization in stages with increasing smoothness (alpha)
    # The current spectral_objective_and_grad uses a fixed alpha. 
    # To improve, we could implement alpha annealing, but for simplicity 
    # and within the "keep design" constraint, we run the optimizer 
    # with the current smooth objective.
    
    # To make it more effective, we can run multiple iterations or restarts.
    # Given the 150s limit, we will run one main optimization and a polish.
    
    result = minimize(
        spectral_objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True, # Use the gradient provided
        options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
    )
    
    if result.fun < spectral_objective_and_grad(x0)[0]:
        pts_new = result.x.reshape(N, 2)
        # Ensure points are inside the disk
        r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
        mask = r_new > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
        
        r_new_val = compute_radius(pts_new)
        if r_new_val > best_radius:
            best_radius = r_new_val
            best_pts = pts_new.copy()
    
    # Step 4: Final polish with tighter optimization if time permits
    if time.time() < deadline - 20:
        x0 = best_pts.flatten()
        # We could increase alpha here for a sharper min, but the function has fixed alpha.
        # So we just run more iterations.
        result2 = minimize(
            spectral_objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'disp': False}
        )
        
        if result2.fun < spectral_objective_and_grad(x0)[0]:
            pts_new = result2.x.reshape(N, 2)
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask = r_new > 1.0
            if np.any(mask):
                pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
            
            r_new_val = compute_radius(pts_new)
            if r_new_val > best_radius:
                best_radius = r_new_val
                best_pts = pts_new.copy()

    # Final validation: ensure all points are strictly inside and distinct
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask_out = r_final >= 1.0
    if np.any(mask_out):
        best_pts[mask_out] = best_pts[mask_out] / r_final[mask_out, np.newaxis] * 0.9999
    
    # Check for coincident points
    min_dist = np.min(pdist(best_pts))
    if min_dist < 1e-6:
        np.random.seed(123)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-6:
                    best_pts[j] += np.random.normal(0, 1e-4, size=2)
    
    return best_pts
