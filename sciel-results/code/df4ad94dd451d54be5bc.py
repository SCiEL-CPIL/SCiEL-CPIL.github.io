"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial constraint
        radial_dists = np.linalg.norm(pts, axis=1)
        if np.any(radial_dists > 1.0):
            return -1.0
        min_radial = 1.0 - np.max(radial_dists)
        
        # Pairwise constraint
        # Vectorized distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        min_pair = np.min(pair_dists) / 2.0
        
        return min(min_pair, min_radial)
    
    def softmin_objective(params, tau=0.005):
        """
        Smooth objective function: -softmin of all constraints.
        Constraints: d_ij / 2 for i<j, 1 - |c_i| for all i.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        # Softmin
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        # Avoid numerical issues if tau is very small
        exp_terms = np.exp(-shifted / tau)
        softmin_val = min_val - tau * np.log(np.sum(exp_terms))
        
        return -softmin_val

    def softmin_gradient(params, tau=0.005):
        """
        Gradient of the softmin objective.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0) # Avoid division by zero
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        # Weights for softmin
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-shifted / tau)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Gradient of softmin w.r.t values is -weights (since obj = -softmin, grad_obj = -grad_softmin = weights? No.)
        # f = -softmin.
        # d(softmin)/dz_k = weights[k].
        # df/dz_k = -weights[k].
        
        grad_z = -weights
        
        grad_pts = np.zeros_like(pts)
        
        # Pairwise contributions
        i_idx, j_idx = iu
        w_pair = grad_z[:len(pair_dists)]
        
        # Vector for each pair: pts[i] - pts[j]
        vec = pts[i_idx] - pts[j_idx]
        norms = dists[iu]
        # Unit vectors
        unit_vec = vec / norms[:, np.newaxis]
        
        # d(pair_dist/2)/d(pts[i]) = 0.5 * unit_vec
        # d(pair_dist/2)/d(pts[j]) = -0.5 * unit_vec
        
        grad_i = 0.5 * unit_vec * w_pair[:, np.newaxis]
        grad_j = -0.5 * unit_vec * w_pair[:, np.newaxis]
        
        np.add.at(grad_pts, i_idx, grad_i)
        np.add.at(grad_pts, j_idx, grad_j)
        
        # Radial contributions
        w_radial = grad_z[len(pair_dists):]
        norms_rad = np.linalg.norm(pts, axis=1)
        # Handle zero norm
        norms_rad_safe = np.where(norms_rad > 1e-10, norms_rad, 1.0)
        unit_rad = pts / norms_rad_safe[:, np.newaxis]
        # d(radial)/d(pts[i]) = -unit_rad
        grad_pts += -w_radial[:, np.newaxis] * unit_rad
        
        return grad_pts.flatten()

    def optimize_from_initial(pts, max_iter=200):
        """Run L-BFGS-B optimization from initial points."""
        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(
                softmin_objective, 
                x0, 
                method='L-BFGS-B', 
                jac=softmin_gradient,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                score = compute_score(new_pts)
                if score > 0:
                    return new_pts, score
        except Exception:
            pass
        return None, -1.0

    def generate_fermat_spiral(scale=1.0):
        """Generate points using Fermat spiral (sunflower pattern)."""
        k = np.arange(N)
        r = np.sqrt(k / N) * scale
        theta = k * 2 * np.pi / (1 + np.sqrt(5)) # Golden angle approx
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    def generate_hex_grid(spacing=0.1):
        """Generate points on a hexagonal grid, clipped to unit disk."""
        pts = []
        # Generate grid points
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) < N:
            # If not enough, increase density
            return generate_hex_grid(spacing * 0.9)
        
        # Select N points closest to center? Or random?
        # Closest to center ensures we fill the disk
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        return pts[idx]

    def generate_random():
        """Generate random points in unit disk."""
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    # List of initialization strategies
    inits = []
    
    # 1. Fermat Spiral
    inits.append(generate_fermat_spiral(0.95))
    
    # 2. Hex Grid
    inits.append(generate_hex_grid(0.1))
    
    # 3. Random
    inits.append(generate_random())
    
    # 4. Another Fermat with different scale
    inits.append(generate_fermat_spiral(0.90))
    
    best_pts = None
    best_score = -1.0
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Evaluate initial
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_pts = init_pts
        
        # Optimize
        opt_pts, opt_score = optimize_from_initial(init_pts)
        if opt_pts is not None and opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
            
    # Final polish if time remains
    if best_pts is not None and time.time() < deadline - 5:
        # Try a second pass with smaller tau or more iterations
        # Re-optimize best_pts
        opt_pts, opt_score = optimize_from_initial(best_pts, max_iter=100)
        if opt_pts is not None and opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts

    # Fallback
    if best_pts is None:
        best_pts = np.zeros((N, 2))
        # Simple grid fallback
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            best_pts[i, 0] = (col - n_per_row/2) * spacing
            best_pts[i, 1] = (row - n_per_row/2) * spacing

    return best_pts
# EVOLVE-BLOCK-END
