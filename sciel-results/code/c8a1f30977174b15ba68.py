"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def construct():
    # Strategy:
    # 1. Optimize a 3D lattice (9 parameters) to generate a symmetric initial set of 150 points.
    # 2. Refine the 150 points individually (450 parameters) to lower energy further.
    
    # --- Phase 1: Lattice Optimization ---
    
    def lattice_energy_and_grad(lattice_flat):
        # lattice_flat is shape (9,) representing 3 vectors a, b, c
        a = lattice_flat[0:3]
        b = lattice_flat[3:6]
        c = lattice_flat[6:9]
        
        # Generate a pool of lattice points
        # Indices from -4 to 4 to ensure we have enough points to pick 150
        # 9^3 = 729 points
        indices = np.array(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), np.arange(-4, 5))).T.reshape(-1, 3)
        
        # Compute points: P = i*a + j*b + k*c
        # indices shape (729, 3)
        # a, b, c shape (3,)
        # P = indices[:, 0:1] * a[None, :] + indices[:, 1:2] * b[None, :] + indices[:, 2:3] * c[None, :]
        P = indices[:, 0:1] * a[None, :] + indices[:, 1:2] * b[None, :] + indices[:, 2:3] * c[None, :]
        
        # Compute norms
        norms = np.linalg.norm(P, axis=1)
        
        # Avoid zero norm (origin)
        # If origin is included, its norm is 0. We should exclude it or handle it.
        # The 150 closest points to origin will include the origin if it's in the pool.
        # But points must be non-zero vectors.
        # Let's mask out the origin (0,0,0) from the pool before selection.
        mask_nonzero = np.any(P != 0, axis=1)
        P_valid = P[mask_nonzero]
        norms_valid = norms[mask_nonzero]
        
        if len(P_valid) < N:
            # Fallback: if we don't have enough points, return a large energy
            return 1e10, np.zeros(9)
            
        # Select the N closest points
        # argsort returns indices
        sort_idx = np.argsort(norms_valid)
        P_selected = P_valid[sort_idx[:N]]
        norms_selected = norms_valid[sort_idx[:N]]
        
        # Normalize to unit sphere
        X = P_selected / norms_selected[:, np.newaxis]
        
        # Calculate Energy
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = np.where(mask, dist2, 1.0)
        
        # Avoid log(0) if points are identical (shouldn't happen with distinct lattice points)
        dist2_safe = np.maximum(dist2_masked, 1e-12)
        
        e_log = -0.25 * np.sum(np.log(dist2_safe[mask]))
        
        # Gradient with respect to lattice vectors
        # This is complex because the selection of points (sort_idx) is discrete.
        # We use a finite difference approximation for the gradient of the selection step?
        # Or we assume the set of points is fixed locally and compute gradient w.r.t P_selected.
        # dE/dP_selected can be computed. Then dP_selected/dLattice.
        # However, P_selected are specific rows of P.
        # Let's compute gradient w.r.t P_selected first.
        
        # dE/dX_k = sum_{j!=k} (X_k - X_j) / dist2_kj * (-1) ?
        # E = -0.25 sum log(dist2).
        # dE/dX_k = -0.25 * sum_{j!=k} (1/dist2_kj) * 2(X_k - X_j)
        # dE/dX_k = -0.5 * sum_{j!=k} (X_k - X_j) / dist2_kj
        
        grad_X = np.zeros_like(X)
        # Reuse diff and dist2_safe
        # grad_X[k] = -0.5 * sum_j (X_k - X_j) / dist2_kj
        # Note: dist2_safe has 1.0 on diagonal, but we mask it out in sum.
        # We can use the full matrix and mask.
        term = diff / dist2_safe[:, :, np.newaxis]
        term[~mask] = 0 # Zero out diagonal and off-diagonal? No, mask is True for off-diagonal.
        # mask is True for i!=j. So ~mask is diagonal.
        # We want sum over j where i!=j.
        # So we sum term over axis 1.
        grad_X = -0.5 * np.sum(term, axis=1)
        
        # Now, X = P_selected / norms_selected[:, np.newaxis]
        # dX_k / dP_k = (1/norm_k) * I - (X_k/norm_k) * X_k^T = (1/norm_k) * (I - X_k X_k^T)
        # dE/dP_selected = grad_X^T dX/dP ?
        # Actually, dE = sum_k grad_X_k . dX_k
        # dX_k = (1/n_k) (I - X_k X_k^T) dP_k
        # dE/dP_k = (1/n_k) (I - X_k X_k^T) grad_X_k
        # Note: grad_X_k is already a vector.
        
        dE_dP = np.zeros_like(P_selected)
        for k in range(N):
            proj = np.eye(3) - np.outer(X[k], X[k])
            dE_dP[k] = (1.0 / norms_selected[k]) * (proj @ grad_X[k])
            
        # Now map dE_dP back to lattice vectors a, b, c
        # P_selected[m] = i_m a + j_m b + k_m c
        # dE/da = sum_m dE_dP[m] * i_m
        # dE/db = sum_m dE_dP[m] * j_m
        # dE/dc = sum_m dE_dP[m] * k_m
        
        # We need the indices for the selected points
        # sort_idx were indices into P_valid.
        # We need indices into the original 'indices' array (before masking nonzero).
        # This is tricky because mask_nonzero removed some points.
        # Let's reconstruct the mapping.
        
        # Alternative: Compute gradient numerically for the lattice part?
        # 9 parameters, numerical gradient is 9 * 2 evaluations.
        # Each evaluation is fast (vectorized).
        # This is safer and simpler to implement correctly given the discrete selection.
        
        return e_log, None # Return None for jac to use numerical differentiation?
        # Or implement finite difference in the minimize call?
        # L-BFGS-B with jac=True expects analytic. If we return None, it fails?
        # We can set jac=False and let scipy use numerical.
        # But numerical gradient of a function with discrete jumps can be noisy.
        # However, for a 9-dim space, it's manageable.
        
    # Let's use numerical gradient for the lattice phase to avoid complexity of chain rule through selection.
    def lattice_energy(lattice_flat):
        a = lattice_flat[0:3]
        b = lattice_flat[3:6]
        c = lattice_flat[6:9]
        
        indices = np.array(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), np.arange(-4, 5))).T.reshape(-1, 3)
        P = indices[:, 0:1] * a[None, :] + indices[:, 1:2] * b[None, :] + indices[:, 2:3] * c[None, :]
        norms = np.linalg.norm(P, axis=1)
        mask_nonzero = np.any(P != 0, axis=1)
        P_valid = P[mask_nonzero]
        norms_valid = norms[mask_nonzero]
        
        if len(P_valid) < N:
            return 1e10
            
        sort_idx = np.argsort(norms_valid)
        P_selected = P_valid[sort_idx[:N]]
        norms_selected = norms_valid[sort_idx[:N]]
        
        X = P_selected / norms_selected[:, np.newaxis]
        
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = np.where(mask, dist2, 1.0)
        dist2_safe = np.maximum(dist2_masked, 1e-12)
        
        e_log = -0.25 * np.sum(np.log(dist2_safe[mask]))
        return e_log

    # Initial lattice: Simple Cubic
    # Scale factor to ensure points are not too close?
    # The energy depends on relative distances, so scale doesn't matter for the normalized points.
    # However, the *selection* of the 150 closest points depends on the shape of the lattice.
    # A cubic lattice selects a cubic cluster.
    # Let's start with identity.
    x0_lattice = np.eye(3).flatten()
    
    res_lattice = minimize(
        lattice_energy, 
        x0_lattice, 
        method='L-BFGS-B', 
        jac=False, # Use numerical
        options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    # Get the best lattice
    best_lattice = res_lattice.x
    a = best_lattice[0:3]
    b = best_lattice[3:6]
    c = best_lattice[6:9]
    
    # Generate initial points for Phase 2
    indices = np.array(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), np.arange(-4, 5))).T.reshape(-1, 3)
    P = indices[:, 0:1] * a[None, :] + indices[:, 1:2] * b[None, :] + indices[:, 2:3] * c[None, :]
    norms = np.linalg.norm(P, axis=1)
    mask_nonzero = np.any(P != 0, axis=1)
    P_valid = P[mask_nonzero]
    norms_valid = norms[mask_nonzero]
    
    sort_idx = np.argsort(norms_valid)
    P_selected = P_valid[sort_idx[:N]]
    norms_selected = norms_valid[sort_idx[:N]]
    x0 = P_selected / norms_selected[:, np.newaxis]
    
    # Add tiny noise to break symmetry for Phase 2
    np.random.seed(42)
    x0 = x0 + 0.001 * np.random.randn(N, 3)
    norms = np.linalg.norm(x0, axis=1, keepdims=True)
    x0 = x0 / norms

    # --- Phase 2: Individual Refinement ---
    
    lambda_pen = 10000.0
    
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        mask = ~np.eye(N, dtype=bool)
        dist2_masked = np.where(mask, dist2, 1.0)
        
        log_dist2 = np.log(dist2_masked)
        e_log = -0.25 * np.sum(log_dist2[mask])
        
        norms2 = np.sum(x ** 2, axis=1)
        pen = lambda_pen * np.sum((norms2 - 1.0) ** 2)
        
        total_energy = e_log + pen
        
        dist2_safe = np.where(mask, dist2, 1.0)
        grad_log = diff / dist2_safe[:, :, np.newaxis]
        grad_log = np.sum(grad_log, axis=1)
        grad_e = -0.5 * grad_log
        
        grad_pen = 4.0 * lambda_pen * (norms2 - 1.0)[:, np.newaxis] * x
        
        total_grad = grad_e + grad_pen
        
        return total_energy, total_grad.flatten()

    res = minimize(
        energy_and_grad, 
        x0.flatten(), 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    x_opt = res.x.reshape(N, 3)
    
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x_final = x_opt / norms
    
    return x_final

# EVOLVE-BLOCK-END
