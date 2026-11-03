import numpy as np
import time

N = 50

def construct():
    # Spectral optimization approach
    # Variable: A (N x N symmetric matrix)
    # Points: X = R @ V, where V are eigenvectors of A, R is random projection
    
    # Fixed random projection matrix R (3 x N)
    # Scale R so that typical norm of R @ v is ~ 0.5
    # E[||R v||^2] = v^T R^T R v. If R entries N(0, sigma^2), E[R^T R] ~ 3 sigma^2 I (approx)
    # So ||X|| ~ sqrt(3) sigma. We want ~0.5.
    # sigma = 0.5 / sqrt(3)
    sigma = 0.5 / np.sqrt(3)
    R = np.random.randn(3, N) * sigma
    
    # Initialize A as random SPD
    # A = B^T B + I
    B = np.random.randn(N, N) * 0.1
    A = B.T @ B + np.eye(N)
    
    def get_points(A):
        # Compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(A)
        # Project to 3D
        X = R @ eigvecs
        return X
    
    def compute_forces(X):
        # X is (N, 3)
        # Returns forces (N, 3) and current radius
        centers = X
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-9)
        d = np.sqrt(d2)
        d3 = d ** 3
        np.fill_diagonal(d3, 1e-9)
        
        force_spheres = np.sum(diff / d3[:, :, np.newaxis], axis=1)
        
        # Wall forces
        force_walls = np.zeros_like(centers)
        K_wall = 100.0
        eps = 1e-6
        for k in range(3):
            c = centers[:, k]
            dist_pos = np.maximum(0.5 - c, eps)
            dist_neg = np.maximum(c + 0.5, eps)
            f_pos = -K_wall / (dist_pos ** 2)
            f_neg = K_wall / (dist_neg ** 2)
            force_walls[:, k] = f_pos + f_neg
            
        total_force = force_spheres + force_walls
        
        # Calculate radius
        d_min = np.min(d[np.triu_indices(N, k=1)])
        w_min = np.min(np.abs(0.5 - np.abs(centers)))
        r_current = min(d_min / 2.0, w_min)
        
        return total_force, r_current

    def compute_grad_A(A, F_X):
        # F_X is (N, 3) forces on points
        # G_V_full = R.T @ F_X (N x N)
        G_V_full = R.T @ F_X
        
        eigvals, eigvecs = np.linalg.eigh(A)
        
        G_A = np.zeros((N, N))
        
        # Vectorized computation of G_A
        # G_A_{pq} = sum_{i != j} (g_i^T v_j) v_{jp} v_{iq} / (lambda_i - lambda_j)
        # Let M_{ij} = (g_i^T v_j) / (lambda_i - lambda_j)
        # G_A = sum_{i != j} M_{ij} v_j v_i^T
        
        # Compute projections
        # proj[i, j] = g_i^T v_j = sum_k G_V_full[k, i] * eigvecs[k, j]
        # G_V_full is N x N (col i is g_i)
        # eigvecs is N x N (col j is v_j)
        # proj = G_V_full.T @ eigvecs  (N x N)
        proj = G_V_full.T @ eigvecs
        
        # Compute denominators
        # diff_lambda[i, j] = lambda_i - lambda_j
        diff_lambda = eigvals[:, np.newaxis] - eigvals[np.newaxis, :]
        
        # Avoid division by zero
        mask = np.abs(diff_lambda) > 1e-12
        # Set diagonal to 0 to avoid i=j terms
        np.fill_diagonal(mask, False)
        
        # M = proj / diff_lambda
        M = np.zeros_like(proj)
        M[mask] = proj[mask] / diff_lambda[mask]
        
        # G_A = sum_j (v_j @ (M.T @ v_i)?) 
        # G_A = sum_{i,j} M_{ij} v_j v_i^T
        # Let V = eigvecs.
        # G_A = V @ (M.T) @ V.T ?
        # (V M.T V.T)_{pq} = sum_{i,j} V_{pi} (M.T)_{ij} V_{qj} = sum_{i,j} V_{pi} M_{ji} V_{qj}
        # Our formula: sum_{i,j} M_{ij} v_j v_i^T -> (pq) component: sum_{i,j} M_{ij} V_{pj} V_{qi}
        # Let's swap i and j in the sum?
        # sum_{i,j} M_{ij} V_{pj} V_{qi}
        # Let k=j, l=i. sum_{k,l} M_{lk} V_{pk} V_{ql}
        # This is (V M.T V.T)_{pq} = sum_{k,l} V_{pk} M.T_{kl} V_{ql} = sum_{k,l} V_{pk} M_{lk} V_{ql}
        # Yes, it matches.
        
        G_A = eigvecs @ M.T @ eigvecs.T
        G_A = (G_A + G_A.T) / 2
        
        return G_A

    start_time = time.time()
    time_limit = 100.0
    
    best_r = 0.0
    best_A = A.copy()
    
    # Optimization loop
    lr = 0.05
    iterations = 5000
    
    for i in range(iterations):
        if time.time() - start_time > time_limit:
            break
            
        X = get_points(A)
        F_X, r_current = compute_forces(X)
        
        if r_current > best_r:
            best_r = r_current
            best_A = A.copy()
            
        # Compute gradient
        G_A = compute_grad_A(A, F_X)
        
        # Gradient Ascent
        norm_G = np.linalg.norm(G_A)
        if norm_G > 1e-9:
            step = lr * G_A / norm_G
            A += step
            A = (A + A.T) / 2 # Enforce symmetry
            
        # Decay learning rate
        lr = 0.05 * (0.001 / 0.05) ** (i / iterations)
        
    # Final polish
    # We can switch to direct coordinate optimization on the best spectral result
    # to fine-tune.
    X_best = get_points(best_A)
    
    # Direct force optimization on X_best
    for i in range(2000):
        if time.time() - start_time > time_limit:
            break
        F_X, r_current = compute_forces(X_best)
        if r_current > best_r:
            best_r = r_current
            
        # Simple gradient ascent on X
        norm_F = np.linalg.norm(F_X)
        if norm_F > 1e-9:
            step = 0.01 * F_X / norm_F
            X_best += step
            X_best = np.clip(X_best, -0.5, 0.5)
            
    return X_best
