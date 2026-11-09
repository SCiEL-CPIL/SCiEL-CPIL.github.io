# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the ratio D / M where:
    D = H_{-1} (entropy of distribution of x - y)
    M = max(H_0, H_1, H_2) (max entropy of x, x+y, x+2y)
    
    P is a 10x10 array P[y, x] for x, y in 0..9.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds budget, keep safety margin
    
    n = 10
    # Grid indices
    # P[y, x] -> row y, col x
    ys = np.arange(n)
    xs = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')  # Y[y, x] = y, X[y, x] = x
    
    # Projection values
    D_vals = X - Y        # x - y
    H0_vals = X           # x
    H1_vals = X + Y       # x + y
    H2_vals = X + 2 * Y   # x + 2y
    
    # Flatten for bincount
    D_flat = D_vals.flatten()
    H0_flat = H0_vals.flatten()
    H1_flat = H1_vals.flatten()
    H2_flat = H2_vals.flatten()
    
    # Unique values and inverse indices
    D_uniq, D_inv = np.unique(D_flat, return_inverse=True)
    H0_uniq, H0_inv = np.unique(H0_flat, return_inverse=True)
    H1_uniq, H1_inv = np.unique(H1_flat, return_inverse=True)
    H2_uniq, H2_inv = np.unique(H2_flat, return_inverse=True)
    
    D_inv_flat = D_inv.flatten()
    H0_inv_flat = H0_inv.flatten()
    H1_inv_flat = H1_inv.flatten()
    H2_inv_flat = H2_inv.flatten()
    
    # Precompute one-hot matrices for gradient computation
    # Each row corresponds to a cell, each column to a bin
    D_onehot = np.zeros((n*n, len(D_uniq)))
    D_onehot[np.arange(n*n), D_inv_flat] = 1.0
    
    H0_onehot = np.zeros((n*n, len(H0_uniq)))
    H0_onehot[np.arange(n*n), H0_inv_flat] = 1.0
    
    H1_onehot = np.zeros((n*n, len(H1_uniq)))
    H1_onehot[np.arange(n*n), H1_inv_flat] = 1.0
    
    H2_onehot = np.zeros((n*n, len(H2_uniq)))
    H2_onehot[np.arange(n*n), H2_inv_flat] = 1.0
    
    def compute_entropies(P_flat):
        """Compute D and M from a flat probability vector."""
        # D: entropy of x - y
        D_probs = D_onehot.T @ P_flat
        D_probs = D_probs[D_probs > 1e-15]
        H_D = -np.sum(D_probs * np.log2(D_probs)) if len(D_probs) > 0 else 0.0
        
        # H0: entropy of x
        H0_probs = H0_onehot.T @ P_flat
        H0_probs = H0_probs[H0_probs > 1e-15]
        H_0 = -np.sum(H0_probs * np.log2(H0_probs)) if len(H0_probs) > 0 else 0.0
        
        # H1: entropy of x + y
        H1_probs = H1_onehot.T @ P_flat
        H1_probs = H1_probs[H1_probs > 1e-15]
        H_1 = -np.sum(H1_probs * np.log2(H1_probs)) if len(H1_probs) > 0 else 0.0
        
        # H2: entropy of x + 2y
        H2_probs = H2_onehot.T @ P_flat
        H2_probs = H2_probs[H2_probs > 1e-15]
        H_2 = -np.sum(H2_probs * np.log2(H2_probs)) if len(H2_probs) > 0 else 0.0
        
        M = max(H_0, H_1, H_2)
        return H_D, M
    
    def objective_and_grad(params):
        """Objective and gradient via analytical computation."""
        # Shift to prevent overflow in exp
        params_shifted = params - np.max(params)
        P = np.exp(params_shifted)
        sum_P = np.sum(P)
        if sum_P <= 0:
            return 1e10, np.zeros_like(params)
        P /= sum_P
        P_flat = P.flatten()
        
        H_D, M = compute_entropies(P_flat)
        
        # Penalize invalid regions where M or D is too small
        if H_D < 1e-6 or M < 1e-6:
            return 1e10, np.zeros_like(params)
        
        # Compute gradient of log(M) - log(D)
        # d/dp_i [ -sum_j p_j log2(p_j) ] = -log2(p_i) - 1/ln(2) for bins with p_i > 0
        
        def entropy_grad(probs, onehot):
            """Gradient of entropy wrt P_flat given probs = onehot.T @ P_flat"""
            grad = np.zeros(len(P_flat))
            mask = probs > 1e-15
            if np.any(mask):
                # dH/dp_i = -log2(probs_bin) - 1/ln(2) for each cell i in that bin
                # = -(log2(probs_bin) + 1/ln(2))
                coeff = -(np.log2(probs[mask]) + 1.0/np.log(2.0))
                # For each bin b with prob > 0, add coeff[b] to all cells in that bin
                for b in np.where(mask)[0]:
                    cells_in_bin = np.where(onehot[:, b] > 0.5)[0]
                    grad[cells_in_bin] += coeff[b]
            return grad
        
        # Gradient of D (H_{-1})
        D_probs = D_onehot.T @ P_flat
        D_mask = D_probs > 1e-15
        D_coeff = np.zeros(len(D_uniq))
        if np.any(D_mask):
            D_coeff[D_mask] = -(np.log2(D_probs[D_mask]) + 1.0/np.log(2.0))
        grad_D = D_onehot @ D_coeff
        
        # Gradient of M = max(H_0, H_1, H_2)
        # Only the argmax contributes to the gradient
        H0_probs = H0_onehot.T @ P_flat
        H1_probs = H1_onehot.T @ P_flat
        H2_probs = H2_onehot.T @ P_flat
        
        H0_mask = H0_probs > 1e-15
        H1_mask = H1_probs > 1e-15
        H2_mask = H2_probs > 1e-15
        
        H0_coeff = np.zeros(len(H0_uniq))
        if np.any(H0_mask):
            H0_coeff[H0_mask] = -(np.log2(H0_probs[H0_mask]) + 1.0/np.log(2.0))
        grad_H0 = H0_onehot @ H0_coeff
        
        H1_coeff = np.zeros(len(H1_uniq))
        if np.any(H1_mask):
            H1_coeff[H1_mask] = -(np.log2(H1_probs[H1_mask]) + 1.0/np.log(2.0))
        grad_H1 = H1_onehot @ H1_coeff
        
        H2_coeff = np.zeros(len(H2_uniq))
        if np.any(H2_mask):
            H2_coeff[H2_mask] = -(np.log2(H2_probs[H2_mask]) + 1.0/np.log(2.0))
        grad_H2 = H2_onehot @ H2_coeff
        
        # Find which entropy is the max
        H_0 = -np.sum(H0_probs[H0_mask] * np.log2(H0_probs[H0_mask])) if np.any(H0_mask) else 0.0
        H_1 = -np.sum(H1_probs[H1_mask] * np.log2(H1_probs[H1_mask])) if np.any(H1_mask) else 0.0
        H_2 = -np.sum(H2_probs[H2_mask] * np.log2(H2_probs[H2_mask])) if np.any(H2_mask) else 0.0
        
        if H_0 >= H_1 and H_0 >= H_2:
            grad_M = grad_H0
        elif H_1 >= H_0 and H_1 >= H_2:
            grad_M = grad_H1
        else:
            grad_M = grad_H2
        
        # d/dp_i [log(M) - log(D)] = grad_M / M - grad_D / D
        # But we need to account for the softmax normalization
        # Actually, let's work in the log-space directly
        
        # Objective: log(M) - log(D)
        # Gradient wrt params (log P): 
        # dP_i/dparam_j = P_i * (delta_ij - sum_k P_k)
        # So dObj/dparam_j = sum_i dObj/dP_i * P_i * (delta_ij - 1)
        # = P_j * (dObj/dP_j - sum_i dObj/dP_i * P_i)
        
        dObj_dP = grad_M / M - grad_D / H_D
        
        # Sum term
        sum_term = np.sum(dObj_dP * P_flat)
        
        grad_params = P_flat * (dObj_dP - sum_term)
        
        return np.log(M) - np.log(H_D), grad_params
    
    def objective(params):
        """Minimize negative log-ratio: -log(D/M) = log(M) - log(D)"""
        val, _ = objective_and_grad(params)
        return val
    
    # Start from uniform distribution
    P_init = np.ones((n, n), dtype=np.float64) / (n * n)
    P_init_flat = P_init.flatten()
    params_init = np.log(P_init_flat)
    
    best_P = P_init.copy()
    best_val = objective(params_init)
    
    # Try multiple optimization approaches with diverse initializations
    initializations = [
        params_init,  # uniform
    ]
    
    # Add diagonal band initializations
    for shift in range(-5, 6, 2):
        P_diag = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x - y == shift or (x - y - shift) % (2*n) == 0:
                    P_diag[y, x] = 1.0
        if np.sum(P_diag) > 0:
            P_diag /= np.sum(P_diag)
            initializations.append(np.log(P_diag.flatten()))
    
    # Add 2D Gaussian initializations
    for cx in [2, 5, 8]:
        for cy in [2, 5, 8]:
            for sigma in [1.0, 2.0]:
                Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
                P_gauss = np.exp(-((Xg - cx)**2 + (Yg - cy)**2) / (2 * sigma**2))
                if np.sum(P_gauss) > 0:
                    P_gauss /= np.sum(P_gauss)
                    initializations.append(np.log(P_gauss.flatten()))
    
    # Add random initializations
    np.random.seed(42)
    for _ in range(10):
        P_rand = np.random.dirichlet(np.ones(n * n))
        initializations.append(np.log(P_rand))
    
    # Optimize from each initialization
    for init_idx, init_params in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        # Try L-BFGS-B with correct gradient
        try:
            result = opt.minimize(
                objective_and_grad,
                init_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                P_opt = np.exp(result.x)
                P_opt /= np.sum(P_opt)
                val_opt, _ = objective_and_grad(result.x)
                if val_opt < best_val:
                    best_val = val_opt
                    best_P = P_opt.reshape(n, n)
        except Exception:
            pass
        
        # Try SLSQP
        try:
            result = opt.minimize(
                objective,
                init_params,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result.fun < best_val:
                P_opt = np.exp(result.x)
                P_opt /= np.sum(P_opt)
                val_opt, _ = objective_and_grad(result.x)
                if val_opt < best_val:
                    best_val = val_opt
                    best_P = P_opt.reshape(n, n)
        except Exception:
            pass
        
        # Try Nelder-Mead (no gradient needed)
        try:
            result = opt.minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_val:
                P_opt = np.exp(result.x)
                P_opt /= np.sum(P_opt)
                val_opt, _ = objective_and_grad(result.x)
                if val_opt < best_val:
                    best_val = val_opt
                    best_P = P_opt.reshape(n, n)
        except Exception:
            pass
    
    # Final polish with L-BFGS-B from best found
    try:
        best_params = np.log(best_P.flatten())
        result = opt.minimize(
            objective_and_grad,
            best_params,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result.fun < best_val:
            P_opt = np.exp(result.x)
            P_opt /= np.sum(P_opt)
            val_opt, _ = objective_and_grad(result.x)
            if val_opt < best_val:
                best_val = val_opt
                best_P = P_opt.reshape(n, n)
    except Exception:
        pass
    
    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total > 1e-9:
        best_P /= total
    else:
        best_P = np.ones((n, n)) / (n * n)
    
    return best_P


# EVOLVE-BLOCK-END
