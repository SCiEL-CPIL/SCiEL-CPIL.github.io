import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly reconstructs matrix multiplication."""
    rank = U.shape[1]
    # T[alpha, beta, gamma] = 1 if consistent, 0 otherwise.
    # We check all (alpha, beta, gamma) or just the consistent ones?
    # The decomposition must be exact for the tensor M_{n,m,p}.
    # M_{n,m,p} has 1s at (i*m+j, j*p+k, k*n+i) and 0s elsewhere.
    # So we must check that the sum is 1 for consistent triples and 0 for others.
    
    # Efficient check:
    # Reshape U, V, W to 3D? No, they are 2D.
    # We can compute the tensor T_recon[alpha, beta, gamma] = sum_r U[alpha,r] V[beta,r] W[gamma,r].
    # This is an Einstein summation: np.einsum('ar,br,cr->abc', U, V, W)
    # Shape: (n*m, m*p, p*n)
    
    try:
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    except Exception:
        return False, 1e10
        
    # Build target tensor
    # Target is 0 everywhere, except at (i*m+j, j*p+k, k*n+i) where it is 1.
    # We don't need to build the full 20x30x24 tensor if we just check the non-zeros and a sample of zeros.
    # But for verification, we need exactness.
    # Let's build the full target tensor for verification if rank is small enough to compute T_recon.
    # 20*30*24 = 14400 floats. This is small.
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1.0
                
    diff = np.abs(T_recon - T_target)
    max_err = np.max(diff)
    return max_err < tol, max_err

def _round_to_half_integer(arr):
    # Round to nearest 0.5
    return np.round(arr * 2) / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Precompute consistent triples for fast loss calculation if needed
    # But np.einsum is fast enough for 20x30x24.
    
    # Strategy: Try to find rank r < 120.
    # We iterate r from 119 down to 50.
    # For each r, we run a few optimization steps.
    
    # To speed up, we can use the schoolbook as a warm start?
    # Schoolbook has 120 columns. If we drop one column, we can try to adjust the rest.
    # Or random init.
    
    # Let's try a "greedy drop" approach first.
    # Start with schoolbook U, V, W (rank 120).
    # Try to zero out one column of U, V, W? No, we need to remove a term.
    # Removing a term means setting the r-th column of U, V, W to 0?
    # No, that changes the sum.
    # We need to find U', V', W' of rank 119 such that sum_{r=1}^{119} ... = T.
    
    # Numerical Optimization:
    # Variables: x = [U.flatten(), V.flatten(), W.flatten()]
    # Loss: sum (T_recon - T_target)^2
    
    def loss_and_grad(x, r):
        if time.time() > deadline:
            return 1e10, np.zeros_like(x)
        
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = T_recon - T_target
        loss = np.sum(diff**2)
        
        # Gradient calculation is complex. Let's use numerical gradient or just minimize without gradient (Nelder-Mead is slow).
        # L-BFGS-B requires gradient.
        # We can approximate gradient or use a simpler optimizer like 'Powell' which doesn't need gradient but is slow.
        # Given the high dimensionality (74*r ~ 8000 variables for r=110), gradient-free is impossible.
        # We need analytic gradient.
        
        # dL/dU[a,r] = 2 * sum_{b,g} (T_recon[a,b,g] - T_target[a,b,g]) * V[b,r] * W[g,r]
        # dL/dV[b,r] = 2 * sum_{a,g} (T_recon[a,b,g] - T_target[a,b,g]) * U[a,r] * W[g,r]
        # dL/dW[g,r] = 2 * sum_{a,b} (T_recon[a,b,g] - T_target[a,b,g]) * U[a,r] * V[b,r]
        
        # This is efficient with einsum.
        err = diff # shape (n*m, m*p, p*n)
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
        grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
        grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    # Pre-compute T_target
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    # Try ranks
    for r in range(119, 50, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random seeds
        for seed in range(3):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            # Initialize
            # Scale init: 1/sqrt(r)
            init_scale = 1.0 / np.sqrt(r)
            x0 = np.random.randn((n*m + m*p + p*n) * r) * init_scale
            
            # Run optimizer
            try:
                res = minimize(
                    loss_and_grad, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < 1e-6:
                    # Extract U, V, W
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Round to half-integers
                    U_r = _round_to_half_integer(U)
                    V_r = _round_to_half_integer(V)
                    W_r = _round_to_half_integer(W)
                    
                    # Verify
                    is_valid, err = _verify_decomposition(U_r, V_r, W_r, n, m, p)
                    if is_valid:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # If rounding failed, try without rounding (if evaluator accepts floats? 
                    # "integer/half-integer entries" is a constraint. So we MUST round.)
                    
            except Exception as e:
                pass
                
    # If no better rank found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
