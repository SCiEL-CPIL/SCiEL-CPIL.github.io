"""Tensor decomposition for 3x3 @ 3x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under 150s wall limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    def build_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = build_schoolbook()
    best_rank = rank_schoolbook

    # Precompute the target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if the (i,j) element of A and (j,k) element of B 
    # contribute to (i,k) element of C. 
    # In the standard matrix multiplication tensor T_{i,j,k}, 
    # the non-zero entries are at (i, j, k) for all i,j,k.
    # Wait, the tensor T is defined such that 
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 
    # if the indices correspond to a valid multiplication step, else 0.
    # The valid steps are exactly (i, j, k) for i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # So T is just an all-ones tensor of size (n, m, p) in the "logical" view,
    # but in the "flattened" view of U, V, W it's a sparse tensor.
    
    # Let's define the target in the flattened space for verification and objective.
    # Size of U: n*m = 9
    # Size of V: m*p = 15
    # Size of W: p*n = 15
    
    # The objective function works on flattened parameters.
    # We need to map the flattened U, V, W to the tensor product.
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Target tensor in flattened form is not straightforward to store as a dense array 
    # of size (9, 15, 15) = 2025 elements, which is small.
    # T_flat[a, b, c] = 1 if (a,b,c) corresponds to (i,j,k) 
    # where a = i*m+j, b = j*p+k, c = k*n+i.
    # Note: b depends on j, c depends on k.
    # So for a fixed (i,j,k):
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T_flat[a, b, c] = 1.
    # All other entries are 0.
    
    T_target = np.zeros((size_u, size_v, size_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def verify_decomposition(U, V, W, R):
        """Check if U, V, W form a valid decomposition."""
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Using einsum for speed
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # Check against T_target
        return np.allclose(T_recon, T_target, atol=1e-9, rtol=1e-9)

    def error_func(params, R):
        """Calculate sum of squared errors."""
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = T_recon - T_target
        return np.sum(diff**2)

    def error_grad(params, R):
        """Calculate error and gradient."""
        U = params[:size_u*R].reshape(size_u, R)
        V = params[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
        W = params[size_u*R + size_v*R :].reshape(size_w, R)
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = T_recon - T_target
        
        # Gradient of E = sum diff^2 w.r.t U, V, W
        # dE/dU[a, r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        # This can be computed as:
        # grad_U = 2 * einsum('abc,br,cr->ar', diff, V, W)
        # grad_V = 2 * einsum('abc,ar,cr->br', diff, U, W)
        # grad_W = 2 * einsum('abc,ar,br->cr', diff, U, V)
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
        grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
        grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return np.sum(diff**2), grad

    def try_rank(R, max_restarts=20):
        """Try to find a decomposition of rank R."""
        total_size = (size_u + size_v + size_w) * R
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            # Scaling: entries in U, V, W are typically small or around 1.
            # For rank 1 terms, if U, V, W are O(1), the sum is O(R).
            # We want the sum to be 1. So maybe scale by 1/sqrt(R)?
            # Or just use standard normal and let the optimizer handle it.
            params0 = np.random.randn(total_size) * 0.5
            
            try:
                res = minimize(error_func, params0, args=(R,), method='L-BFGS-B', 
                               jac=error_grad,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 2000})
                
                if res.fun < 1e-8:
                    U = res.x[:size_u*R].reshape(size_u, R)
                    V = res.x[size_u*R : size_u*R + size_v*R].reshape(size_v, R)
                    W = res.x[size_u*R + size_v*R :].reshape(size_w, R)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_snap, V_snap, W_snap, R):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                pass
                
        return None

    # Search for lower ranks
    # Known: R(3,3,3) = 23, R(3,3,2) = 13.
    # Upper bound for R(3,3,5) is 36.
    # We search from 36 up to 44.
    for R in range(36, 45):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R, max_restarts=50)
        if result is not None:
            return result

    # Return schoolbook if no better found
    return best_result
# EVOLVE-BLOCK-END
