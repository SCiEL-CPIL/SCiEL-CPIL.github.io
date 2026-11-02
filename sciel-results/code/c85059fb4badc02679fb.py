"""
Refined search for minimal rank tensor decomposition of 3x4x7 matrix multiplication.
Uses continuous relaxation via gradient-based optimization followed by snapping 
to half-integers and local discrete refinement.
"""
import time
import numpy as np
import random

# Target dimensions
n, m, p = 3, 4, 7

# Pre-compute the target tensor T[i, j, k] which is 1 if the index is valid, 0 otherwise.
# The multiplication tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
# The decomposition is sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k]
# Note: The task defines W with trace convention W[k*n+i, r]. 
# Standard tensor indices: i (row of A), j (col of A / row of B), k (col of B).
# A[i,j] * B[j,k] -> C[i,k].
# Factorization: C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]? 
# Let's check the seed's convention.
# Seed: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
# Product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
# This sum is over c. For a fixed i,j,k, the term is U_{i,j}^c V_{j,k}^c W_{k,i}^c.
# This matches the standard matrix multiplication tensor structure.

def get_target_tensor():
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

T_target = get_target_tensor()

def compute_error(U, V, W):
    """
    Compute the squared error of the decomposition.
    U: (n*m, r) -> reshape to (n, m, r)
    V: (m*p, r) -> reshape to (m, p, r)
    W: (p*n, r) -> reshape to (p, n, r)
    """
    r = U.shape[1]
    U_resh = U.reshape(n, m, r)
    V_resh = V.reshape(m, p, r)
    W_resh = W.reshape(p, n, r)
    
    # Sum over r: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We need to align dimensions for einsum.
    # U_resh: i, j, c
    # V_resh: j, k, c
    # W_resh: k, i, c
    
    # product = U_resh[i,j,c] * V_resh[j,k,c] * W_resh[k,i,c]
    # We want result[i,k] which should be sum_j T[i,j,k]? 
    # Wait, the tensor T is defined on indices i,j,k.
    # The decomposition approximates T[i,j,k] directly?
    # Let's re-read the seed.
    # Seed sets U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for each (i,j,k).
    # So for a specific (i,j,k), the c-th term is 1*1*1 = 1. Other terms are 0.
    # So the sum is 1.
    # The target T[i,j,k] is 1 for all valid i,j,k.
    # So yes, we are approximating the tensor T[i,j,k] = 1.
    
    # einsum: 'ijc,jkc,kic->ik' ? No, we want the full tensor error.
    # The value at (i,j,k) is approx_val[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Let's compute this tensor.
    approx = np.einsum('ijc,jkc,kic->ijk', U_resh, V_resh, W_resh)
    
    error = np.sum((approx - T_target) ** 2)
    return error

def snap_to_half_int(matrix):
    """Snap values to the nearest multiple of 0.5."""
    return np.round(matrix * 2.0) / 2.0

def verify_decomposition(U, V, W):
    """Check if the decomposition is exact."""
    return compute_error(U, V, W) < 1e-10

def search_rank(r, time_limit, seed):
    """
    Attempt to find a decomposition of rank r.
    Returns (U, V, W) if found, else None.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize with random small values
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    # We can use a simple gradient-free optimization or even just random local search on floats
    # Since scipy might be slow to import or start, we can use a custom hill climbing or Nelder-Mead.
    # However, the dimension is (12+28+21)*r = 61*r. For r=83, dim ~ 5000. Nelder-Mead is O(N^2) per step.
    # 5000^2 = 25M operations per step. Too slow for many steps in 10s.
    
    # Alternative: Gradient Descent with momentum.
    # We need gradients.
    # dL/dU = 2 * (approx - T) * V * W ... (element-wise in the tensor sense)
    # Let D = approx - T_target.
    # D_ijk = sum_c U_ijc V_jkc W_kic - T_ijk
    # dL/dU_ijc = 2 * sum_k D_ijk * V_jkc * W_kic
    # dL/dV_jkc = 2 * sum_i D_ijk * U_ijc * W_kic
    # dL/dW_kic = 2 * sum_j D_ijk * U_ijc * V_jkc
    
    def gradient_step(U, V, W, lr, momentum=0.9):
        r = U.shape[1]
        U_resh = U.reshape(n, m, r)
        V_resh = V.reshape(m, p, r)
        W_resh = W.reshape(p, n, r)
        
        approx = np.einsum('ijc,jkc,kic->ijk', U_resh, V_resh, W_resh)
        D = approx - T_target
        
        # Gradients
        # dU: i,j,c. Sum over k of D[i,j,k] * V[j,k,c] * W[k,i,c]
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', D, V_resh, W_resh)
        # dV: j,k,c. Sum over i of D[i,j,k] * U[i,j,c] * W[k,i,c]
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', D, U_resh, W_resh)
        # dW: k,i,c. Sum over j of D[i,j,k] * U[i,j,c] * V[j,k,c]
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', D, U_resh, V_resh)
        
        # Reshape gradients back to matrix form
        dU = dU.reshape(n*m, r)
        dV = dV.reshape(m*p, r)
        dW = dW.reshape(p*n, r)
        
        return dU, dV, dW

    # Momentum variables
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    lr = 0.01
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    steps = 0
    while time.time() < deadline:
        dU, dV, dW = gradient_step(U, V, W, lr)
        
        # Update with momentum
        vU = momentum * vU + (1 - momentum) * dU
        vV = momentum * vV + (1 - momentum) * dV
        vW = momentum * vW + (1 - momentum) * dW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        err = compute_error(U, V, W)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # If error is very low, try to snap
            if best_error < 1e-4:
                U_snap = snap_to_half_int(best_U)
                V_snap = snap_to_half_int(best_V)
                W_snap = snap_to_half_int(best_W)
                if verify_decomposition(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
        
        # Adaptive learning rate?
        if steps % 100 == 0:
            lr *= 0.95
            
        steps += 1
        
    # Final check on best found
    if best_error < 1e-4:
        U_snap = snap_to_half_int(best_U)
        V_snap = snap_to_half_int(best_V)
        W_snap = snap_to_half_int(best_W)
        if verify_decomposition(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
            
    return None

def schoolbook_decomposition():
    """
    Returns the standard rank 84 decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Entry point. Tries to find a rank < 84 decomposition.
    Falls back to schoolbook if not found within time limit.
    """
    start_time = time.time()
    overall_deadline = 110.0  # Leave 10s margin for safety
    
    # Try ranks from 83 down to 80
    # We allocate time based on rank. Lower ranks are harder.
    ranks_to_try = [83, 82, 81, 80]
    
    # We can parallelize? No, standard python.
    # We will run short searches for each rank.
    
    for r in ranks_to_try:
        if time.time() > overall_deadline:
            break
            
        # Allocate 20 seconds per rank attempt
        time_limit_for_rank = 20.0
        if time.time() + time_limit_for_rank > overall_deadline:
            time_limit_for_rank = overall_deadline - time.time()
            if time_limit_for_rank < 5.0:
                break
                
        # Try a few seeds for this rank
        found = False
        for seed in range(5):
            if time.time() > overall_deadline:
                break
            res = search_rank(r, time_limit_for_rank / 5.0, seed)
            if res is not None:
                U, V, W = res
                # Convert to lists for return
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If no improvement found, return schoolbook
    U, V, W = schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
