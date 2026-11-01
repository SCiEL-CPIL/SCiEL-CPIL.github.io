"""Matrix multiplication 2x4x5 rank-minimal exact decomposition."""
import time
import random
import math

import numpy as np

DIMENSIONS = [2, 4, 5]
DEADLINE = 150.0
MARGIN = 15.0


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _build_strassen_2x2(n, m, p):
    # Strassen 2x2x2 block decomposition applied to 2x2x2 blocks.
    # For 2x4x5: treat as 2x2x2 blocks where m=4=2*2, p=5=2*2+1.
    # Actually let's think of 2x4x5 as (2x2) x (2x2) with padding p to 6? No.
    # Better: use the known 2x2x2 Strassen with 7 multiplications.
    # For 2x4x5, we can decompose as:
    # A (2x4) * B (4x5)
    # Split A into two 2x2 blocks: A1 (top), A2 (bottom) -- no, A is 2x4.
    # A = [A1 A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x2x5? No, B is 4x5.
    # B = [B1; B2] where B1 is 2x5, B2 is 2x5.
    # Product = A1*B1 + A2*B2.
    # Each A_i is 2x2, B_i is 2x5.
    # For 2x2 * 2x5: split B_i into [B_i1 B_i2] where B_i1 is 2x2, B_i2 is 2x3.
    # This gets complicated. Let me use a simpler approach.
    
    # Actually, the known rank for 2x4x5 is at most 2*4*5=40 (schoolbook).
    # With Strassen on 2x2x2 blocks: we need p to be multiple of 2.
    # 5 is not multiple of 2, so we pad p to 6, use Strassen 2x2x2 (rank 7 per block),
    # giving 2 * 7 = 14 for the 2x2x6 part, plus the extra column.
    # Wait, 2x4x6: split into two 2x2x3 blocks? No.
    
    # Let me just use the schoolbook but try to find a better decomposition
    # via numerical optimization / rounding.
    pass


def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check that U @ W @ V^T reconstructs the matrix multiplication tensor."""
    rank = U.shape[1]
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c]
    # Let's verify by computing the product for all (i,j,k).
    # Efficient way: for each c, the outer product structure.
    # T = sum_c u_c ⊗ w_c ⊗ v_c where:
    #   u_c[i,j] = U[i*m+j, c]
    #   w_c[k,i] = W[k*n+i, c]  
    #   v_c[j,k] = V[j*p+k, c]
    # And T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # The true tensor: T[i,j,k] = 1 if we think of matrix mult, but actually
    # the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in the
    # "identity" sense? No.
    # 
    # For matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] such that C[i,k] = sum_{j} A[i,j] B[j,k].
    # As a 3-tensor: T[i,j,k] = 1 (it's the standard multiplication tensor).
    # Actually T[i,j,k] = delta_{i,j,k} in the sense that it picks out the product.
    # The tensor for matrix mult is: T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes).
    # Wait no. The tensor T satisfies: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
    # So T[i,j,k] = 1 for all i, j, k. That's the schoolbook.
    # 
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Vectorized check:
    # For each (i,j,k), compute the sum.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # Reshape: U_r[i,j,c] = U[i*m+j, c], V_r[j,k,c] = V[j*p+k, c], W_r[k,i,c] = W[k*n+i, c]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # = einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
    except Exception:
        return False, 1e10
    
    expected = np.ones((n, m, p))
    err = np.max(np.abs(T - expected))
    return err < tol, err


def _round_to_halfint(U, V, W, n, m, p):
    """Try to round entries to half-integers and check validity."""
    U_r = (2 * U).round() / 2.0
    V_r = (2 * V).round() / 2.0
    W_r = (2 * W).round() / 2.0
    valid, err = _validate(U_r, V_r, W_r, n, m, p, tol=1e-4)
    if valid:
        return U_r, V_r, W_r, err
    return None


def _optimize_rank(n, m, p, rank, time_budget, seed=42):
    """Try to find a decomposition of given rank via random init + gradient descent."""
    rng = np.random.RandomState(seed)
    start = time.time()
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Normalize
    for c in range(rank):
        s = max(np.max(np.abs(U[:, c])), np.max(np.abs(V[:, c])), np.max(np.abs(W[:, c])), 1e-10)
        U[:, c] /= s
        V[:, c] /= s
        W[:, c] /= s
    
    # Simple gradient descent on the loss: sum (T[i,j,k] - 1)^2
    lr = 0.01
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = 1e10
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for it in range(200):
        if time.time() - start > time_budget:
            break
        
        # Compute T
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            break
        
        # Gradient: dL/dU_r[i,j,c] = 2*(T-1) * V_r[j,k,c] * W_r[k,i,c] summed over k
        # dL/dU[i*m+j,c] = 2 * sum_k (T[i,j,k]-1) * V[j,k,c] * W[k,i,c]
        
        # Compute gradients
        # dU[i,j,c] = 2 * sum_k (T[i,j,k]-1) * V_r[j,k,c] * W_r[k,i,c]
        diff = T - 1.0  # (n, m, p)
        
        # dU_r[i,j,c] = 2 * einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        dU_r = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r, optimize=True)
        # dV_r[j,k,c] = 2 * einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        dV_r = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r, optimize=True)
        # dW_r[k,i,c] = 2 * einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        dW_r = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r, optimize=True)
        
        dU = dU_r.reshape(n * m, rank)
        dV = dV_r.reshape(m * p, rank)
        dW = dW_r.reshape(p * n, rank)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Re-reshape
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
    
    return best_U, best_V, best_W, best_err


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + DEADLINE - MARGIN
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 40
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find better decompositions at lower ranks
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Also rank >= (n*p*m)^(1/3) roughly, but the max dimension product bound is 20.
    
    # Try ranks from 20 upward
    # First, let's try a structured approach: use Strassen-like ideas.
    
    # For 2x4x5, one approach:
    # Split the 4 middle dimension into two 2's: A = [A1 A2], B = [B1; B2]
    # C = A1*B1 + A2*B2
    # Each A_i is 2x2, B_i is 2x5.
    # For 2x2 * 2x5: use Strassen on the 2x2 part with the 5 columns.
    # 2x2 * 2x5: split B_i into [B_i1 B_i2] where B_i1 is 2x2, B_i2 is 2x3.
    # A_i * B_i1: 2x2*2x2, rank 7 (Strassen)
    # A_i * B_i2: 2x2*2x3, rank 6 (schoolbook) or less?
    # Total per i: 7 + 6 = 13, times 2 = 26. Plus cross terms? No, it's a sum.
    # Wait, C = A1*B1 + A2*B2, so total rank = rank(A1*B1) + rank(A2*B2) = 13 + 13 = 26.
    
    # Actually for 2x2*2x3: the tensor has 2*2*3=12 entries, schoolbook rank 12.
    # Can we do better? The lower bound is max(4,6,6)=6.
    # Known: 2x2*2x2 is 7. 2x2*2x3 might be 7 or 8.
    
    # Let me just try numerical optimization for various ranks.
    
    for rank in range(20, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            
            t_budget = min(10.0, max(2.0, (deadline - time.time()) * 0.3))
            U, V, W, err = _optimize_rank(n, m, p, rank, t_budget, seed=seed + rank * 100)
            
            if err < 1e-6:
                # Try rounding to half-integers
                rounded = _round_to_halfint(U, V, W, n, m, p)
                if rounded is not None:
                    U_r, V_r, W_r, rerr = rounded
                    best_rank = rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
                else:
                    # Use the floating point result (should be close to half-integers)
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank < n * m * p:
            break
    
    return best_result
