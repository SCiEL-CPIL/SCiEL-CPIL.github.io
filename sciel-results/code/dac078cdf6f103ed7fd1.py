"""Matrix multiplication 2x4 x 4x8 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Build the multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 for all valid indices (the multiplication tensor)
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Try to find exact rank-r decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait - let me re-read the convention.
    # U shape: (n*m, rank) - indexed by (i,j)
    # V shape: (m*p, rank) - indexed by (j,k)  
    # W shape: (p*n, rank) - indexed by (k,i)
    # Output: C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition means: for each column c, the product of three slices
    # gives a contribution to the tensor.
    
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is flattened to (n*m, rank), V to (m*p, rank), W to (p*n, rank)
    
    # Let me verify: the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # A rank-1 term U[i,j,c]*V[j,k,c]*W[k,i,c] contributes to T[i,j,k]
    
    # Schoolbook: rank = n*m*p = 64
    # Each term picks specific i,j,k
    
    # Strategy: Use iterative rank reduction
    # Start with schoolbook, try to merge terms
    
    def verify_decomposition(U, V, W, T_target, tol=1e-9):
        """Verify that U,V,W exactly decompose T_target."""
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Compute reconstructed tensor
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            # Outer product of U[:, :, c], V[:, :, c], W[:, :, c]
            # U[i,j,c] * V[j,k,c] * W[k,i,c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
        
        if np.max(np.abs(T_recon - T_target)) > tol:
            return False
        return True
    
    def verify_decomposition_fast(U, V, W, T_target, tol=1e-9):
        """Faster verification using einsum."""
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T_recon[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        T_recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        if np.max(np.abs(T_recon - T_target)) > tol:
            return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies
    # Use the fact that the tensor is constant (all ones)
    # T[i,j,k] = 1 for all i,j,k
    
    # A rank-r decomposition exists if we can find vectors such that
    # sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c] = 1 for all i,j,k
    
    # Let's try small ranks systematically
    # For rank r, we need to solve a system of n*m*p = 64 equations
    
    # Try rank from 1 upward, but that's expensive. Let's try smarter.
    
    # Key insight: T[i,j,k] = 1 is a rank-1 tensor in the sense that it's
    # the outer product of all-ones vectors! 
    # T[i,j,k] = 1*1*1 = (1_i)(1_j)(1_k)
    # So rank 1 should work!
    
    # U[i,j,c] = 1 for all i,j when c=0
    # V[j,k,c] = 1 for all j,k when c=0  
    # W[k,i,c] = 1 for all k,i when c=0
    # Then sum_c U*V*W = 1*1*1 = 1 = T[i,j,k]
    
    U1 = np.ones((n * m, 1), dtype=np.float64)
    V1 = np.ones((m * p, 1), dtype=np.float64)
    W1 = np.ones((p * n, 1), dtype=np.float64)
    
    if verify_decomposition_fast(U1, V1, W1, T):
        # Rank 1 works!
        return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}
    
    # That should work. Let me double-check the indexing convention.
    # U shape: (n*m, rank) - row index is i*m + j
    # V shape: (m*p, rank) - row index is j*p + k
    # W shape: (p*n, rank) - row index is k*n + i
    
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # With U = all ones, V = all ones, W = all ones:
    # T[i,j,k] = 1 * 1 * 1 = 1 ✓
    
    # So rank 1 is exact!
    
    # But wait - let me make sure the evaluator checks this correctly.
    # The output factor follows W[k*n+i, r] convention.
    
    return {"U": U1.tolist(), "V": V1.tolist(), "W": W1.tolist()}

