"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses deterministic algebraic construction with fast einsum verification.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    def verify_fast(U, V, W, tol=1e-9):
        """
        Fast verification using einsum.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        We want T[i,j,k] == 1 for all i,j,k.
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        
        # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        
        if np.max(np.abs(T - 1.0)) > tol:
            return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition (rank n*m*p = 144)."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()
    
    def try_kronecker_decomposition():
        """
        Try to exploit Kronecker product structure.
        4x4 * 4x9: think of 9 = 3*3, so we can view the second matrix as 4x(3*3).
        
        Strategy: Decompose using tensor product structure.
        If we can write the multiplication as a Kronecker product of smaller multiplications,
        we can combine their decompositions.
        
        For 4x4 * 4x9:
        - View as (2*2)x(2*2) * (2*2)x(3*3)
        - This doesn't directly factor nicely.
        
        Alternative: Use the fact that 4 = 2*2 and 9 = 3*3.
        The multiplication tensor M_{4,4,9} can potentially be decomposed using
        the Kronecker product of M_{2,2,3} tensors if the structure aligns.
        
        Actually, let's try a direct algebraic approach:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        For each (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k].
        The tensor is T[i,j,k] = 1 (since we're decomposing the identity map on the multiplication).
        
        Let's try structured patterns that exploit symmetries.
        """
        if time.time() > deadline - 30:
            return None
        
        # Pattern 1: Diagonal in i and k, full in j
        # For each (i,k) pair, use a separate term with U[i,j]=1, V[j,k]=1, W[k,i]=1
        # That's just schoolbook.
        
        # Pattern 2: Group terms by i
        # For fixed i, C[i,k] = sum_j A[i,j]*B[j,k]
        # Use m terms for each i: rank = n*m*p (schoolbook again)
        
        # Pattern 3: Use FFT-like transforms
        # For the j-sum, use a transform that reduces the number of terms
        # If m=4, we can use a 4-point transform (FFT or Walsh-Hadamard)
        # This could reduce the effective rank
        
        # Walsh-Hadamard transform approach for the j-dimension:
        # H_4 is a 4x4 Hadamard matrix
        # A' = A * H^T, B' = H * B, then C = A' * B' * H^{-T}
        # The multiplication A'*B' still requires m*p terms per row of A'
        # But the transforms are free (just reindexing)
        
        # Actually, the rank of matrix multiplication is invariant under
        # invertible linear transformations on the input/output spaces.
        # So transforms don't reduce rank.
        
        # Let's try: Can we share terms across different (i,k) pairs?
        # A term (u, v, w) contributes to T[i,j,k] = u[i,j]*v[j,k]*w[k,i]
        # For this to equal 1 for all (i,j,k), we need specific structure.
        
        # The only way sum_c u_c[i,j]*v_c[j,k]*w_c[k,i] = 1 for all i,j,k
        # is if the terms are carefully designed.
        
        # Try: rank = n*p terms, one for each (i,k) pair
        # For term (i,k): u[i,j]=1 for all j, v[j,k]=1 for all j, w[k,i]=1
        # Then T[i',j',k'] = sum_{(i,k)} u[i,j]*v[j,k]*w[k,i] * delta(i'=i) * delta(j'=j) * delta(k'=k)
        # Wait, this doesn't work directly.
        
        # Let me think more carefully.
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # If we set U[i*m+j, c] = delta(i, i_c) * a_c(j), 
        #              V[j*p+k, c] = delta(k, k_c) * b_c(j),
        #              W[k*n+i, c] = delta(i, i_c) * c_c(k)
        # Then T[i,j,k] = sum_c delta(i,i_c)*a_c(j)*delta(k,k_c)*b_c(j)*delta(i,i_c)*c_c(k)
        # = sum_c delta(i,i_c)*delta(k,k_c)*a_c(j)*b_c(j)*c_c(k)
        
        # For this to be 1 for all (i,j,k), we need for each (i,k):
        # sum_{c: i_c=i, k_c=k} a_c(j)*b_c(j)*c_c(k) = 1 for all j
        
        # This is still complex. Let's just try random structured patterns.
        return None
    
    def try_structured_patterns(rank):
        """
        Try various deterministic structured patterns at a given rank.
        """
        if time.time() > deadline - 20:
            return None
        
        # Pattern A: Block diagonal in i, with shared j-structure
        # Divide i into blocks, use shared V and W within blocks
        
        # Pattern B: Use the fact that 4x4 multiplication has rank at most 4*4*4=64
        # and 4x9 has rank at most 4*4*9=144
        # The combined rank is at most 144 (schoolbook)
        
        # Let's try: for each of the 4 rows of A, use 4*9=36 terms
        # Total: 4*36 = 144 (schoolbook)
        
        # Can we do better? 
        # Key insight: if two (i,k) pairs can share a term, we save rank.
        # A term contributes u[i,j]*v[j,k]*w[k,i] to T[i,j,k].
        # For it to be nonzero, we need u[i,j] != 0, v[j,k] != 0, w[k,i] != 0.
        
        # If we make u, v, w sparse in specific ways, we can share terms.
        
        # Try: rank = n*m + m*p + p*n - (n-1)*(m-1)*(p-1) or similar
        # This is getting too speculative. Let's try concrete patterns.
        
        patterns = []
        
        # Pattern 1: One term per (i,j) pair, with V and W structured
        # rank = n*m = 16
        # U[i*m+j, c] = 1 if c corresponds to (i,j), else 0
        # V[j*p+k, c] = some function of j,k
        # W[k*n+i, c] = some function of k,i
        # Need: sum_j [1 * V[j*p+k, c_ij] * W[k*n+i, c_ij]] = 1 for all k
        # where c_ij is the column for (i,j)
        # This requires: sum_j V[j*p+k, c_ij] * W[k*n+i, c_ij] = 1
        # For fixed i,k: sum_j V[j*p+k, c_ij] * W[k*n+i, c_ij] = 1
        # We have m=4 terms in the sum. Can we choose V and W to make this work?
        # V[j*p+k, c_ij] * W[k*n+i, c_ij] are products.
        # If we set V[j*p+k, c_ij] = 1/m for all j,k, and W[k*n+i, c_ij] = 1 for all k,i:
        # sum_j (1/m) * 1 = m/m = 1. But V[j*p+k, c_ij] = 1/m is not integer/half-integer
        # unless m divides evenly. 1/4 is not half-integer.
        
        # Try: V[j*p+k, c_ij] = 1 for j=0, 0 otherwise; W[k*n+i, c_ij] = 1 for all
        # Then sum_j V[j*p+k, c_ij] * W[k*n+i, c_ij] = 1*1 + 0+0+0 = 1. Works!
        # But this means only j=0 contributes, so C[i,k] = A[i,0]*B[0,k], which is wrong.
        
        # The issue: we need the sum over j to reconstruct the full product.
        # With only n*m terms, we can't have the full j-dependence in V and W.
        
        # Pattern 2: Use rank = m*p*n / some_factor
        # Let's just try random ±1 matrices and see if any work.
        
        np.random.seed(rank * 7919)
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Random ±1 with varying sparsity
            sparsity = 0.2 + 0.1 * attempt
            for c in range(rank):
                mask_U = np.random.random(n*m) < sparsity
                mask_V = np.random.random(m*p) < sparsity
                mask_W = np.random.random(p*n) < sparsity
                U[mask_U, c] = np.random.choice([-1, 1], size=mask_U.sum())
                V[mask_V, c] = np.random.choice([-1, 1], size=mask_V.sum())
                W[mask_W, c] = np.random.choice([-1, 1], size=mask_W.sum())
            
            if verify_fast(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try structured patterns for various ranks
    best_result = None
    
    # First, try to find any valid decomposition below schoolbook
    # Start from rank 100 and go up
    for rank in range(100, schoolbook_rank + 1):
        if time.time() > deadline - 25:
            break
        
        result = try_structured_patterns(rank)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
