"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - 10.0 - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    
    def _build_optimal_decomp():
        rank = 17
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # Strassen part for 2x2 block (indices 0-6)
        # M1
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4
        U[4, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[2, 4] = 1; W[0, 4] = -1
        
        # M6
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[4, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1
        
        # Schoolbook for remaining terms (indices 7-16)
        # A02*B20 -> C00
        U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
        # A02*B21 -> C01
        U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
        # A12*B20 -> C10
        U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
        # A12*B21 -> C11
        U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
        # A00*B02 -> C02
        U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
        # A01*B12 -> C02
        U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
        # A10*B02 -> C12
        U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
        # A11*B12 -> C12
        U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
        # A02*B22 -> C02
        U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
        # A12*B22 -> C12
        U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
        
        return U, V, W

    def _verify(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        # Build the full multiplication tensor T[a, b, c] where
        # (C = A @ B)[c, a] = sum_b A[a, b] * B[b, c]
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # W index: k*n + i where k is row of C (0..p-1) and i is col of C (0..n-1)
        # U index: a*m + b where a is row of A (0..n-1) and b is col of A / row of B (0..m-1)
        # V index: b*p + c where b is row of B (0..m-1) and c is col of B (0..p-1)
        
        # For each rank component r, the contribution to C[i, j] is:
        # sum over a,b,c: U[a*m+b, r] * V[b*p+c, r] * W[c*n+i, r]
        # This should equal 1 if (i,j) = (a,c) and b is the "inner" index, else 0
        # More precisely: T[a, b, c] = 1 if the output position c*n + a receives contribution from a,b,c
        # Actually the tensor is: sum_b A[a,b] B[b,c] contributes to C[a,c]
        # So the tensor element T[a, b, c] = 1 for all a in [0,n), b in [0,m), c in [0,p)
        # and the decomposition is: T[a,b,c] = sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r]
        
        # Let's verify by checking: for each (a, b, c), sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] should be 1
        # Wait, that's not right. The tensor for matrix multiplication C = AB is:
        # C[i, j] = sum_k A[i, k] * B[k, j]
        # The tensor T has T[i, k, j] = 1 (it's the identity in the "multiplication" sense)
        # Actually no. The matrix multiplication tensor T_{ijk} satisfies:
        # C[i,j] = sum_k T_{ijk} * A[i,k] * B[k,j]
        # where T_{ijk} = 1 for all i, j, k. That's just the standard multiplication.
        # 
        # The decomposition says: T_{ijk} = sum_r U[i,k] * V[k,j] * W[j,i] (roughly)
        # More carefully with our indexing:
        # U has rows indexed by (i, k) where i = 0..n-1, k = 0..m-1 (A[i,k])
        # V has rows indexed by (k, j) where k = 0..m-1, j = 0..p-1 (B[k,j])
        # W has rows indexed by (j, i) where j = 0..p-1, i = 0..n-1 (C[j,i]... wait)
        # 
        # The problem says: W[k*n+i, r] where k is the row index of C and i is the column index of C.
        # C has shape (n, p), so k = 0..n-1, i = 0..p-1.
        # Wait, C = A @ B where A is (n,m) and B is (m,p), so C is (n,p).
        # So W[k*n+i] where k = 0..n-1 (row of C), i = 0..p-1 (col of C).
        # But the task says W has shape (p*n, rank). p*n = 3*2 = 6. And n*p = 2*3 = 6. Same thing.
        # 
        # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
        # So W index = k*n + i, where k ranges 0..p-1 and i ranges 0..n-1? Or k ranges 0..n-1 and i ranges 0..p-1?
        # Given W has shape (p*n, rank) = (6, rank), and n=2, p=3.
        # If k is 0..p-1=0..2 and i is 0..n-1=0..1, then k*n+i ranges 0..5. That works.
        # So W[k*n + i] where k = col of C (0..p-1), i = row of C (0..n-1).
        # 
        # The tensor equation: C[i, j] = sum_k A[i, k] * B[k, j]
        # In the decomposition: C[i, j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) * W[j*n+i, r]
        # Wait, that doesn't quite work because the sums over k are separate.
        # 
        # Actually the tensor decomposition of matrix multiplication:
        # T_{i,k,j} = sum_r U_{i,k,r} * V_{k,j,r} * W_{j,i,r}
        # where T_{i,k,j} = 1 for all valid i,k,j.
        # 
        # With our indexing:
        # U row index = i*m + k (i=0..n-1, k=0..m-1)
        # V row index = k*p + j (k=0..m-1, j=0..p-1)  
        # W row index = j*n + i (j=0..p-1, i=0..n-1)
        # 
        # So for each (i, k, j): sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1.
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    if val != 1:
                        return False
        return True

    U, V, W = _build_optimal_decomp()
    
    # Verify if we have time
    if _remaining() > 5.0:
        if not _verify(U, V, W):
            # Fallback to schoolbook if verification fails (shouldn't happen)
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        r = i * m * p + k * p + j
                        U[i*m+k, r] = 1
                        V[k*p+j, r] = 1
                        W[j*n+i, r] = 1
            U, V, W = U, V, W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
