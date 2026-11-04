"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor for matmul is T[i,j,k] = 1 for all valid (i,j,k) in the standard sense?
    # No. The matrix multiplication tensor T has T[i,j,k] = 1 if the entry (i,k) of the product
    # depends on A[i,j] * B[j,k]. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's the full tensor of ones? No.
    # The matrix multiplication tensor is defined as T(A,B) = A*B where (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # As a trilinear form, this is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] where C = A*B.
    # Actually, the standard convention: the matmul tensor T has T[i,j,k] = 1 for all i,j,k.
    # Let me re-read the task: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 5 matrix."
    # The tensor is T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p).
    # Because (AB)[i,k] = sum_j A[i,j] * B[j,k], so the trilinear form is sum_{i,j,k} A[i,j] B[j,k] (AB)[i,k]
    # which equals sum_{i,j,k} A[i,j] B[j,k] * sum_l A[i,l] B[l,k]... no.
    # 
    # Let me think again. The matrix multiplication map is: (A, B) -> A*B.
    # As a tensor, this is a 3-tensor T in U* ⊗ V* ⊗ W* where U is the space of A's, V is space of B's, W is space of (AB)'s.
    # T[a,b,c] = 1 if c = a*b (in the sense of the bilinear map).
    # More precisely, T is the tensor such that for all A, B, C: 
    #   sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = <C, A*B> = sum_{i,k} C[k,i] (A*B)[i,k] = sum_{i,k} C[k,i] sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    # 
    # So we need to decompose the all-ones tensor of shape (n, m, p) into rank-r terms.
    # A rank-1 decomposition term is u_i * v_j * w_k.
    # So we need: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank_truncated_svd(n, m, p, max_rank, deadline):
    """
    Try to find a rank-r decomposition of the all-ones tensor T[i,j,k]=1.
    This is equivalent to finding matrices U, V, W such that
    T = sum_c u_c ⊗ v_c ⊗ w_c where T is the all-ones tensor.
    
    The all-ones tensor of shape (n,m,p) has a special structure. It can be written as
    T[i,j,k] = 1 = (sum of rank-1 terms).
    
    One approach: note that the all-ones tensor is the outer product of three all-ones vectors.
    So it has rank 1! T[i,j,k] = 1*1*1 = 1.
    
    Wait, that means the rank is 1!
    U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    Then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1. Yes!
    
    So the rank is 1.
    """
    # The all-ones tensor has rank 1.
    rank = 1
    U = [[1]]
    V = [[1]]
    W = [[1]]
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (all-ones tensor).
    # This tensor has rank 1: T = u ⊗ v ⊗ w where u, v, w are all-ones vectors.
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    
    rank = 1
    U = [[1]]
    V = [[1]]
    W = [[1]]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
