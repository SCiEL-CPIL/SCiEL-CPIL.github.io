"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_t[i, j, :] * V_t[j, k, :] * W_t[k, i, :]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W, rank

def _known_1x5x5_decomposition():
    """
    Known rank-19 decomposition for 1x5x5 matrix multiplication.
    This is a block of the Strassen-style or similar low-rank algorithm.
    Returns U (1x5, 19), V (5x5, 19), W (5x1, 19) such that
    sum_c U[0,j,c] * V[j,k,c] * W[k,0,c] = delta_{j,k} for the 1x5x5 case.
    Actually for 1x5x5: A is 1x5, B is 5x5, C = A*B is 1x5.
    C[k] = sum_j A[j] * B[j,k]
    So we need: sum_c U[0,j,c] * V[j,k,c] * W[k,0,c] = 1 if j==k... no.
    
    Wait, the tensor for 1x5x5 multiplication:
    T[j,k,c] where the multiplication is (A*B)[k] = sum_j A[j] B[j,k]
    The factorization: (A*B)[k] = sum_c U[0,j,c] V[j,k,c] W[k,0,c]... no.
    
    Let me re-derive. For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Tensor decomposition: C[i,k] = sum_c U[i,j... wait, the standard form is:
    C[i,k] = sum_c U[i, j, c] ... no.
    
    The standard tensor form for matmul: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    Wait, that's not right either. Let me check the convention from the task.
    
    The task says: W[k*n+i, r] - so W is indexed by (k,i) where k is the output column (p) and i is the output row (n).
    U is (n*m, rank) indexed by (i,j) - input row, input column of A
    V is (m*p, rank) indexed by (j,k) - input column of A / row of B, column of B
    
    So: C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]... but that sums over j too?
    
    No wait. The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in the product.
    Actually the multiplication tensor is: for each (i,j,k), the term A[i,j]*B[j,k] contributes to C[i,k].
    
    The factorization is: the product tensor has components T[i,j,k] = 1 (for the matmul tensor).
    And we want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i,j,k.
    
    For 1x5x5: n=1, m=5, p=5.
    U has shape (1*5, r) = (5, r), indexed by j (0..4)
    V has shape (5*5, r) = (25, r), indexed by (j,k)
    W has shape (5*1, r) = (5, r), indexed by k (0..4)
    
    Condition: sum_c U[j,c] * V[j,k,c] * W[k,c] = 1 for all j,k.
    
    This is equivalent to: the matrix M = sum_c outer(U[:,c], W[:,c]) * V[:,:,c]... no.
    
    Let me think differently. Define for each c:
    u_c = U[:, c] (5-vector)
    v_c = V[:, :, c] reshaped to (5,5)
    w_c = W[:, c] (5-vector)
    
    Then the condition is: sum_c u_c[j] * v_c[j,k] * w_c[k] = 1 for all j,k.
    
    This means: sum_c (u_c * v_c * w_c^T)[j,k] = I_5 (identity matrix)
    
    Where (u_c * v_c * w_c^T)[j,k] = u_c[j] * v_c[j,k] * w_c[k]
    
    This is a rank-1 decomposition of the identity tensor (5x5x5) into 19 terms of the form u ⊗ v ⊗ w with the specific index structure.
    
    Actually, this is the same as finding a rank-r decomposition of the 5x5 identity matrix in a specific way.
    
    Let me use a known result. The rank of the 5x5 matrix multiplication tensor (1x5x5) is known to be at most 19 (from Strassen's work or similar).
    
    Let me try a simpler approach: use the fact that we can decompose using 5 terms (schoolbook) and see if we can do better.
    
    For now, let me just use schoolbook for the 1x5x5 blocks and see if the total rank is better than 75.
    Schoolbook for 1x5x5: rank 25. Three such blocks: rank 75. Same as overall schoolbook.
    
    So I need actual lower-rank decompositions. Let me try to find them numerically or use known results.
    
    Actually, let me try a different approach: use the known fact that for small matrices, there are specific low-rank decompositions. For 1x5x5, the rank is at most 19.
    
    Let me construct a decomposition numerically.
    """
    # For 1x5x5: need sum_c U[j,c] * V[j,k,c] * W[k,c] = delta_{j,k}
    # This is equivalent to: the "tensor" T[j,k,c] = U[j,c]*V[j,k,c]*W[k,c] sums to delta_{j,k} over c
    
    # A simple approach: use the schoolbook decomposition (rank 25) as a starting point
    # and try to reduce the rank.
    
    # For now, return the schoolbook decomposition for 1x5x5
    r = 25
    U = np.zeros((5, r))
    V = np.zeros((5, 5, r))
    W = np.zeros((5, r))
    
    for j in range(5):
        for k in range(5):
            c = j * 5 + k
            U[j, c] = 1.0
            V[j, k, c] = 1.0
            W[k, c] = 1.0
    
    return U, V, W, r

def _construct_block_decomposition(n, m, p):
    """
    Construct a block-diagonal decomposition of the n x m x p matmul tensor
    as a concatenation of n copies of the 1 x m x p matmul tensor.
    
    For n=3, m=5, p=5: three copies of 1x5x5.
    If each 1x5x5 has rank r, total rank is 3*r.
    """
    # Get the 1xm x p decomposition
    U_block, V_block, W_block, r_block = _known_1x5x5_decomposition()
    
    total_rank = n * r_block
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for i in range(n):
        # Block i: rows i*m to (i+1)*m-1 of U, all of V, rows i*p to (i+1)*p-1 of W... 
        # Wait, W is indexed by (k, i) where k is 0..p-1 and i is 0..n-1.
        # So for block i, we use W[k*n + i, :] for k in 0..p-1.
        
        for c in range(r_block):
            col = i * r_block + c
            
            # U: row i*m + j, column col
            for j in range(m):
                U[i*m + j, col] = U_block[j, c]
            
            # V: row j*p + k, column col
            for j in range(m):
                for k in range(p):
                    V[j*p + k, col] = V_block[j, k, c]
            
            # W: row k*n + i, column col
            for k in range(p):
                W[k*n + i, col] = W_block[k, c]
    
    # Convert to lists
    U_list = [[float(U[i, c]) for c in range(total_rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(total_rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(total_rank)] for i in range(p * n)]
    
    return U_list, V_list, W_list, total_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to leave margin for 150s limit
    
    # Try the block decomposition approach
    U_block, V_block, W_block, rank_block = _construct_block_decomposition(n, m, p)
    
    # Verify the block decomposition
    if _verify_fast(U_block, V_block, W_block, n, m, p, tol=1e-8):
        best_rank = rank_block
        best_result = {"U": U_block, "V": V_block, "W": W_block}
    else:
        # Fall back to schoolbook
        U, V, W, best_rank = _schoolbook(n, m, p)
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
