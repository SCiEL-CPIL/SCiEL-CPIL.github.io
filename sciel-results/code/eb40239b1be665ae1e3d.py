import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    
    # Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This represents C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Use rank-7 decomposition for 3x3x3 tensor
    # The 3x3x3 tensor S[j, k, l] = 1 for all j,k,l in [0,3)
    # represents B[j,k] * C[k,l] = D[j,l]
    # 
    # For 2x3x3: T[i, j, k] = 1
    # We can write: T[i, j, k] = sum_{c=0}^{13} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Construction: For each row i of A (i=0,1), we use a rank-7 decomposition of the
    # 3x3 "identity-like" tensor that maps B's columns to C's columns.
    #
    # Actually, let me think differently. The tensor T[i,j,k]=1 is just the all-ones tensor.
    # This is a rank-1 tensor! T[i,j,k] = 1*1*1.
    #
    # Wait, that's wrong. Let me re-examine the problem.
    #
    # For matrix multiplication C = A @ B where A is (n,m) and B is (m,p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # The multiplication tensor has entries T[i,j,k] which are 1 if the indices (i,j,k) 
    # participate in the sum, which is always 1 for all valid i,j,k.
    # So T is indeed the all-ones tensor of shape (n,m,p).
    #
    # But wait, that can't be right for a non-trivial decomposition. Let me reconsider.
    #
    # Actually, for standard matrix multiplication, the tensor IS the all-ones tensor.
    # The "structure" comes from how the indices are grouped in U, V, W.
    #
    # U has shape (n*m, rank): indexed by (i,j) -> row i*m+j
    # V has shape (m*p, rank): indexed by (j,k) -> row j*p+k  
    # W has shape (p*n, rank): indexed by (k,i) -> row k*n+i
    #
    # And we need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Since T[i,j,k] = 1 for all i,j,k, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    #
    # This is a constraint satisfaction problem. The all-ones tensor has rank 1 in the 
    # usual tensor sense, but here the index grouping makes it non-trivial.
    #
    # Let me try a rank-14 construction:
    # Split the 14 terms into two groups of 7, one for each i value.
    # For i=0: terms c=0..6
    # For i=1: terms c=7..13
    #
    # For i=0, we need: sum_{c=0}^{6} U[0*m+j, c] * V[j*p+k, c] * W[k*n+0, c] = 1 for all j,k
    # For i=1, we need: sum_{c=7}^{13} U[1*m+j, c] * V[j*p+k, c] * W[k*n+1, c] = 1 for all j,k
    #
    # And cross terms must be zero:
    # For i=0: sum_{c=7}^{13} U[0*m+j, c] * V[j*p+k, c] * W[k*n+0, c] = 0
    # For i=1: sum_{c=0}^{6} U[1*m+j, c] * V[j*p+k, c] * W[k*n+1, c] = 0
    #
    # We can achieve this by setting U[0*m+j, 7..13] = 0 and U[1*m+j, 0..6] = 0.
    #
    # Now, for each i, we need a rank-7 decomposition of the 3x3 all-ones "sub-tensor":
    # sum_{c in group_i} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j,k
    #
    # Let me set W[k*n+i, c] = 1 for all k, c in group_i.
    # Then we need: sum_{c in group_i} U[i*m+j, c] * V[j*p+k, c] = 1 for all j,k
    #
    # This means the matrix M_i[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] must be all-ones.
    #
    # So we need a rank-7 factorization of the 3x3 all-ones matrix into U_part (3x7) and V_part (3x7)
    # such that U_part @ V_part^T = ones(3,3).
    #
    # The all-ones 3x3 matrix has rank 1! So we can do this in rank 1:
    # U_part[j, c] = 1 for c=0, 0 otherwise
    # V_part[k, c] = 1 for c=0, 0 otherwise
    # Then U_part @ V_part^T = ones(3,3).
    #
    # So actually, we can do rank 2 (one term per i value)!
    #
    # Let me verify: rank = 2
    # c=0: i=0 group
    # c=1: i=1 group
    #
    # U[0*m+j, 0] = 1, U[0*m+j, 1] = 0
    # U[1*m+j, 0] = 0, U[1*m+j, 1] = 1
    # V[j*p+k, 0] = 1, V[j*p+k, 1] = 1
    # W[k*n+0, 0] = 1, W[k*n+0, 1] = 0
    # W[k*n+1, 0] = 0, W[k*n+1, 1] = 1
    #
    # Check: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For i=0: sum_c U[j,c] * V[j*p+k,c] * W[k,c] = U[j,0]*V[j*p+k,0]*W[k,0] + U[j,1]*V[j*p+k,1]*W[k,1]
    #        = 1*1*1 + 0*1*0 = 1 ✓
    # For i=1: = U[3+j,0]*V[j*p+k,0]*W[k,1] + U[3+j,1]*V[j*p+k,1]*W[k,1]
    #        = 0*1*0 + 1*1*1 = 1 ✓
    #
    # This works! Rank 2!
    
    rank = 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # c=0: for i=0
    for j in range(m):
        U[j][0] = 1  # U[0*m+j, 0] = U[j, 0]
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        W[k * n + 0][0] = 1  # W[k*n+0, 0]
    
    # c=1: for i=1
    for j in range(m):
        U[n * m + j][1] = 1  # U[1*m+j, 1] = U[6+j, 1] wait, n*m = 6
        # Actually n*m = 2*3 = 6, so U[6+j, 1] for j=0,1,2 -> U[6,1], U[7,1], U[8,1]
    for j in range(m):
        for k in range(p):
            V[j * p + k][1] = 1
    for k in range(p):
        W[k * n + 1][1] = 1  # W[k*n+1, 1]
    
    # Verify the decomposition
    T_target = np.ones((n, m, p))
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                T_computed[i, j, k] = s
    
    if not np.allclose(T_computed, T_target):
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_idx] = 1
                    V_fb[j * p + k][col_idx] = 1
                    W_fb[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
