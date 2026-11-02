import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 48 decomposition: 16 columns x 3 terms per column (2x2 x 2x1 rank 3)
    # For each column j, compute C[:,j] = A * B[:,j] using rank-3 2x2x1 decomposition
    # T[i,k,j] = delta(i,k) for i,k in {0,1}, j in {0..15}
    # 
    # 2x2 x 2x1 rank-3 decomposition:
    # M1 = A[0,0] * b[0]
    # M2 = A[0,1] * b[1]
    # M3 = (A[1,0] + A[1,1]) * (b[0] + b[1])
    # C[0] = M1 + M2
    # C[1] = M3 + A[1,0]*b[0] + A[1,1]*b[1] - M3... 
    # 
    # Actually let me use a verified rank-3 scheme:
    # M1 = A00 * b0
    # M2 = A11 * b1
    # M3 = (A00 + A11) * (b0 + b1)
    # C0 = M1 + M3 - M2 - A11*b0 ... no
    #
    # Standard rank-3 for 2x2x1:
    # M1 = A00*b0
    # M2 = A01*b1
    # M3 = (A10+A11)*(b0+b1)
    # C0 = M1 + M2
    # C1 = M3 + (A10*b0 + A11*b1) - M3 ... this is circular
    #
    # Let me just use rank 4 per column (safe) = 64, or try a different approach.
    #
    # Alternative: use the 8-block Strassen (rank 56) but try to merge blocks.
    # 
    # Let me try rank 40: 8 blocks x 5 terms. For 2x2x2, can we do rank 5? No, min is 7.
    #
    # New structural approach: view as 4x1 x 2x16. The 4x1 matrix is A flattened.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # This is a 4x2 x 16 tensor (flattened A is 4-dim, B is 2x16=32-dim, C is 2x16=32-dim)
    # T[a,k2,j] where a = i*m+k (0..3), k2 = k (0..1), j (0..15)
    # T[a,k2,j] = 1 if a == i*m+k and k2==k, i.e., if floor(a/m)==i and a%m==k and k2==k
    # So T[a,k2,j] = 1 if a%m == k2, else 0. (j is free)
    #
    # This tensor factors as: for each j, T[:, :, j] = same matrix M where M[a,k2] = 1 if a%m==k2
    # M is a 4x2 matrix: [[1,0],[0,1],[1,0],[0,1]]
    # The tensor is M ⊗ I_16 (Kronecker product with identity on j)
    # Rank of M ⊗ I_p = rank(M) * rank(I_p) = rank(M) * p if M is non-zero
    # rank(M) for [[1,0],[0,1],[1,0],[0,1]]: this matrix has rank 2
    # So tensor rank = 2 * 16 = 32? No, tensor rank of Kronecker product is product of ranks only for specific cases.
    #
    # Actually, the tensor T[a,k2,j] = M[a,k2] * delta_j (no, j is free)
    # T[a,k2,j] = M[a,k2] for all j (same for every j)
    # This means T = sum_j (M ⊗ e_j) where e_j is unit vector in j-space
    # The rank of this tensor: since M has rank 2 as a matrix, and the j-dimension is 16,
    # the tensor rank should be 2*16 = 32 if M can be decomposed as sum of 2 rank-1 matrices.
    # M = u1*v1^T + u2*v2^T where u1=[1,0,1,0]^T, v1=[1,0]^T, u2=[0,1,0,1]^T, v2=[0,1]^T
    # So T = sum_j sum_{s=1}^{2} (u_s ⊗ e_j) ⊗ (v_s ⊗ e_j)
    # This gives a rank-32 decomposition!
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # u1 = [1,0,1,0]^T (indices in U: 0,1,2,3 correspond to (i,k) pairs)
    # u2 = [0,1,0,1]^T
    # v1 = [1,0]^T (indices in V: k*p+j, so for k=0: 0..15, for k=1: 16..31)
    # v2 = [0,1]^T
    # W[j*n+i] for the output
    
    for j in range(p):
        # Term s=0: u1 * v1 * w_j
        col0 = 2 * j
        # U part: u1 = [1,0,1,0]
        U[0][col0] = 1  # i=0,k=0
        U[2][col0] = 1  # i=1,k=0
        # V part: v1 = [1,0], so V[k*p+j] for k=0 is 1
        V[0 * p + j][col0] = 1
        # W part: W[j*n+i] should give C[i,j] = A[0,0]*B[0,j] + ... 
        # We need W[j*n+i] such that sum over terms gives delta(i,k)
        # For term col0: U[0]*V[0,j]*W[j,0] + U[2]*V[0,j]*W[j,1] should contribute to T
        # T[0,0,j] = U[0,col0]*V[0,j,col0]*W[j,0,col0] = 1*1*W[j,0,col0] should be 1
        # T[1,0,j] = U[2,col0]*V[0,j,col0]*W[j,1,col0] = 1*1*W[j,1,col0] should be 1
        W[j * n + 0][col0] = 1
        W[j * n + 1][col0] = 1
        
        # Term s=1: u2 * v2 * w_j
        col1 = 2 * j + 1
        # U part: u2 = [0,1,0,1]
        U[1][col1] = 1  # i=0,k=1
        U[3][col1] = 1  # i=1,k=1
        # V part: v2 = [0,1], so V[k*p+j] for k=1 is 1
        V[1 * p + j][col1] = 1
        # W part:
        # T[0,1,j] = U[1,col1]*V[1,j,col1]*W[j,0,col1] = 1*1*W[j,0,col1] should be 1
        # T[1,1,j] = U[3,col1]*V[1,j,col1]*W[j,1,col1] = 1*1*W[j,1,col1] should be 1
        W[j * n + 0][col1] = 1
        W[j * n + 1][col1] = 1
    
    # Verify: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # For i=0,k=0,j: only col=2j has U[0,2j]=1, V[0*p+j, 2j]=1, W[j*n+0, 2j]=1 -> product=1
    #   col=2j+1: U[0,2j+1]=0 -> no contribution
    #   Other cols: U[0, other]=0
    # So T[0,0,j] = 1. Correct.
    # For i=0,k=1,j: only col=2j+1 has U[1,2j+1]=1, V[1*p+j, 2j+1]=1, W[j*n+0, 2j+1]=1 -> product=1
    # So T[0,1,j] = 1. Correct.
    # For i=1,k=0,j: only col=2j has U[2,2j]=1, V[0*p+j, 2j]=1, W[j*n+1, 2j]=1 -> product=1
    # So T[1,0,j] = 1. Correct.
    # For i=1,k=1,j: only col=2j+1 has U[3,2j+1]=1, V[1*p+j, 2j+1]=1, W[j*n+1, 2j+1]=1 -> product=1
    # So T[1,1,j] = 1. Correct.
    # For i!=k: T[i,k,j] = 0. Correct.
    
    # Quick verification
    max_err = 0.0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r_idx in range(rank):
                    val += U[i*m+k][r_idx] * V[k*p+j][r_idx] * W[j*n+i][r_idx]
                expected = 1.0 if i == k else 0.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to naive rank 64
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * m * p + k * p + j
                    U[i*m+k][idx] = 1
                    V[k*p+j][idx] = 1
                    W[j*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
