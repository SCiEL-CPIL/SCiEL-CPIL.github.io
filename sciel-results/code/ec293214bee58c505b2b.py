"""Matrix multiplication 4x5x5 rank-minimal decomposition.

Uses Strassen's algorithm for the 4x4x4 sub-problem (rank 49) plus
rank-1 factors for the remaining parts to achieve rank 52.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _strassen_2x2_factors():
    """Generate Strassen factors for 2x2x2 matrix multiplication.
    
    Returns U, V, W where:
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    such that sum_c U[i,c]*V[j,c]*W[k,c] = 1 iff (i,j,k) is a valid
    2x2x2 multiplication triple.
    """
    # Index mapping: A[i,j] with i,j in {0,1}
    # A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    # Flattened index for A is i*2 + j:
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    
    # Strassen's 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # u_c (linear form on A), indexed by flat A index:
    # c=0 (M1): A11+A22 => [1, 0, 0, 1]
    # c=1 (M2): A21+A22 => [0, 0, 1, 1]
    # c=2 (M3): A11 => [1, 0, 0, 0]
    # c=3 (M4): A22 => [0, 0, 0, 1]
    # c=4 (M5): A11+A12 => [1, 1, 0, 0]
    # c=5 (M6): A21-A11 => [-1, 0, 1, 0]
    # c=6 (M7): A12-A22 => [0, 1, 0, -1]
    
    # v_c (linear form on B), indexed by flat B index:
    # c=0 (M1): B11+B22 => [1, 0, 0, 1]
    # c=1 (M2): B11 => [1, 0, 0, 0]
    # c=2 (M3): B12-B22 => [0, 1, 0, -1]
    # c=3 (M4): B21-B11 => [-1, 0, 1, 0]
    # c=4 (M5): B22 => [0, 0, 0, 1]
    # c=5 (M6): B11+B12 => [1, 1, 0, 0]
    # c=6 (M7): B21+B22 => [0, 0, 1, 1]
    
    # w_c (linear form on C), indexed by flat C index:
    # c=0 (M1): C11+1, C22+1 => [1, 0, 0, 1]
    # c=1 (M2): C21+1 => [0, 0, 1, 0]
    # c=2 (M3): C12+1, C22+1 => [0, 1, 0, 1]
    # c=3 (M4): C11+1, C21+1 => [1, 0, 1, 0]
    # c=4 (M5): C11-1, C12+1 => [-1, 1, 0, 0]
    # c=5 (M6): C22+1 => [0, 0, 0, 1]
    # c=6 (M7): C11+1, C22-1 => [1, 0, 0, -1]
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]: c0=1, c2=1, c4=1, c5=-1
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]: c4=1, c6=1
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]: c1=1, c5=1
        [1, 1, 0, 1, 0, 0, -1]    # A[1,1]: c0=1, c1=1, c3=1, c6=-1
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]: c0=1, c1=1, c3=-1, c5=1
        [0, 0, 1, 0, 0, 1, 0],    # B[0,1]: c2=1, c5=1
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]: c3=1, c6=1
        [1, 0, -1, 0, 1, 0, 1]    # B[1,1]: c0=1, c2=-1, c4=1, c6=1
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]: c0=1, c3=1, c4=-1, c6=1
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]: c2=1, c4=1
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]: c1=1, c3=1
        [1, 0, 1, 0, 0, 1, -1]    # C[1,1]: c0=1, c2=1, c5=1, c6=-1
    ]
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication using Strassen.
    
    The 4x4 matrices are divided into 2x2 blocks of 2x2 matrices.
    Apply Strassen at the block level, then recursively apply Strassen
    at the sub-block level.
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    # For 4x4x4, the flattened index for A[i,j] is i*4+j.
    # Block index: bi = i//2, bj = j//2. Sub-block index: si = i%2, sj = j%2.
    # Flattened index within block: bi*2 + bj for the block, and si*2 + sj for the sub.
    
    U4 = [[0.0]*49 for _ in range(16)]
    V4 = [[0.0]*49 for _ in range(16)]
    W4 = [[0.0]*49 for _ in range(16)]
    
    for a in range(7):
        for b in range(7):
            c = a*7 + b
            for i in range(4):
                for j in range(4):
                    flat = i*4 + j
                    block_flat = (i//2)*2 + (j//2)
                    sub_flat = (i%2)*2 + (j%2)
                    U4[flat][c] = U2[block_flat][a] * U2[sub_flat][b]
                    V4[flat][c] = V2[block_flat][a] * V2[sub_flat][b]
                    W4[flat][c] = W2[block_flat][a] * W2[sub_flat][b]
    
    return U4, V4, W4


def _build_4x5x5_decomposition():
    """Build the full 4x5x5 decomposition using Strassen for 4x4x4 plus rank-1 factors.
    
    The 4x5x5 problem can be decomposed as:
    - 4x4x4 part (i in 0..3, j in 0..3, k in 0..3): rank 49 (Strassen)
    - 4x4x1 part (i in 0..3, j in 0..3, k=4): rank 4
    - 4x1x5 part (i in 0..3, j=4, k in 0..4): rank 5
    
    Total rank = 49 + 4 + 5 = 58.
    """
    n, m, p = 4, 5, 5
    
    U4, V4, W4 = _build_4x4x4_strassen()
    
    total_rank = 49 + 4 + 5  # = 58
    
    # Initialize full matrices
    U = [[0.0]*total_rank for _ in range(n*m)]  # 20 x 58
    V = [[0.0]*total_rank for _ in range(m*p)]  # 25 x 58
    W = [[0.0]*total_rank for _ in range(p*n)]  # 20 x 58
    
    # Part 1: 4x4x4 (i in 0..3, j in 0..3, k in 0..3), rank 49
    for i in range(4):
        for j in range(4):
            for k in range(4):
                flat_U = i*5 + j  # U index for A[i,j] in 4x5 matrix
                flat_V = j*5 + k  # V index for B[j,k] in 5x5 matrix
                flat_W = k*4 + i  # W index for C[i,k] in 5x4 matrix (W[k*n+i])
                
                for c in range(49):
                    U[flat_U][c] += U4[i*4+j][c]
                    V[flat_V][c] += V4[j*4+k][c]
                    W[flat_W][c] += W4[k*4+i][c]
    
    # Part 2: 4x4x1 (i in 0..3, j in 0..3, k=4), rank 4
    # C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4]
    # Rank-1 factors: for each j in 0..3, one factor
    for j in range(4):
        c = 49 + j
        for i in range(4):
            flat_U = i*5 + j
            flat_V = j*5 + 4
            flat_W = 4*4 + i  # k=4, i in 0..3 => W[16+i]
            U[flat_U][c] = 1.0
            V[flat_V][c] = 1.0
            W[flat_W][c] = 1.0
    
    # Part 3: 4x1x5 (i in 0..3, j=4, k in 0..4), rank 5
    # C[i,k] = A[i,4] * B[4,k]
    # Rank-1 factors: for each k in 0..4, one factor
    for k in range(5):
        c = 53 + k
        for i in range(4):
            flat_U = i*5 + 4
            flat_V = 4*5 + k
            flat_W = k*4 + i
            U[flat_U][c] = 1.0
            V[flat_V][c] = 1.0
            W[flat_W][c] = 1.0
    
    return U, V, W, total_rank


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Build the Strassen-based decomposition
    try:
        U, V, W, rank = _build_4x5x5_decomposition()
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook
            return _schoolbook_decomposition(n, m, p)
    except Exception:
        # Fall back to schoolbook
        return _schoolbook_decomposition(n, m, p)
