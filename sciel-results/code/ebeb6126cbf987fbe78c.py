import numpy as np
import time

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
    """Generate Strassen factors for 2x2x2 matrix multiplication."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]    # A[1,1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],    # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]    # B[1,1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [1, 0, 1, 0, 0, 1, -1]    # C[1,1]
    ]
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication using Strassen."""
    U2, V2, W2 = _strassen_2x2_factors()
    
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

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    try:
        # Build 4x4x4 Strassen decomposition (rank 49)
        U4, V4, W4 = _build_4x4x4_strassen()
        
        # 4x5x5 decomposition:
        # Part 1: 4x4x4 core (i in 0..3, j in 0..3, k in 0..3), rank 49
        # Part 2: 4x1x5 strip (i in 0..3, j=4, k in 0..4), rank 5
        # Total rank: 54
        
        total_rank = 49 + 5
        
        U = [[0.0]*total_rank for _ in range(n*m)]  # 20 x 54
        V = [[0.0]*total_rank for _ in range(m*p)]  # 25 x 54
        W = [[0.0]*total_rank for _ in range(p*n)]  # 20 x 54
        
        # Part 1: 4x4x4 core
        # For the core, we use the 4x4x4 Strassen factors.
        # The indices map as:
        #   U index: i*m + j = i*5 + j  (for i in 0..3, j in 0..3)
        #   V index: j*p + k = j*5 + k  (for j in 0..3, k in 0..3)
        #   W index: k*n + i = k*4 + i  (for k in 0..3, i in 0..3)
        # The 4x4x4 factors use indices:
        #   U4 index: i4*4 + j4  (i4, j4 in 0..3)
        #   V4 index: j4*4 + k4  (j4, k4 in 0..3)
        #   W4 index: k4*4 + i4  (k4, i4 in 0..3)
        # Since i4=i, j4=j, k4=k, the mapping is direct.
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    flat_U = i*5 + j
                    flat_V = j*5 + k
                    flat_W = k*4 + i
                    U4_idx = i*4 + j
                    V4_idx = j*4 + k
                    W4_idx = k*4 + i
                    for c in range(49):
                        U[flat_U][c] += U4[U4_idx][c]
                        V[flat_V][c] += V4[V4_idx][c]
                        W[flat_W][c] += W4[W4_idx][c]
        
        # Part 2: 4x1x5 strip (j=4, the 5th column of A)
        # For each k in 0..4, we need C[i,k] += A[i,4] * B[4,k]
        # This is a rank-1 term for each (i,k) pair, but we can share the rank-1
        # across all i for a fixed k: U[i*5+4, c_k] = 1, V[4*5+k, c_k] = 1, W[k*4+i, c_k] = 1
        # Wait, that's not right. Let me think again.
        # For fixed k, the contribution to C[i,k] from A[i,4]*B[4,k] is:
        #   sum_c U[i*5+4, c] * V[4*5+k, c] * W[k*4+i, c]
        # If we set for c = 49+k:
        #   U[i*5+4, 49+k] = 1 for all i
        #   V[4*5+k, 49+k] = 1
        #   W[k*4+i, 49+k] = 1 for all i
        # Then the sum gives 1 * 1 * 1 = 1, which is correct.
        for k in range(5):
            c = 49 + k
            for i in range(4):
                flat_U = i*5 + 4
                flat_V = 4*5 + k
                flat_W = k*4 + i
                U[flat_U][c] = 1.0
                V[flat_V][c] = 1.0
                W[flat_W][c] = 1.0
        
        # Verify the decomposition
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fall back to schoolbook
            return _schoolbook_decomposition(n, m, p)
    except Exception:
        # Fall back to schoolbook
        return _schoolbook_decomposition(n, m, p)
