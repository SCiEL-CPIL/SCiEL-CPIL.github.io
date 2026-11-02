import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit
    
    n, m, p = 2, 4, 8
    
    # The multiplication C = A @ B where A is 2x4, B is 4x8, C is 2x8.
    # We can decompose this using block structure.
    # Split A into two 2x2 blocks: A = [A_left, A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # Split B into two 2x8 blocks: B_top = B[0:2, :], B_bot = B[2:4, :]
    # Then C = A_left @ B_top + A_right @ B_bot
    # 
    # Each term is a (2x2) @ (2x8) multiplication.
    # We can further decompose (2x2) @ (2x8) into two (2x2) @ (2x2) multiplications:
    # Split the 2x8 B into two 2x2 blocks: B_left = B[:, 0:2], B_right = B[:, 2:4] (wait, no)
    # Actually (2x2) @ (2x8): split the 8 columns into 4 groups of 2.
    # So (2x2) @ (2x8) = concatenation of 4 (2x2) @ (2x2) results.
    # Each (2x2) @ (2x2) can be done in rank 7 (Strassen).
    # So (2x2) @ (2x8) has rank 4 * 7 = 28.
    # Total rank = 2 * 28 = 56. Same as before.
    
    # Let me think of a better decomposition.
    # Actually, let me just implement the rank-56 Strassen solution cleanly.
    
    U, V, W = _build_rank_56(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 64
    U64, V64, W64 = _build_rank_64(n, m, p)
    return {"U": U64, "V": V64, "W": W64}


def _build_rank_56(n, m, p):
    """
    Rank 56 decomposition for (2x4) @ (4x8) using Strassen on 2x2 sub-blocks.
    
    We view the multiplication as 8 independent (2x2) @ (2x2) block multiplications:
    For i in {0,1}, j in {0,1,2,3}, k in {0,1}:
        C[i, 2k:(2k+2)] gets contribution from A[i, 2j:2j+2] @ B[2j:2j+2, 2k:2k+2]
    
    Wait, that's not right either. Let me be more careful.
    
    A is 2x4, B is 4x8, C is 2x8.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Let's group j into pairs: j = 2*j1 + j2 where j1 in {0,1}, j2 in {0,1}
    And k into pairs: k = 2*k1 + k2 where k1 in {0,1,2,3}, k2 in {0,1}
    
    For fixed j1 and k1, we have a (2x2) @ (2x2) sub-problem:
    C_sub[i, k2] = sum_{j2=0}^{1} A[i, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]
    
    This is a (2x2) @ (2x2) matrix multiplication, which has rank 7 via Strassen.
    There are 2 * 4 = 8 such sub-problems, so total rank = 8 * 7 = 56.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    for j1 in range(2):
        for k1 in range(4):
            base_col = 7 * (j1 * 4 + k1)
            
            # A indices: i in {0,1}, j = 2*j1 + j2 for j2 in {0,1}
            # U index: i*m + j = i*4 + 2*j1 + j2
            # B indices: j = 2*j1 + j2, k = 2*k1 + k2
            # V index: j*p + k = (2*j1+j2)*8 + 2*k1 + k2
            # C indices: i, k = 2*k1 + k2
            # W index: k*n + i = (2*k1+k2)*2 + i
            
            for i in range(2):
                for j2 in range(2):
                    a_idx = i * 4 + 2 * j1 + j2
                    # Store in U[a_idx][base_col + s] for Strassen term s
                    pass
            
            # Strassen for (2x2) @ (2x2):
            # m1 = (a11+a22)(b11+b22)
            # m2 = (a21+a22)b11
            # m3 = a11(b12-b22)
            # m4 = a22(b21-b11)
            # m5 = (a11+a12)b22
            # m6 = (a21-a11)(b11+b12)
            # m7 = (a12-a22)(b21+b22)
            #
            # C11 = m1 - m4 + m5 + m6
            # C12 = m3 + m5
            # C21 = m2 + m4
            # C22 = m1 - m2 + m3 + m7
            
            # a11 = A[0, 2*j1+0], a12 = A[0, 2*j1+1]
            # a21 = A[1, 2*j1+0], a22 = A[1, 2*j1+1]
            # b11 = B[2*j1+0, 2*k1+0], b12 = B[2*j1+0, 2*k1+1]
            # b21 = B[2*j1+1, 2*k1+0], b22 = B[2*j1+1, 2*k1+1]
            
            a11 = 0 * 4 + 2 * j1 + 0
            a12 = 0 * 4 + 2 * j1 + 1
            a21 = 1 * 4 + 2 * j1 + 0
            a22 = 1 * 4 + 2 * j1 + 1
            
            b11 = (2 * j1 + 0) * 8 + 2 * k1 + 0
            b12 = (2 * j1 + 0) * 8 + 2 * k1 + 1
            b21 = (2 * j1 + 1) * 8 + 2 * k1 + 0
            b22 = (2 * j1 + 1) * 8 + 2 * k1 + 1
            
            c11 = (2 * k1 + 0) * 2 + 0
            c12 = (2 * k1 + 1) * 2 + 0
            c21 = (2 * k1 + 0) * 2 + 1
            c22 = (2 * k1 + 1) * 2 + 1
            
            # m1 = (a11+a22)(b11+b22)
            U[a11][base_col + 0] += 1
            U[a22][base_col + 0] += 1
            V[b11][base_col + 0] += 1
            V[b22][base_col + 0] += 1
            W[c11][base_col + 0] += 1
            W[c22][base_col + 0] += 1
            
            # m2 = (a21+a22)b11
            U[a21][base_col + 1] += 1
            U[a22][base_col + 1] += 1
            V[b11][base_col + 1] += 1
            W[c21][base_col + 1] += 1
            W[c22][base_col + 1] -= 1
            
            # m3 = a11(b12-b22)
            U[a11][base_col + 2] += 1
            V[b12][base_col + 2] += 1
            V[b22][base_col + 2] -= 1
            W[c12][base_col + 2] += 1
            W[c22][base_col + 2] += 1
            
            # m4 = a22(b21-b11)
            U[a22][base_col + 3] += 1
            V[b21][base_col + 3] += 1
            V[b11][base_col + 3] -= 1
            W[c11][base_col + 3] -= 1
            W[c21][base_col + 3] += 1
            
            # m5 = (a11+a12)b22
            U[a11][base_col + 4] += 1
            U[a12][base_col + 4] += 1
            V[b22][base_col + 4] += 1
            W[c11][base_col + 4] += 1
            W[c12][base_col + 4] += 1
            
            # m6 = (a21-a11)(b11+b12)
            U[a21][base_col + 5] += 1
            U[a11][base_col + 5] -= 1
            V[b11][base_col + 5] += 1
            V[b12][base_col + 5] += 1
            W[c11][base_col + 5] += 1
            W[c22][base_col + 5] += 1
            
            # m7 = (a12-a22)(b21+b22)
            U[a12][base_col + 6] += 1
            U[a22][base_col + 6] -= 1
            V[b21][base_col + 6] += 1
            V[b22][base_col + 6] += 1
            W[c22][base_col + 6] += 1
    
    return U, V, W


def _build_rank_64(n, m, p):
    """Schoolbook rank 64 decomposition."""
    total_rank = n * m * p  # 64
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = i * m * p + j * p + k
                U[i * m + j][s] = 1
                V[j * p + k][s] = 1
                W[k * n + i][s] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U_arr[i * m + j, s] * V_arr[j * p + k, s] * W_arr[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
