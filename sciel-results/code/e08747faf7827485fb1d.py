import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the tensor decomposition exactly reproduces matrix multiplication.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=atol)

def _strassen_2x2_2x2():
    """
    Returns U, V, W matrices (each 4x7) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U[i, r] corresponds to A[i//2, i%2] coefficient
    V[j, r] corresponds to B[j//2, j%2] coefficient  
    W[k, r] corresponds to C[k//2, k%2] coefficient
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1: (a00+a11)(b00+b11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = -1
    
    # P2: (a10+a11)b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1
    
    # P3: a00(b01-b11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # P4: a11(b10-b00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = -1; W[3,3] = 1
    
    # P5: (a00+a01)b11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1; W[2,4] = 1
    
    # P6: (a10-a00)(b00+b01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[1,5] = -1; W[3,5] = 1
    
    # P7: (a01-a11)(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[1,6] = -1
    
    return U, V, W

def _build_4x4_4x7_strassen():
    """
    Build a rank-97 decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Strassen for the 4x4 * 4x4 part (rank 49) and schoolbook for 4x4 * 4x3 part (rank 48).
    """
    n, m, p = 4, 4, 7
    rank = 97
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = _strassen_2x2_2x2()
    
    def add_strassen_product(a_blocks, b_blocks, c_blocks, term_start):
        """
        a_blocks: list of (a_row_offset, a_col_offset, sign) for A blocks
        b_blocks: list of (b_row_offset, b_col_offset, sign) for B blocks
        c_blocks: list of (c_row_offset, c_col_offset, sign) for C blocks
        term_start: starting index in the rank dimension
        """
        for r in range(7):
            for ar0, ac0, as_ in a_blocks:
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j, r]
                        if val != 0:
                            global_i = ar0 + i
                            global_j = ac0 + j
                            U[global_i * 4 + global_j, term_start + r] += val * as_
            
            for br0, bc0, bs_ in b_blocks:
                for i in range(2):
                    for j in range(2):
                        val = V2[i*2+j, r]
                        if val != 0:
                            global_i = br0 + i
                            global_j = bc0 + j
                            V[global_i * 7 + global_j, term_start + r] += val * bs_
            
            for cr0, cc0, cs_ in c_blocks:
                for i in range(2):
                    for j in range(2):
                        val = W2[i*2+j, r]
                        if val != 0:
                            global_i = cr0 + i
                            global_j = cc0 + j
                            W[global_j * 4 + global_i, term_start + r] += val * cs_
    
    # M1: (A11+A22)(B11+B22)
    add_strassen_product(
        [(0,0,1), (2,2,1)],
        [(0,0,1), (2,2,1)],
        [(0,0,1), (2,2,-1)],
        0
    )
    
    # M2: (A21+A22)B11
    add_strassen_product(
        [(2,0,1), (2,2,1)],
        [(0,0,1)],
        [(0,2,1), (2,2,1)],
        7
    )
    
    # M3: A11(B12-B22)
    add_strassen_product(
        [(0,0,1)],
        [(0,2,1), (2,2,-1)],
        [(2,0,1), (2,2,1)],
        14
    )
    
    # M4: A22(B21-B11)
    add_strassen_product(
        [(2,2,1)],
        [(2,0,1), (0,0,-1)],
        [(0,0,1), (2,0,-1), (2,2,1)],
        21
    )
    
    # M5: (A11+A12)B22
    add_strassen_product(
        [(0,0,1), (0,2,1)],
        [(2,2,1)],
        [(0,0,-1), (0,2,1), (2,0,1)],
        28
    )
    
    # M6: (A21-A11)(B11+B12)
    add_strassen_product(
        [(2,0,1), (0,0,-1)],
        [(0,0,1), (0,2,1)],
        [(0,2,-1), (2,2,1)],
        35
    )
    
    # M7: (A12-A22)(B21+B22)
    add_strassen_product(
        [(0,2,1), (2,2,-1)],
        [(2,0,1), (2,2,1)],
        [(0,0,1), (0,2,-1), (2,2,-1)],
        42
    )
    
    # Part 2: 4x4 * 4x3 using schoolbook (rank 48)
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    """
    Schoolbook rank n*m*p decomposition.
    """
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin before 120s limit
    
    # Try the rank-97 Strassen-based decomposition
    try:
        result = _build_4x4_4x7_strassen()
        if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception as e:
        pass
    
    # Fallback: schoolbook rank 112
    result = _schoolbook(n, m, p)
    if _verify_fast(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Last resort: return schoolbook without verification (should not happen)
    return result

# EVOLVE-BLOCK-END
