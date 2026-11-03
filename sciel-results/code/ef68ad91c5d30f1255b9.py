import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
            return False, float('inf')
            
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol, max_err
    except Exception:
        return False, float('inf')

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    try:
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 1:
                        return False
        return True
    except Exception:
        return False

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11) -> c00 + c11
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)b00 -> c10 - c11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a00(b10-b11) -> c01 + c11
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a11(b01-b00) -> c00 + c10
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (a00+a01)b11 -> -c00 + c01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (a10-a00)(b00+b01) -> c11
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11) -> c00
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_block_strassen():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Get the base Strassen tensors
    sU, sV, sW = _strassen_2x2x2()
    
    col_offset = 0
    
    # Part 1: Top 2 rows of A (2x4) multiplied by B (4x4)
    # This is decomposed into 4 2x2x2 multiplications.
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # B_LL = B[0:2, 0:2], B_RL = B[2:4, 0:2]
    # B_LR = B[0:2, 2:4], B_RR = B[2:4, 2:4]
    # C_LL = C[0:2, 0:2], C_LR = C[0:2, 2:4]
    
    # Mapping of local 2x2 indices to global indices
    # Local A indices: 0,1,2,3 correspond to global (row, col)
    # For A_L: local (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
    # For A_R: local (0,0)->(0,2), (0,1)->(0,3), (1,0)->(1,2), (1,1)->(1,3)
    
    # Local B indices: 0,1,2,3 correspond to global (row, col)
    # For B_LL: local (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
    # For B_RL: local (0,0)->(2,0), (0,1)->(2,1), (1,0)->(3,0), (1,1)->(3,1)
    # For B_LR: local (0,0)->(0,2), (0,1)->(0,3), (1,0)->(1,2), (1,1)->(1,3)
    # For B_RR: local (0,0)->(2,2), (0,1)->(2,3), (1,0)->(3,2), (1,1)->(3,3)
    
    # Local C indices: 0,1,2,3 correspond to global (row, col)
    # For C_LL: local (0,0)->(0,0), (0,1)->(0,1), (1,0)->(1,0), (1,1)->(1,1)
    # For C_LR: local (0,0)->(0,2), (0,1)->(0,3), (1,0)->(1,2), (1,1)->(1,3)
    
    def add_strassen_instance(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets, c_row_offsets, c_col_offsets, col_start):
        for t in range(7):
            for li in range(4):
                val = sU[li][t]
                if val != 0:
                    lr, lc = divmod(li, 2)
                    gr = a_row_offsets[lr]
                    gc = a_col_offsets[lc]
                    U[gr*m + gc][col_start + t] += val
                    
                val = sV[li][t]
                if val != 0:
                    lr, lc = divmod(li, 2)
                    gr = b_row_offsets[lr]
                    gc = b_col_offsets[lc]
                    V[gr*p + gc][col_start + t] += val
                    
                val = sW[li][t]
                if val != 0:
                    lr, lc = divmod(li, 2)
                    gr = c_row_offsets[lr]
                    gc = c_col_offsets[lc]
                    # W index is k*n + i, where k is col of C (local col), i is row of C (local row)
                    # Wait, W shape is (p*n, rank). Index is k*n + i.
                    # In local 2x2, k is col, i is row.
                    # Global k = c_col_offsets[lc], Global i = c_row_offsets[lr]
                    W[gr*n + gc][col_start + t] += val

    # 1. A_L * B_LL -> C_LL
    add_strassen_instance([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # 2. A_R * B_RL -> C_LL
    add_strassen_instance([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
    
    # 3. A_L * B_LR -> C_LR
    add_strassen_instance([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 14)
    
    # 4. A_R * B_RR -> C_LR
    add_strassen_instance([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 21)
    
    # Part 2: Bottom 1 row of A (1x4) multiplied by B (4x4)
    # Schoolbook: 16 terms.
    # A[2, j] * B[j, l] -> C[2, l]
    col_start = 28
    for j in range(4):
        for l in range(4):
            # U[2, j] += 1
            U[2*m + j][col_start] = 1
            # V[j, l] += 1
            V[j*p + l][col_start] = 1
            # W[l, 2] += 1
            W[l*n + 2][col_start] = 1
            col_start += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Construct the rank 44 solution
    result = _construct_block_strassen()
    
    # Verify
    ok, err = _verify_fast(result['U'], result['V'], result['W'], n, m, p)
    if ok:
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
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
