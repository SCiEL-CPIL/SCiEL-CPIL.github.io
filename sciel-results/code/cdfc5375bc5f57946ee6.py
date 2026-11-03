import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Check a few random samples or all if small
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _combine_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions for M_{n,m,p1} and M_{n,m,p2} into M_{n,m,p1+p2}.
    """
    r1 = len(decomp1["U"][0])
    r2 = len(decomp2["U"][0])
    r = r1 + r2
    
    U = [row + [0]*r2 for row in decomp1["U"]]
    
    V = [[0]*r for _ in range(m * (p1 + p2))]
    for j in range(m):
        for k in range(p1):
            V[j*(p1+p2) + k] = decomp1["V"][j*p1 + k][:r1] + [0]*r2
        for k in range(p2):
            V[j*(p1+p2) + p1 + k] = [0]*r1 + decomp2["V"][j*p2 + k]
            
    W = [[0]*r for _ in range((p1 + p2) * n)]
    for k in range(p1):
        for i in range(n):
            W[k*n + i] = decomp1["W"][k*n + i][:r1] + [0]*r2
    for k in range(p2):
        for i in range(n):
            W[(p1+k)*n + i] = [0]*r1 + decomp2["W"][k*n + i]
            
    return {"U": U, "V": V, "W": W}

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2x2 Strassen blocks.
    Rank 49.
    """
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    inner = _strassen_2x2()
    
    outer_A = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    outer_B = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    outer_C = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 1, 0],
        [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner["U"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j][col] = coeff_U
                            
                    coeff_V = outer_B[t][b] * inner["V"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_j_B = 2 * (b // 2) + ei
                            g_k_B = 2 * (b % 2) + ej
                            V[g_j_B * 4 + g_k_B][col] = coeff_V
                            
                    coeff_W = outer_C[t][b] * inner["W"][li*2 + lj][r]
                    for ei in range(2):
                        for ej in range(2):
                            g_k_C = 2 * (b // 2) + ei
                            g_i_C = 2 * (b % 2) + ej
                            W[g_k_C * 4 + g_i_C][col] = coeff_W

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    d1 = _decomp_4x4x4()
    d2 = _decomp_4x4x4()
    d3 = _schoolbook(4, 4, 1)
    
    d12 = _combine_decomps(d1, d2, 4, 4, 4, 4)
    d_total = _combine_decomps(d12, d3, 4, 4, 8, 1)
    
    if _verify_fast(d_total["U"], d_total["V"], d_total["W"], n, m, p):
        return d_total
    else:
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
