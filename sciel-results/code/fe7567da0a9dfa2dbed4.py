"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22): C11 += M1, C22 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11: C21 += M2, C22 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22): C12 += M3, C22 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11): C11 += M4, C21 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a11+a12)*b22: C11 -= M5, C12 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21-a11)*(b11+b12): C22 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22): C11 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_3x3():
    """Return rank-27 schoolbook decomposition for 3x3x3 matrix multiplication.
    U: (9, 27), V: (9, 27), W: (9, 27)
    """
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def _block_decomposition_6x6x6_strassen_outer():
    """
    Decompose 6x6x6 matrix multiplication using Strassen rank-7 for 2x2 outer blocks
    and Schoolbook rank-27 for 3x3 inner blocks.
    Total rank = 7 * 27 = 189.
    Structure: 2x2 blocks of 3x3 matrices.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    sbU, sbV, sbW = _schoolbook_3x3()
    
    col = 0
    for c_outer in range(7):
        for c_inner in range(27):
            # U part
            for i_outer in range(2):
                for j_outer in range(2):
                    if sU[i_outer*2+j_outer][c_outer] != 0:
                        val_u = sU[i_outer*2+j_outer][c_outer]
                        for i_inner in range(3):
                            for j_inner in range(3):
                                if sbU[i_inner*3+j_inner][c_inner] != 0:
                                    u_row = (i_outer*3 + i_inner)*6 + (j_outer*3 + j_inner)
                                    U[u_row][col] += val_u * sbU[i_inner*3+j_inner][c_inner]
            
            # V part
            for j_outer in range(2):
                for k_outer in range(2):
                    if sV[j_outer*2+k_outer][c_outer] != 0:
                        val_v = sV[j_outer*2+k_outer][c_outer]
                        for j_inner in range(3):
                            for k_inner in range(3):
                                if sbV[j_inner*3+k_inner][c_inner] != 0:
                                    v_row = (j_outer*3 + j_inner)*6 + (k_outer*3 + k_inner)
                                    V[v_row][col] += val_v * sbV[j_inner*3+k_inner][c_inner]
            
            # W part
            for k_outer in range(2):
                for i_outer in range(2):
                    if sW[k_outer*2+i_outer][c_outer] != 0:
                        val_w = sW[k_outer*2+i_outer][c_outer]
                        for k_inner in range(3):
                            for i_inner in range(3):
                                if sbW[k_inner*3+i_inner][c_inner] != 0:
                                    w_row = (k_outer*3 + k_inner)*6 + (i_outer*3 + i_inner)
                                    W[w_row][col] += val_w * sbW[k_inner*3+i_inner][c_inner]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy einsum."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_np = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_np = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_np = np.array(W, dtype=np.float64).reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_np, V_np, W_np)
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-Outer + Schoolbook-Inner decomposition (rank 189)
    best = _block_decomposition_6x6x6_strassen_outer()
    
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback: schoolbook (rank 216)
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
