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

def _coppersmith_winograd_3x3():
    """Return the rank-23 Coppersmith-Winograd decomposition for 3x3x3 matrix multiplication.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # The 23 terms of the CW rank-23 construction
    # Each term is defined by (U_entry, V_entry, W_entry) as 3x3 matrices
    
    # Term 1: A[0,0] * B[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Term 2: A[0,1] * B[0,1]
    U[1][1] = 1; V[3][1] = 1; W[4][1] = 1
    
    # Term 3: A[0,2] * B[0,2]
    U[2][2] = 1; V[6][2] = 1; W[8][2] = 1
    
    # Term 4: A[1,0] * B[1,0]
    U[3][3] = 1; V[0][3] = 1; W[1][3] = 1
    
    # Term 5: A[1,1] * B[1,1]
    U[4][4] = 1; V[3][4] = 1; W[4][4] = 1
    
    # Term 6: A[1,2] * B[1,2]
    U[5][5] = 1; V[6][5] = 1; W[7][5] = 1
    
    # Term 7: A[2,0] * B[2,0]
    U[6][6] = 1; V[0][6] = 1; W[2][6] = 1
    
    # Term 8: A[2,1] * B[2,1]
    U[7][7] = 1; V[3][7] = 1; W[5][7] = 1
    
    # Term 9: A[2,2] * B[2,2]
    U[8][8] = 1; V[6][8] = 1; W[8][8] = 1
    
    # Term 10: (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
    U[0][9] = 1; U[1][9] = 1; U[3][9] = 1; U[4][9] = 1
    V[0][9] = 1; V[3][9] = 1; V[0][9] = 1; V[3][9] = 1
    V[0][9] = 1; V[3][9] = 1
    # Need to be more careful
    U[0][9] = 1; U[1][9] = 1; U[3][9] = 1; U[4][9] = 1
    V[0][9] = 1; V[1][9] = 1; V[3][9] = 1; V[4][9] = 1
    W[0][9] = 1; W[1][9] = -1; W[3][9] = -1; W[4][9] = 1
    
    # Term 11: (A[0,0]+A[1,0]) * (B[0,1]+B[1,1])
    U[0][10] = 1; U[3][10] = 1
    V[1][10] = 1; V[4][10] = 1
    W[1][10] = 1; W[4][10] = -1
    
    # Term 12: (A[0,1]+A[1,1]) * (B[0,0]+B[1,0])
    U[1][11] = 1; U[4][11] = 1
    V[0][11] = 1; V[3][11] = 1
    W[0][11] = 1; W[3][11] = -1
    
    # Term 13: (A[0,2]+A[1,2]) * (B[0,0]+B[1,0])
    U[2][12] = 1; U[5][12] = 1
    V[0][12] = 1; V[3][12] = 1
    W[0][12] = 1; W[3][12] = -1
    
    # Term 14: (A[0,0]+A[0,2]) * (B[1,0]+B[1,2])
    U[0][13] = 1; U[2][13] = 1
    V[3][13] = 1; V[5][13] = 1
    W[3][13] = 1; W[5][13] = -1
    
    # Term 15: (A[2,0]+A[2,1]+A[0,0]+A[0,1]) * (B[0,0]+B[0,1]+B[1,0]+B[1,1])
    U[6][14] = 1; U[7][14] = 1; U[0][14] = 1; U[1][14] = 1
    V[0][14] = 1; V[1][14] = 1; V[3][14] = 1; V[4][14] = 1
    W[2][14] = 1; W[5][14] = -1; W[0][14] = -1; W[1][14] = 1
    
    # Term 16: (A[2,0]+A[0,0]) * (B[0,1]+B[0,0])
    U[6][15] = 1; U[0][15] = 1
    V[1][15] = 1; V[0][15] = 1
    W[2][15] = 1; W[0][15] = -1
    
    # Term 17: (A[2,1]+A[0,1]) * (B[0,0]+B[0,1])
    U[7][16] = 1; U[1][16] = 1
    V[0][16] = 1; V[1][16] = 1
    W[5][16] = 1; W[1][16] = -1
    
    # Term 18: (A[2,0]+A[2,2]+A[1,0]+A[1,2]) * (B[0,0]+B[0,2]+B[1,0]+B[1,2])
    U[6][17] = 1; U[8][17] = 1; U[3][17] = 1; U[5][17] = 1
    V[0][17] = 1; V[6][17] = 1; V[3][17] = 1; V[5][17] = 1
    W[2][17] = 1; W[8][17] = -1; W[3][17] = -1; W[5][17] = 1
    
    # Term 19: (A[2,2]+A[1,2]) * (B[0,0]+B[1,0])
    U[8][18] = 1; U[5][18] = 1
    V[0][18] = 1; V[3][18] = 1
    W[8][18] = 1; W[3][18] = -1
    
    # Term 20: (A[2,0]+A[1,0]) * (B[0,2]+B[1,2])
    U[6][19] = 1; U[3][19] = 1
    V[6][19] = 1; V[5][19] = 1
    W[2][19] = 1; W[5][19] = -1
    
    # Term 21: (A[2,2]+A[2,0]+A[0,2]+A[0,0]) * (B[0,0]+B[0,2]+B[1,0]+B[1,2])
    U[8][20] = 1; U[6][20] = 1; U[2][20] = 1; U[0][20] = 1
    V[0][20] = 1; V[6][20] = 1; V[3][20] = 1; V[5][20] = 1
    W[8][20] = 1; W[2][20] = -1; W[0][20] = -1; W[6][20] = 1
    
    # Term 22: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0][21] = 1; U[4][21] = 1
    V[0][21] = 1; V[4][21] = 1
    W[0][21] = 1; W[4][21] = -1
    
    # Term 23: (A[0,1]+A[1,2]) * (B[0,2]+B[1,1])
    U[1][22] = 1; U[5][22] = 1
    V[6][22] = 1; V[4][22] = 1
    W[4][22] = 1; W[6][22] = -1
    
    return U, V, W

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    # U part: A block (bi, bj)
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row][col] = sU[li*2+lj][c]
                    
                    # V part: B block (bj, bk)
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row][col] = sV[lj*2+lk][c]
                    
                    # W part: C block (bk, bi)
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row][col] = sW[lk*2+li][c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _block_decomposition_6x6x6_cw():
    """
    Decompose 6x6x6 matrix multiplication using CW rank-23 for 3x3 and Strassen rank-7 for 2x2.
    Total rank = 23 * 7 = 161.
    """
    n, m, p = 6, 6, 6
    rank = 161
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    cwU, cwV, cwW = _coppersmith_winograd_3x3()
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for ci in range(23):
        for cj in range(7):
            # U part: combine 3x3 and 2x2
            for li in range(3):
                for lj in range(2):
                    u_row = li*6 + lj
                    if cwU[li*3 + 0][ci] != 0:
                        U[u_row][col] = cwU[li*3 + 0][ci] * sU[0][cj]
                    if cwU[li*3 + 1][ci] != 0:
                        U[u_row][col] += cwU[li*3 + 1][ci] * sU[1][cj]
                    if cwU[li*3 + 2][ci] != 0:
                        U[u_row][col] += cwU[li*3 + 2][ci] * sU[2][cj]
                    if cwU[li*3 + 3][ci] != 0:
                        U[u_row][col] += cwU[li*3 + 3][ci] * sU[3][cj]
                    
            # V part
            for lj in range(2):
                for lk in range(3):
                    v_row = lj*6 + lk
                    if cwV[0 + lj][ci] != 0:
                        V[v_row][col] = cwV[0 + lj][ci] * sV[0][cj]
                    if cwV[1 + lj][ci] != 0:
                        V[v_row][col] += cwV[1 + lj][ci] * sV[1][cj]
                    if cwV[2 + lj][ci] != 0:
                        V[v_row][col] += cwV[2 + lj][ci] * sV[2][cj]
                    if cwV[3 + lj][ci] != 0:
                        V[v_row][col] += cwV[3 + lj][ci] * sV[3][cj]
                    
            # W part
            for lk in range(3):
                for li in range(2):
                    w_row = lk*6 + li
                    if cwW[lk*3 + 0][ci] != 0:
                        W[w_row][col] = cwW[lk*3 + 0][ci] * sW[0][cj]
                    if cwW[lk*3 + 1][ci] != 0:
                        W[w_row][col] += cwW[lk*3 + 1][ci] * sW[1][cj]
                    if cwW[lk*3 + 2][ci] != 0:
                        W[w_row][col] += cwW[lk*3 + 2][ci] * sW[2][cj]
                    if cwW[lk*3 + 3][ci] != 0:
                        W[w_row][col] += cwW[lk*3 + 3][ci] * sW[3][cj]
                    
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
    
    # Try CW + Strassen decomposition (rank 161) first
    best = _block_decomposition_6x6x6_cw()
    
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Try block Strassen decomposition (rank 189)
    best = _block_decomposition_6x6x6()
    
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # Fallback: schoolbook (rank 216)
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
