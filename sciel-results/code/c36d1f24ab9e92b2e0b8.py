"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # U, V, W shapes (4, 7)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A11(B21-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: (A11-A22)B22
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A12+A22)(B22-B11)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A11-A12)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A21-A11)(B11+B21)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _winograd_3x3():
    """Return the rank-15 Winograd decomposition for 3x3x3 matrix multiplication."""
    # U, V, W shapes (9, 15)
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(9)]
    W = [[0]*15 for _ in range(9)]
    
    # Indices: i*3+j for U, j*3+k for V, k*3+i for W
    
    # 1. (a00+a11+a22)(b00+b11+b22)
    U[0][0] = 1; U[4][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[4][0] = 1; V[8][0] = 1
    W[0][0] = 1; W[4][0] = 1; W[8][0] = 1
    
    # 2. (a01+a11+a21)(b10+b11+b12)
    U[1][1] = 1; U[4][1] = 1; U[7][1] = 1
    V[3][1] = 1; V[4][1] = 1; V[5][1] = 1
    W[1][1] = 1; W[4][1] = 1; W[7][1] = 1
    
    # 3. (a02+a12+a22)(b20+b21+b22)
    U[2][2] = 1; U[5][2] = 1; U[8][2] = 1
    V[6][2] = 1; V[7][2] = 1; V[8][2] = 1
    W[2][2] = 1; W[5][2] = 1; W[8][2] = 1
    
    # 4. (a01+a11)(b10+b11)
    U[1][3] = 1; U[4][3] = 1
    V[3][3] = 1; V[4][3] = 1
    W[1][3] = 1; W[4][3] = 1
    
    # 5. (a02+a12)(b20+b21)
    U[2][4] = 1; U[5][4] = 1
    V[6][4] = 1; V[7][4] = 1
    W[2][4] = 1; W[5][4] = 1
    
    # 6. (a01+a10)(b10+b00)
    U[1][5] = 1; U[3][5] = 1
    V[3][5] = 1; V[0][5] = 1
    W[1][5] = 1; W[3][5] = 1
    
    # 7. (a02+a20)(b20+b00)
    U[2][6] = 1; U[6][6] = 1
    V[6][6] = 1; V[0][6] = 1
    W[2][6] = 1; W[6][6] = 1
    
    # 8. (a02+a12)(b21+b11)
    U[2][7] = 1; U[5][7] = 1
    V[7][7] = 1; V[4][7] = 1
    W[2][7] = 1; W[5][7] = 1
    
    # 9. (a01+a20)(b10+b00)
    U[1][8] = 1; U[6][8] = 1
    V[3][8] = 1; V[0][8] = 1
    W[1][8] = 1; W[6][8] = 1
    
    # 10. (a11+a21)(b11+b12)
    U[4][9] = 1; U[7][9] = 1
    V[4][9] = 1; V[5][9] = 1
    W[4][9] = 1; W[7][9] = 1
    
    # 11. (a12+a22)(b21+b22)
    U[5][10] = 1; U[8][10] = 1
    V[7][10] = 1; V[8][10] = 1
    W[5][10] = 1; W[8][10] = 1
    
    # 12. (a11+a12)(b11+b21)
    U[4][11] = 1; U[5][11] = 1
    V[4][11] = 1; V[7][11] = 1
    W[4][11] = 1; W[5][11] = 1
    
    # 13. (a21+a22)(b12+b22)
    U[7][12] = 1; U[8][12] = 1
    V[5][12] = 1; V[8][12] = 1
    W[7][12] = 1; W[8][12] = 1
    
    # 14. (a01+a11+a21-a10-a12-a20)(b10+b11+b12-b00-b20-b21)
    U[1][13] = 1; U[4][13] = 1; U[7][13] = 1
    U[3][13] = -1; U[5][13] = -1; U[6][13] = -1
    V[3][13] = 1; V[4][13] = 1; V[5][13] = 1
    V[0][13] = -1; V[6][13] = -1; V[7][13] = -1
    W[1][13] = 1; W[4][13] = 1; W[7][13] = 1
    W[3][13] = -1; W[5][13] = -1; W[6][13] = -1
    
    # 15. (a02+a12+a22-a01-a20-a11)(b20+b21+b22-b02-b12-b11)
    U[2][14] = 1; U[5][14] = 1; U[8][14] = 1
    U[1][14] = -1; U[6][14] = -1; U[4][14] = -1
    V[6][14] = 1; V[7][14] = 1; V[8][14] = 1
    V[2][14] = -1; V[5][14] = -1; V[4][14] = -1
    W[2][14] = 1; W[5][14] = 1; W[8][14] = 1
    W[1][14] = -1; W[6][14] = -1; W[4][14] = -1

    return U, V, W

def construct():
    n, m, p = 6, 6, 6
    rank = 7 * 15
    U_total = [[0]*rank for _ in range(36)]
    V_total = [[0]*rank for _ in range(36)]
    W_total = [[0]*rank for _ in range(36)]
    
    sU, sV, sW = _strassen_2x2()
    wU, wV, wW = _winograd_3x3()
    
    for k in range(7):
        for w in range(15):
            col = k * 15 + w
            for bi in range(2):
                for bj in range(2):
                    su = sU[bi*2+bj][k]
                    if su == 0: continue
                    for li in range(3):
                        for lj in range(3):
                            u_row = (bi*3+li)*6 + (bj*3+lj)
                            u_val = su * wU[li*3+lj][w]
                            if u_val != 0:
                                U_total[u_row][col] = u_val
            for bj in range(2):
                for bk in range(2):
                    sv = sV[bj*2+bk][k]
                    if sv == 0: continue
                    for lj in range(3):
                        for lk in range(3):
                            v_row = (bj*3+lj)*6 + (bk*3+lk)
                            v_val = sv * wV[lj*3+lk][w]
                            if v_val != 0:
                                V_total[v_row][col] = v_val
            for bk in range(2):
                for bi in range(2):
                    sw = sW[bk*2+bi][k]
                    if sw == 0: continue
                    for lk in range(3):
                        for li in range(3):
                            w_row = (bk*3+lk)*6 + (bi*3+li)
                            w_val = sw * wW[lk*3+li][w]
                            if w_val != 0:
                                W_total[w_row][col] = w_val

    return {"U": U_total, "V": V_total, "W": W_total}

# EVOLVE-BLOCK-END
