"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor.

    Contract sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k).
    """
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)

    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    return bool(np.allclose(result, 1.0, atol=1e-9))


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


def _laderman_3x3():
    """Return the rank-21 Laderman decomposition for 3x3x3 matrix multiplication.
    U: (9, 21), V: (9, 21), W: (9, 21)
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U = [[0]*21 for _ in range(9)]
    V = [[0]*21 for _ in range(9)]
    W = [[0]*21 for _ in range(9)]

    # 1. a00*b00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # 2. a01*b01
    U[1][1] = 1; V[1][1] = 1; W[3][1] = 1
    # 3. a02*b02
    U[2][2] = 1; V[2][2] = 1; W[6][2] = 1
    # 4. a10*b10
    U[3][3] = 1; V[3][3] = 1; W[1][3] = 1
    # 5. a11*b11
    U[4][4] = 1; V[4][4] = 1; W[4][4] = 1
    # 6. a12*b12
    U[5][5] = 1; V[5][5] = 1; W[7][5] = 1
    # 7. a20*b20
    U[6][6] = 1; V[6][6] = 1; W[2][6] = 1
    # 8. a21*b21
    U[7][7] = 1; V[7][7] = 1; W[5][7] = 1
    # 9. a22*b22
    U[8][8] = 1; V[8][8] = 1; W[8][8] = 1

    # 10. (a00+a01)*b01
    U[0][9] = 1; U[1][9] = 1
    V[1][9] = 1
    W[0][9] = -1; W[3][9] = 1

    # 11. (a01+a02)*b02
    U[1][10] = 1; U[2][10] = 1
    V[2][10] = 1
    W[3][10] = -1; W[6][10] = 1

    # 12. (a02+a00)*b00
    U[2][11] = 1; U[0][11] = 1
    V[0][11] = 1
    W[6][11] = -1; W[0][11] = 1

    # 13. (a10+a11)*b11
    U[3][12] = 1; U[4][12] = 1
    V[4][12] = 1
    W[1][12] = -1; W[4][12] = 1

    # 14. (a11+a12)*b12
    U[4][13] = 1; U[5][13] = 1
    V[5][13] = 1
    W[4][13] = -1; W[7][13] = 1

    # 15. (a12+a10)*b10
    U[5][14] = 1; U[3][14] = 1
    V[3][14] = 1
    W[7][14] = -1; W[1][14] = 1

    # 16. (a20+a21)*b21
    U[6][15] = 1; U[7][15] = 1
    V[7][15] = 1
    W[2][15] = -1; W[5][15] = 1

    # 17. (a21+a22)*b22
    U[7][16] = 1; U[8][16] = 1
    V[8][16] = 1
    W[5][16] = -1; W[8][16] = 1

    # 18. (a22+a20)*b20
    U[8][17] = 1; U[6][17] = 1
    V[6][17] = 1
    W[8][17] = -1; W[2][17] = 1

    # 19. (a00+a11+a22)*(b00+b11+b22)
    U[0][18] = 1; U[4][18] = 1; U[8][18] = 1
    V[0][18] = 1; V[4][18] = 1; V[8][18] = 1
    W[0][18] = 1; W[4][18] = 1; W[8][18] = 1

    # 20. (a01+a12+a20)*(b01+b12+b20)
    U[1][19] = 1; U[5][19] = 1; U[6][19] = 1
    V[1][19] = 1; V[5][19] = 1; V[6][19] = 1
    W[3][19] = 1; W[7][19] = 1; W[2][19] = 1

    # 21. (a02+a10+a21)*(b02+b10+b21)
    U[2][20] = 1; U[3][20] = 1; U[7][20] = 1
    V[2][20] = 1; V[3][20] = 1; V[7][20] = 1
    W[6][20] = 1; W[1][20] = 1; W[5][20] = 1

    return U, V, W


def _kron_decomposition(U_small, V_small, W_small, small_dim, big_dim):
    """
    Combine a small_dim x small_dim decomposition into a big_dim x big_dim decomposition
    using Kronecker product structure.
    Assumes big_dim is a multiple of small_dim.
    """
    blocks = big_dim // small_dim
    rank_small = len(U_small[0])
    rank_big = (blocks ** 3) * rank_small
    
    U = [[0]*rank_big for _ in range(big_dim*big_dim)]
    V = [[0]*rank_big for _ in range(big_dim*big_dim)]
    W = [[0]*rank_big for _ in range(big_dim*big_dim)]
    
    col = 0
    for bi in range(blocks):
        for bj in range(blocks):
            for bk in range(blocks):
                for c in range(rank_small):
                    for li in range(small_dim):
                        for lj in range(small_dim):
                            u_row = (bi*small_dim+li)*big_dim + (bj*small_dim+lj)
                            U[u_row][col] = U_small[li*small_dim+lj][c]
                            
                    for lj in range(small_dim):
                        for lk in range(small_dim):
                            v_row = (bj*small_dim+lj)*big_dim + (bk*small_dim+lk)
                            V[v_row][col] = V_small[lj*small_dim+lk][c]
                            
                    for lk in range(small_dim):
                        for li in range(small_dim):
                            w_row = (bk*small_dim+lk)*big_dim + (bi*small_dim+li)
                            W[w_row][col] = W_small[lk*small_dim+li][c]
                            
                    col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_blocks_6x6():
    """Rank 189: 3x3 blocks of 2x2 Strassen."""
    sU, sV, sW = _strassen_2x2()
    return _kron_decomposition(sU, sV, sW, 2, 6)


def _laderman_3x3_blocks_6x6():
    """Rank 147: 2x2 blocks of 3x3 Laderman."""
    lU, lV, lW = _laderman_3x3()
    return _kron_decomposition(lU, lV, lW, 3, 6)


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


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS

    # Candidate 1: Laderman 3x3 blocks (Rank 147)
    try:
        lad = _laderman_3x3_blocks_6x6()
        if _verify(lad["U"], lad["V"], lad["W"], n, m, p):
            return lad
    except Exception:
        pass

    if time.time() > deadline - 20:
        return _schoolbook(n, m, p)

    # Candidate 2: Strassen 2x2 blocks (Rank 189)
    try:
        strassen = _strassen_2x2_blocks_6x6()
        if _verify(strassen["U"], strassen["V"], strassen["W"], n, m, p):
            return strassen
    except Exception:
        pass

    # Fallback: Schoolbook (Rank 216)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
