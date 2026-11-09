"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def _strassen_2x2_2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[2][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _strassen_4x4_4x4():
    """Return (U, V, W) for 4x4x4 Strassen-like decomposition of rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2_2x2()
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for t_top in range(7):
        for t_inner in range(7):
            col = t_top * 7 + t_inner
            for a_block in range(4):
                coeff_a = U2[a_block][t_top]
                if coeff_a == 0:
                    continue
                for r in range(2):
                    for c in range(2):
                        br, bc = a_block // 2, a_block % 2
                        i = br * 2 + r
                        j = bc * 2 + c
                        a_idx = i * 4 + j
                        inner_idx = r * 2 + c
                        inner_coeff_a = U2[inner_idx][t_inner]
                        if inner_coeff_a == 0:
                            continue
                        U[a_idx][col] += coeff_a * inner_coeff_a
            
            for b_block in range(4):
                coeff_b = V2[b_block][t_top]
                if coeff_b == 0:
                    continue
                for r in range(2):
                    for c in range(2):
                        br, bc = b_block // 2, b_block % 2
                        j = br * 2 + r
                        k = bc * 2 + c
                        b_idx = j * 4 + k
                        inner_idx = r * 2 + c
                        inner_coeff_b = V2[inner_idx][t_inner]
                        if inner_coeff_b == 0:
                            continue
                        V[b_idx][col] += coeff_b * inner_coeff_b
            
            for c_block in range(4):
                coeff_c = W2[c_block][t_top]
                if coeff_c == 0:
                    continue
                for r in range(2):
                    for c in range(2):
                        br, bc = c_block // 2, c_block % 2
                        k = br * 2 + r
                        i = bc * 2 + c
                        c_idx = k * 4 + i
                        inner_idx = r * 2 + c
                        inner_coeff_c = W2[inner_idx][t_inner]
                        if inner_coeff_c == 0:
                            continue
                        W[c_idx][col] += coeff_c * inner_coeff_c
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    U4, V4, W4 = _strassen_4x4_4x4()
    
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for half in range(2):
        for r in range(49):
            col = half * 49 + r
            for i in range(n*m):
                U[i][col] = U4[i][r]
            for j in range(m):
                for k in range(4):
                    b_idx = j * p + (half * 4 + k)
                    v4_idx = j * 4 + k
                    V[b_idx][col] = V4[v4_idx][r]
            for k in range(4):
                for i in range(n):
                    w_idx = k * n + (half * 4 + i)
                    w4_idx = k * n + i
                    W[w_idx][col] = W4[w4_idx][r]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
