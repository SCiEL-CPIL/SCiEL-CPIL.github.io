import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 8 groups: 2 j-blocks, 4 k-blocks
    for j_b in range(2):
        for k_b in range(4):
            j_s = j_b * 2
            k_s = k_b * 2
            
            # Strassen steps
            # M1: (X11+X22)(Y11+Y22)
            U[0 * m + j_s][col] = 1
            U[1 * m + j_s + 1][col] = 1
            V[j_s * p + k_s][col] = 1
            V[(j_s + 1) * p + k_s + 1][col] = 1
            # P11 += M1
            W[k_s * n + 0][col] += 1
            # P22 += M1
            W[(k_s + 1) * n + 1][col] += 1
            col += 1
            
            # M2: (X21+X22)Y11
            U[1 * m + j_s][col] = 1
            U[1 * m + j_s + 1][col] = 1
            V[j_s * p + k_s][col] = 1
            # P21 += M2
            W[k_s * n + 1][col] += 1
            # P22 -= M2
            W[(k_s + 1) * n + 1][col] -= 1
            col += 1
            
            # M3: X11(Y12-Y22)
            U[0 * m + j_s][col] = 1
            V[j_s * p + k_s + 1][col] = 1
            V[(j_s + 1) * p + k_s + 1][col] = -1
            # P12 += M3
            W[(k_s + 1) * n + 0][col] += 1
            # P22 += M3
            W[(k_s + 1) * n + 1][col] += 1
            col += 1
            
            # M4: X22(Y21-Y11)
            U[1 * m + j_s + 1][col] = 1
            V[(j_s + 1) * p + k_s][col] = 1
            V[j_s * p + k_s][col] = -1
            # P11 += M4
            W[k_s * n + 0][col] += 1
            # P21 += M4
            W[k_s * n + 1][col] += 1
            col += 1
            
            # M5: (X11+X12)Y22
            U[0 * m + j_s][col] = 1
            U[0 * m + j_s + 1][col] = 1
            V[(j_s + 1) * p + k_s + 1][col] = 1
            # P11 -= M5
            W[k_s * n + 0][col] -= 1
            # P12 += M5
            W[(k_s + 1) * n + 0][col] += 1
            col += 1
            
            # M6: (X21-X11)(Y11+Y12)
            U[1 * m + j_s][col] = 1
            U[0 * m + j_s][col] = -1
            V[j_s * p + k_s][col] = 1
            V[j_s * p + k_s + 1][col] = 1
            # P22 += M6
            W[(k_s + 1) * n + 1][col] += 1
            col += 1
            
            # M7: (X12-X22)(Y21+Y22)
            U[0 * m + j_s + 1][col] = 1
            U[1 * m + j_s + 1][col] = -1
            V[(j_s + 1) * p + k_s][col] = 1
            V[(j_s + 1) * p + k_s + 1][col] = 1
            # P11 += M7
            W[k_s * n + 0][col] += 1
            # P22 += M7
            W[(k_s + 1) * n + 1][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
