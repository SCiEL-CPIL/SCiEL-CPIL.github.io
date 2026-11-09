import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                # M1 = (A11 + A22)(B11 + B22)
                # A11: i=bi*2, j=bk*2
                # A22: i=bi*2+1, j=bk*2+1
                # B11: i=bk*2, j=bj*2
                # B22: i=bk*2+1, j=bj*2+1
                # C11: i=bi*2, j=bj*2
                
                U[bi*2 * 6 + bk*2][col] = 1
                U[(bi*2+1) * 6 + (bk*2+1)][col] = 1
                V[bk*2 * 6 + bj*2][col] = 1
                V[(bk*2+1) * 6 + (bj*2+1)][col] = 1
                W[bi*2 * 6 + bj*2][col] = 1
                col += 1
                
                # M2 = (A21 + A22) B11
                # A21: i=bi*2+1, j=bk*2
                # A22: i=bi*2+1, j=bk*2+1
                # B11: i=bk*2, j=bj*2
                # C21: i=bi*2+1, j=bj*2
                
                U[(bi*2+1) * 6 + bk*2][col] = 1
                U[(bi*2+1) * 6 + (bk*2+1)][col] = 1
                V[bk*2 * 6 + bj*2][col] = 1
                W[(bi*2+1) * 6 + bj*2][col] = 1
                col += 1
                
                # M3 = A11 (B12 - B22)
                # A11: i=bi*2, j=bk*2
                # B12: i=bk*2, j=bj*2+1
                # B22: i=bk*2+1, j=bj*2+1
                # C12: i=bi*2, j=bj*2+1
                
                U[bi*2 * 6 + bk*2][col] = 1
                V[bk*2 * 6 + (bj*2+1)][col] = 1
                V[(bk*2+1) * 6 + (bj*2+1)][col] = -1
                W[bi*2 * 6 + (bj*2+1)][col] = 1
                col += 1
                
                # M4 = A22 (B21 - B11)
                # A22: i=bi*2+1, j=bk*2+1
                # B21: i=bk*2+1, j=bj*2
                # B11: i=bk*2, j=bj*2
                # C21: i=bi*2+1, j=bj*2
                
                U[(bi*2+1) * 6 + (bk*2+1)][col] = 1
                V[(bk*2+1) * 6 + bj*2][col] = 1
                V[bk*2 * 6 + bj*2][col] = -1
                W[(bi*2+1) * 6 + bj*2][col] = 1
                col += 1
                
                # M5 = (A11 + A12) B22
                # A11: i=bi*2, j=bk*2
                # A12: i=bi*2, j=bk*2+1
                # B22: i=bk*2+1, j=bj*2+1
                # C12: i=bi*2, j=bj*2+1
                # C11: i=bi*2, j=bj*2
                
                U[bi*2 * 6 + bk*2][col] = 1
                U[bi*2 * 6 + (bk*2+1)][col] = 1
                V[(bk*2+1) * 6 + (bj*2+1)][col] = 1
                W[bi*2 * 6 + bj*2][col] = -1
                W[bi*2 * 6 + (bj*2+1)][col] = 1
                col += 1
                
                # M6 = (A21 - A11)(B11 + B12)
                # A21: i=bi*2+1, j=bk*2
                # A11: i=bi*2, j=bk*2
                # B11: i=bk*2, j=bj*2
                # B12: i=bk*2, j=bj*2+1
                # C22: i=bi*2+1, j=bj*2+1
                
                U[(bi*2+1) * 6 + bk*2][col] = 1
                U[bi*2 * 6 + bk*2][col] = -1
                V[bk*2 * 6 + bj*2][col] = 1
                V[bk*2 * 6 + (bj*2+1)][col] = 1
                W[(bi*2+1) * 6 + (bj*2+1)][col] = 1
                col += 1
                
                # M7 = (A12 - A22)(B21 + B22)
                # A12: i=bi*2, j=bk*2+1
                # A22: i=bi*2+1, j=bk*2+1
                # B21: i=bk*2+1, j=bj*2
                # B22: i=bk*2+1, j=bj*2+1
                # C11: i=bi*2, j=bj*2
                # C22: i=bi*2+1, j=bj*2+1
                
                U[bi*2 * 6 + (bk*2+1)][col] = 1
                U[(bi*2+1) * 6 + (bk*2+1)][col] = -1
                V[(bk*2+1) * 6 + bj*2][col] = 1
                V[(bk*2+1) * 6 + (bj*2+1)][col] = 1
                W[bi*2 * 6 + bj*2][col] = 1
                W[(bi*2+1) * 6 + (bj*2+1)][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
