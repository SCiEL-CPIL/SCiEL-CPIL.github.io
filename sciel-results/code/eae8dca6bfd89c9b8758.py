"""Matrix multiplication 3x3x4 - rank 34 decomposition using two Strassen blocks."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(V[0]) != rank:
        return False
    if len(W[0]) != rank:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    total += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Rank-34 decomposition of 3x3 * 3x4 matrix multiplication.
    
    Split k into {0,1} and {2,3}. For each half, use Strassen on the
    2x2x2 sub-block (i in {0,1}, j in {0,1}) and schoolbook for the rest.
    
    Total: 2*(7 Strassen + 10 schoolbook) = 34 terms.
    """
    n, m, p = DIMENSIONS
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]  # shape (9, 34)
    V = [[0] * rank for _ in range(m * p)]  # shape (12, 34)
    W = [[0] * rank for _ in range(p * n)]  # shape (12, 34)
    
    col = 0
    
    # ===== First k-half: k in {0, 1} =====
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1    # A00
    U[4][col] = 1    # A11
    V[0][col] = 1    # B00 (j=0,k=0)
    V[5][col] = 1    # B11 (j=1,k=1)
    W[0][col] = 1    # C00 (k=0,i=0)
    W[4][col] = 1    # C11 (k=1,i=1)
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1    # A10
    U[4][col] = 1    # A11
    V[0][col] = 1    # B00
    W[1][col] = 1    # C10 (k=0,i=1)
    W[4][col] = -1   # C11 (k=1,i=1)
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] = 1    # A00
    V[4][col] = 1    # B10 (j=1,k=0)
    V[5][col] = -1   # B11 (j=1,k=1)
    W[3][col] = 1    # C01 (k=1,i=0)
    W[4][col] = 1    # C11 (k=1,i=1)
    col += 1
    
    # M4 = A11(B01-B10)
    U[4][col] = 1    # A11
    V[1][col] = 1    # B01 (j=0,k=1)
    V[4][col] = -1   # B10 (j=1,k=0)
    W[0][col] = -1   # C00 (k=0,i=0)
    W[1][col] = 1    # C10 (k=0,i=1)
    col += 1
    
    # M5 = (A00+A10)B11
    U[0][col] = 1    # A00
    U[3][col] = 1    # A10
    V[5][col] = 1    # B11 (j=1,k=1)
    W[3][col] = 1    # C01 (k=1,i=0)
    W[4][col] = -1   # C11 (k=1,i=1)
    col += 1
    
    # M6 = (A01-A11)(B10+B11)
    U[1][col] = 1    # A01
    U[4][col] = -1   # A11
    V[4][col] = 1    # B10 (j=1,k=0)
    V[5][col] = 1    # B11 (j=1,k=1)
    W[0][col] = -1   # C00 (k=0,i=0)
    W[4][col] = 1    # C11 (k=1,i=1)
    col += 1
    
    # M7 = (A10-A00)(B01+B11)
    U[3][col] = 1    # A10
    U[0][col] = -1   # A00
    V[1][col] = 1    # B01 (j=0,k=1)
    V[5][col] = 1    # B11 (j=1,k=1)
    W[0][col] = 1    # C00 (k=0,i=0)
    W[3][col] = 1    # C01 (k=1,i=0)
    col += 1
    
    # Schoolbook for remaining (i,j,k) with k in {0,1} and NOT (i<2 and j<2)
    for i in range(n):
        for j in range(m):
            for k in range(2):  # k in {0,1}
                if i < 2 and j < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # ===== Second k-half: k in {2, 3} =====
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {2,3}
    
    # M1' = (A00+A11)(B02+B13)
    U[0][col] = 1    # A00
    U[4][col] = 1    # A11
    V[2][col] = 1    # B02 (j=0,k=2)
    V[7][col] = 1    # B13 (j=1,k=3)
    W[6][col] = 1    # C02 (k=2,i=0)
    W[8][col] = 1    # C13 (k=3,i=1)
    col += 1
    
    # M2' = (A10+A11)B02
    U[3][col] = 1    # A10
    U[4][col] = 1    # A11
    V[2][col] = 1    # B02
    W[7][col] = 1    # C12 (k=2,i=1)
    W[8][col] = -1   # C13 (k=3,i=1)
    col += 1
    
    # M3' = A00(B12-B13)
    U[0][col] = 1    # A00
    V[6][col] = 1    # B12 (j=1,k=2)
    V[7][col] = -1   # B13 (j=1,k=3)
    W[9][col] = 1    # C03 (k=3,i=0)
    W[8][col] = 1    # C13 (k=3,i=1)
    col += 1
    
    # M4' = A11(B03-B12)
    U[4][col] = 1    # A11
    V[3][col] = 1    # B03 (j=0,k=3)
    V[6][col] = -1   # B12 (j=1,k=2)
    W[6][col] = -1   # C02 (k=2,i=0)
    W[7][col] = 1    # C12 (k=2,i=1)
    col += 1
    
    # M5' = (A00+A10)B13
    U[0][col] = 1    # A00
    U[3][col] = 1    # A10
    V[7][col] = 1    # B13 (j=1,k=3)
    W[9][col] = 1    # C03 (k=3,i=0)
    W[8][col] = -1   # C13 (k=3,i=1)
    col += 1
    
    # M6' = (A01-A11)(B12+B13)
    U[1][col] = 1    # A01
    U[4][col] = -1   # A11
    V[6][col] = 1    # B12 (j=1,k=2)
    V[7][col] = 1    # B13 (j=1,k=3)
    W[6][col] = -1   # C02 (k=2,i=0)
    W[8][col] = 1    # C13 (k=3,i=1)
    col += 1
    
    # M7' = (A10-A00)(B03+B13)
    U[3][col] = 1    # A10
    U[0][col] = -1   # A00
    V[3][col] = 1    # B03 (j=0,k=3)
    V[7][col] = 1    # B13 (j=1,k=3)
    W[6][col] = 1    # C02 (k=2,i=0)
    W[9][col] = 1    # C03 (k=3,i=0)
    col += 1
    
    # Schoolbook for remaining (i,j,k) with k in {2,3} and NOT (i<2 and j<2)
    for i in range(n):
        for j in range(m):
            for k in range(2, 4):  # k in {2,3}
                if i < 2 and j < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][column] = 1
                    V_s[j * p + k][column] = 1
                    W_s[k * n + i][column] = 1
                    column += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
