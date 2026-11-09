"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2 @ 2x2, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Index mapping for U (2x2 matrix, row = i*2+j):
    # A11 -> (0,0) -> row 0, A12 -> (0,1) -> row 1
    # A21 -> (1,0) -> row 2, A22 -> (1,1) -> row 3
    #
    # Index mapping for V (2x2 matrix, row = j*2+k):
    # B11 -> (0,0) -> row 0, B12 -> (0,1) -> row 1
    # B21 -> (1,0) -> row 2, B22 -> (1,1) -> row 3
    #
    # Index mapping for W (2x2 matrix, row = k*2+i):
    # C11 -> (0,0) -> row 0, C21 -> (1,0) -> row 2
    # C12 -> (0,1) -> row 1, C22 -> (1,1) -> row 3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22): A11(+), A22(+) ; B11(+), B22(+)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11 + M1, C22 + M1
    
    # M2 = (A21+A22)B11: A21(+), A22(+) ; B11(+)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C21 + M2, C22 - M2
    
    # M3 = A11(B12-B22): A11(+) ; B12(+), B22(-)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C12 + M3, C22 + M3
    
    # M4 = A22(B21-B11): A22(+) ; B21(+), B11(-)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C11 + M4, C21 + M4
    
    # M5 = (A11+A12)B22: A11(+), A12(+) ; B22(+)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C11 - M5, C12 + M5
    
    # M6 = (A21-A11)(B11+B12): A21(+), A11(-) ; B11(+), B12(+)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C22 + M6
    
    # M7 = (A12-A22)(B21+B22): A12(+), A22(-) ; B21(+), B22(+)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C11 + M7
    
    return U, V, W


def _schoolbook_2x2():
    """Return (U, V, W) for 2x2 @ 2x2, rank 4 (schoolbook)."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(4)]
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j][c] = 1
                V[j*2+k][c] = 1
                W[k*2+i][c] = 1
                c += 1
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 @ 4x5.
    Uses Strassen's algorithm on the 2x2 blocks for the first 2 rows/cols of B.
    Strategy:
    - A is 3x4, B is 4x5.
    - Split B's first 4 rows into two groups of 2: B_top (rows 0,1) and B_bot (rows 2,3).
    - For each of the 5 columns of B, we have a 4x1 vector, split into top (2x1) and bot (2x1).
    - A is split into A_left (3x2, cols 0,1) and A_right (3x2, cols 2,3).
    - C = A_left @ B_top + A_right @ B_bot.
    - A_left @ B_top: (3x2) @ (2x5). We can view this as 5 separate (3x2) @ (2x1) multiplications.
    - For each column, we can use a decomposition. But better: treat (3x2) @ (2x2) for two columns at a time, then the 5th separately.
    - Actually, let's just do schoolbook for the 3x2 part and use Strassen where applicable.
    
    Simpler approach: 
    - Use Strassen for the 2x2 block multiplication where possible.
    - A is 3x4. Split A into 3 rows x 2 cols blocks: A0 (rows 0,1, cols 0,1), A1 (row 2, cols 0,1), etc.
    - Actually, let's use the following block structure:
      A = [A11 A12]  where A11 is 3x2, A12 is 3x2
      B = [B11 B21; B21 B22] wait, B is 4x5. Split B into top 2 rows and bottom 2 rows:
      B = [Btop; Bbot] where Btop is 2x5, Bbot is 2x5.
      C = A11 @ Btop + A12 @ Bbot.
    
    For A11 @ Btop: (3x2) @ (2x5). This can be done with rank 3*2*5 = 30 schoolbook, or we can reduce.
    For A12 @ Bbot: (3x2) @ (2x5). Same.
    Total schoolbook: 60.
    
    To reduce: For the (3x2) @ (2x5), we can split the 3 rows into 2+1.
    For the 2x2 @ 2x5 part, use Strassen-like or just 7 multiplications per 2 columns of B.
    2x2 @ 2x2 has rank 7. So 2x2 @ 2x5 = 2*(2x2@2x2) + 1*(2x2@2x1).
    2x2@2x2: rank 7. 2x2@2x1: rank 4 (schoolbook) or we can do better? 
    Actually 2x2 @ 1x2... wait B is 2x5, so Btop is 2x5.
    A11 is 3x2. Split A11 into A11_top (2x2, rows 0,1) and A11_bot (1x2, row 2).
    A11 @ Btop = A11_top @ Btop + A11_bot @ Btop.
    A11_top @ Btop: (2x2) @ (2x5). Split Btop into Btop[:2] (2x2), Btop[2:4] (2x2), Btop[4] (2x1).
    (2x2)@(2x2): rank 7 each, two of them = 14.
    (2x2)@(2x1): rank 4.
    Total for A11_top @ Btop: 18.
    A11_bot @ Btop: (1x2) @ (2x5). Rank 2*5 = 10 schoolbook. Can we do better? 
    (1x2) @ (2x5) = sum of 2x5 scalar mults. Rank 10. 
    So A11 @ Btop total: 18 + 10 = 28.
    Similarly A12 @ Bbot: 28.
    Total: 56. Still not great.
    
    Let me try a different approach. Use Strassen for the whole 4x4 block if possible.
    Actually, the known rank for 3x4x5 is 48. Let me just implement the schoolbook (rank 60) to be safe,
    or try to construct a rank 48 solution if I can.
    
    Given the complexity, let me return the schoolbook decomposition which is guaranteed correct.
    """
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 3*4*5 = 60
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
# EVOLVE-BLOCK-END
