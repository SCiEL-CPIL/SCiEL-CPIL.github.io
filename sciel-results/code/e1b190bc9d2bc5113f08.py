"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    # Standard Strassen: 7 multiplications
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0,0] = 1; U[3,0] = 1  # A00 + A11
    V[0,0] = 1; V[3,0] = 1  # B00 + B11
    W[0,0] = 1; W[3,3] = 1  # contributes to C00 and C11
    # Actually W[k*n+i, c] so W[0*2+0,0]=1 for C[0,0], W[1*2+1,0]=1 for C[1,1]
    W[0,0] = 1; W[3,0] = 1
    
    # Let me redo this properly.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1   # C00 += M1
    W[3,0] = 1   # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1   # C10 += M2
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1   # C01 += M3
    W[3,2] = -1  # C11 -= M3
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = -1  # C00 -= M4
    W[2,3] = 1   # C10 += M4
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[1,4] = 1   # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[0,5] = 1   # C00 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[3,6] = 1   # C11 += M7
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    
    # Try block decomposition: split 4x5 * 5x5
    # Split A (4x5) into top 2 rows, bottom 2 rows
    # Split B (5x5) into left 2 cols, right 3 cols (and top 2 rows, bottom 3 rows)
    # This gives subproblems that we can handle
    
    # Actually, let's try a simpler approach: use the tensor rank directly
    # For 4x5x5, try to find a decomposition with rank < 100
    
    # The matrix multiplication tensor has rank at most n*m*p = 100
    # We can try to reduce by using linearity
    
    # One approach: if we can factor some dimensions, we can reduce rank
    # 4 = 2+2, 5 = 2+3
    
    # Let's try: C = A*B where A is 4x5, B is 5x5
    # Split A into A_top (2x5) and A_bot (2x5)
    # Split B into B_left (5x2) and B_right (5x3)
    # C_top_left = A_top * B_left (2x2)
    # C_top_right = A_top * B_right (2x3)
    # C_bot_left = A_bot * B_left (2x2)
    # C_bot_right = A_bot * B_right (2x3)
    
    # For 2x2: use Strassen rank 7
    # For 2x3: need to find rank
    
    # For 2x3x2 (A:2x3, B:3x2, C:2x2): schoolbook rank 12, can we do better?
    # For 2x3x3 (A:2x3, B:3x3, C:2x3): schoolbook rank 18
    
    # This is getting complex. Let me just return schoolbook for now since
    # finding a provably lower rank for 4x5x5 is non-trivial.
    
    # Actually, let me try a simple optimization: if any dimension is 1, 
    # we can reduce rank. But 4,5,5 are all > 1.
    
    # Let me try to verify if we can do better than 100 by checking
    # known bounds. The tensor rank of [4,5,5] is at most 100.
    # Known lower bounds are weaker.
    
    # Given the time constraint and complexity, return schoolbook.
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
