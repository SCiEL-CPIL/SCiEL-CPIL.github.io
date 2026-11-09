"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    
    # Rank calculation
    # Strassen for 2x2x2 blocks.
    # A is 2x3. Split A into A_left (2x2) and A_right (2x1).
    # B is 3x8. Split B into B_top (2x8) and B_bottom (1x8).
    # A_left * B_top: 2x2 * 2x8.
    # Split B_top into 4 blocks of 2x2.
    # Each 2x2*2x2 uses 7 multiplications.
    # Total Strassen multiplications: 4 * 7 = 28.
    # A_right * B_bottom: 2x1 * 1x8.
    # This is 16 multiplications.
    # Total rank = 28 + 16 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    start_time = time.time()
    deadline = start_time + 120.0
    
    column = 0
    
    # Strassen Part
    # A_left indices: i in 0..1, j in 0..1
    # B_top indices: j in 0..1, k in 0..7
    # We process 4 blocks of k: [0,1], [2,3], [4,5], [6,7]
    
    for c in range(4):
        k0 = 2 * c
        k1 = 2 * c + 1
        
        # Strassen coefficients for A (2x2)
        # M1: (A00 + A11) -> U[0,0]=1, U[1,1]=1
        # M2: (A10 + A11) -> U[1,0]=1, U[1,1]=1
        # M3: A00 -> U[0,0]=1
        # M4: A11 -> U[1,1]=1
        # M5: (A00 + A01) -> U[0,0]=1, U[0,1]=1
        # M6: (A10 - A00) -> U[1,0]=1, U[0,0]=-1
        # M7: (A01 - A11) -> U[0,1]=1, U[1,1]=-1
        
        # Strassen coefficients for B (2x2)
        # M1: (B00 + B11) -> V[0,k0]=1, V[1,k1]=1
        # M2: B00 -> V[0,k0]=1
        # M3: (B10 - B11) -> V[1,k0]=1, V[1,k1]=-1
        # M4: (B01 - B00) -> V[0,k1]=1, V[0,k0]=-1
        # M5: B11 -> V[1,k1]=1
        # M6: (B00 + B01) -> V[0,k0]=1, V[0,k1]=1
        # M7: (B10 + B11) -> V[1,k0]=1, V[1,k1]=1
        
        # Strassen coefficients for C (2x2)
        # C00 = M1 + M4 - M5 + M7 -> W[k0,0]: M1=1, M4=1, M5=-1, M7=1
        # C01 = M3 + M5 -> W[k1,0]: M3=1, M5=1
        # C10 = M2 + M4 -> W[k0,1]: M2=1, M4=1
        # C11 = M1 - M2 + M3 + M6 -> W[k1,1]: M1=1, M2=-1, M3=1, M6=1
        
        # Note: W index is k*n + i.
        # C[i, k]. i is row, k is col.
        # W row index: k * 2 + i.
        
        # Term 1: M1
        U[0 * m + 0][column] = 1
        U[1 * m + 1][column] = 1
        V[0 * p + k0][column] = 1
        V[1 * p + k1][column] = 1
        W[k0 * n + 0][column] = 1
        W[k1 * n + 1][column] = 1
        column += 1
        
        # Term 2: M2
        U[1 * m + 0][column] = 1
        U[1 * m + 1][column] = 1
        V[0 * p + k0][column] = 1
        W[k0 * n + 1][column] = 1
        W[k1 * n + 1][column] = -1
        column += 1
        
        # Term 3: M3
        U[0 * m + 0][column] = 1
        V[1 * p + k0][column] = 1
        V[1 * p + k1][column] = -1
        W[k1 * n + 0][column] = 1
        W[k1 * n + 1][column] = 1
        column += 1
        
        # Term 4: M4
        U[1 * m + 1][column] = 1
        V[0 * p + k1][column] = 1
        V[0 * p + k0][column] = -1
        W[k0 * n + 0][column] = 1
        W[k0 * n + 1][column] = 1
        column += 1
        
        # Term 5: M5
        U[0 * m + 0][column] = 1
        U[0 * m + 1][column] = 1
        V[1 * p + k1][column] = 1
        W[k0 * n + 0][column] = -1
        W[k1 * n + 0][column] = 1
        column += 1
        
        # Term 6: M6
        U[1 * m + 0][column] = 1
        U[0 * m + 0][column] = -1
        V[0 * p + k0][column] = 1
        V[0 * p + k1][column] = 1
        W[k1 * n + 1][column] = 1
        column += 1
        
        # Term 7: M7
        U[0 * m + 1][column] = 1
        U[1 * m + 1][column] = -1
        V[1 * p + k0][column] = 1
        V[1 * p + k1][column] = 1
        W[k0 * n + 0][column] = 1
        column += 1
        
    # Schoolbook Part for A_right * B_bottom
    # A_right is column 2 of A. i in 0..1.
    # B_bottom is row 2 of B. k in 0..7.
    # Product contributes to C[i, k].
    # U[i, 2] = 1
    # V[2, k] = 1
    # W[k, i] = 1
    
    for i in range(n):
        for k in range(p):
            U[i * m + 2][column] = 1
            V[2 * p + k][column] = 1
            W[k * n + i][column] = 1
            column += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
