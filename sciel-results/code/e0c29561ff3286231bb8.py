"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Use Strassen's 2x2x2 algorithm for each pair of columns.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) -> 7*7 = 49 rank
    # 1 single column (14) -> 4 rank (trivial)
    # Total rank = 53
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    def add_strassen_block(base_r, k0, k1):
        """
        Strassen decomposition for 2x2 A times 2x2 B (columns k0, k1).
        
        B matrix columns are k0 and k1, rows are 0 and 1.
        B0k0 = B[0][k0], B0k1 = B[0][k1]
        B1k0 = B[1][k0], B1k1 = B[1][k1]
        
        7 products:
        P1 = (A00+A11)(B0k0+B1k1)
        P2 = (A10+A11)(B0k0)
        P3 = A00(B0k1-B1k1)
        P4 = A11(B1k0-B0k0)
        P5 = (A00+A01)(B1k1)
        P6 = (A10-A00)(B0k0+B0k1)
        P7 = (A01-A11)(B1k0+B1k1)
        
        Output C = A*B:
        C0k0 = A00*B0k0 + A01*B1k0 = P1 + P4 - P5 + P7
        C1k0 = A10*B0k0 + A11*B1k0 = P2 + P4
        C0k1 = A00*B0k1 + A01*B1k1 = P3 + P5
        C1k1 = A10*B0k1 + A11*B1k1 = P1 - P2 + P3 + P6
        """
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1: (A00+A11)(B0k0+B1k1)
        U[0][r1] = 1   # A00
        U[3][r1] = 1   # A11
        V[0 * p + k0][r1] = 1   # B0k0
        V[1 * p + k1][r1] = 1   # B1k1
        
        # P2: (A10+A11)(B0k0)
        U[2][r2] = 1   # A10
        U[3][r2] = 1   # A11
        V[0 * p + k0][r2] = 1   # B0k0
        
        # P3: A00(B0k1-B1k1)
        U[0][r3] = 1   # A00
        V[0 * p + k1][r3] = 1   # B0k1
        V[1 * p + k1][r3] = -1  # B1k1
        
        # P4: A11(B1k0-B0k0)
        U[3][r4] = 1   # A11
        V[1 * p + k0][r4] = 1   # B1k0
        V[0 * p + k0][r4] = -1  # B0k0
        
        # P5: (A00+A01)(B1k1)
        U[0][r5] = 1   # A00
        U[1][r5] = 1   # A01
        V[1 * p + k1][r5] = 1   # B1k1
        
        # P6: (A10-A00)(B0k0+B0k1)
        U[2][r6] = 1   # A10
        U[0][r6] = -1  # A00
        V[0 * p + k0][r6] = 1   # B0k0
        V[0 * p + k1][r6] = 1   # B0k1
        
        # P7: (A01-A11)(B1k0+B1k1)
        U[1][r7] = 1   # A01
        U[3][r7] = -1  # A11
        V[1 * p + k0][r7] = 1   # B1k0
        V[1 * p + k1][r7] = 1   # B1k1
        
        # Reconstruction
        # C0k0 = P1 + P4 - P5 + P7 -> W[2*k0]
        W[2 * k0][r1] = 1
        W[2 * k0][r4] = 1
        W[2 * k0][r5] = -1
        W[2 * k0][r7] = 1
        
        # C1k0 = P2 + P4 -> W[2*k0+1]
        W[2 * k0 + 1][r2] = 1
        W[2 * k0 + 1][r4] = 1
        
        # C0k1 = P3 + P5 -> W[2*k1]
        W[2 * k1][r3] = 1
        W[2 * k1][r5] = 1
        
        # C1k1 = P1 - P2 + P3 + P6 -> W[2*k1+1]
        W[2 * k1 + 1][r1] = 1
        W[2 * k1 + 1][r2] = -1
        W[2 * k1 + 1][r3] = 1
        W[2 * k1 + 1][r6] = 1
    
    # Add 7 Strassen blocks for column pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
    
    # Add trivial block for column 14 (single column, 4 multiplications)
    base_r = 49
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Q1 = A00 * B0,14 -> C0,14
    U[0][r1] = 1
    V[0 * p + 14][r1] = 1
    W[2 * 14][r1] = 1
    
    # Q2 = A01 * B1,14 -> C0,14
    U[1][r2] = 1
    V[1 * p + 14][r2] = 1
    W[2 * 14][r2] = 1
    
    # Q3 = A10 * B0,14 -> C1,14
    U[2][r3] = 1
    V[0 * p + 14][r3] = 1
    W[2 * 14 + 1][r3] = 1
    
    # Q4 = A11 * B1,14 -> C1,14
    U[3][r4] = 1
    V[1 * p + 14][r4] = 1
    W[2 * 14 + 1][r4] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
