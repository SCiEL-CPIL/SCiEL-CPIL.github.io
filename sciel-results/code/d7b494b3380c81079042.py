"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 8 independent 2x2x2 sub-problems
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2 columns in B
    for t in range(4):
        # 2 parts: s=0 uses A cols 0,1 and B rows 0,1
        # s=1 uses A cols 2,3 and B rows 2,3
        for s in range(2):
            # Strassen 7 steps
            for c in range(7):
                # Define indices for A (U) and B (V)
                # A indices: i in {0,1}, j in {0,1} for s=0, j in {2,3} for s=1
                # Local j' in {0,1} -> Global j = s*2 + j'
                # B indices: j in {0,1} for s=0, j in {2,3} for s=1
                # Local j'' in {0,1} -> Global j = s*2 + j''
                # k in {0,1} -> Global k = 2*t + k'
                
                # U entries (A)
                # X00: i=0, j=s*2
                # X01: i=0, j=s*2+1
                # X10: i=1, j=s*2
                # X11: i=1, j=s*2+1
                
                # V entries (B)
                # Y00: j=s*2, k=2*t
                # Y01: j=s*2, k=2*t+1
                # Y10: j=s*2+1, k=2*t
                # Y11: j=s*2+1, k=2*t+1
                
                # W entries (C)
                # Z00: i=0, k=2*t
                # Z01: i=0, k=2*t+1
                # Z10: i=1, k=2*t
                # Z11: i=1, k=2*t+1
                
                if c == 0: # M1
                    # U: X00, X11
                    U[0 * m + s*2][col] = 1
                    U[1 * m + s*2+1][col] = 1
                    # V: Y00, Y11
                    V[s*2 * p + 2*t][col] = 1
                    V[(s*2+1) * p + 2*t+1][col] = 1
                    # W: Z00, Z01, Z10, Z11
                    W[2*t * n + 0][col] = 1
                    W[(2*t+1) * n + 0][col] = -1
                    W[2*t * n + 1][col] = 1
                    W[(2*t+1) * n + 1][col] = 1
                elif c == 1: # M2
                    # U: X10, X11
                    U[1 * m + s*2][col] = 1
                    U[1 * m + s*2+1][col] = 1
                    # V: Y00
                    V[s*2 * p + 2*t][col] = 1
                    # W: Z01, Z10
                    W[(2*t+1) * n + 0][col] = 1
                    W[2*t * n + 1][col] = 1
                elif c == 2: # M3
                    # U: X00
                    U[0 * m + s*2][col] = 1
                    # V: Y10, Y11
                    V[(s*2+1) * p + 2*t][col] = 1
                    V[(s*2+1) * p + 2*t+1][col] = -1
                    # W: Z01, Z11
                    W[(2*t+1) * n + 0][col] = 1
                    W[(2*t+1) * n + 1][col] = 1
                elif c == 3: # M4
                    # U: X11
                    U[1 * m + s*2+1][col] = 1
                    # V: Y00, Y01
                    V[s*2 * p + 2*t][col] = 1
                    V[s*2 * p + 2*t+1][col] = -1
                    # W: Z00, Z01
                    W[2*t * n + 0][col] = -1
                    W[(2*t+1) * n + 0][col] = -1
                elif c == 4: # M5
                    # U: X00, X10
                    U[0 * m + s*2][col] = 1
                    U[1 * m + s*2][col] = 1
                    # V: Y11
                    V[(s*2+1) * p + 2*t+1][col] = 1
                    # W: Z00, Z10
                    W[2*t * n + 0][col] = -1
                    W[2*t * n + 1][col] = 1
                elif c == 5: # M6
                    # U: X10, X00
                    U[1 * m + s*2][col] = 1
                    U[0 * m + s*2][col] = -1
                    # V: Y00, Y01
                    V[s*2 * p + 2*t][col] = 1
                    V[s*2 * p + 2*t+1][col] = 1
                    # W: Z01, Z11
                    W[(2*t+1) * n + 0][col] = 1
                    W[(2*t+1) * n + 1][col] = 1
                elif c == 6: # M7
                    # U: X01, X11
                    U[0 * m + s*2+1][col] = 1
                    U[1 * m + s*2+1][col] = -1
                    # V: Y10, Y11
                    V[(s*2+1) * p + 2*t][col] = 1
                    V[(s*2+1) * p + 2*t+1][col] = 1
                    # W: Z00, Z11
                    W[2*t * n + 0][col] = 1
                    W[(2*t+1) * n + 1][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
