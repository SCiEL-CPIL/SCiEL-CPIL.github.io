"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen rank for 2x2 is 7.
    # We have p/m = 3 blocks of 2 columns in B.
    # Total rank = 7 * 3 = 21.
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(g-e)
    # M4 = d(f-c)
    # M5 = (a+b)f
    # M6 = (c-b)(e+f)
    # M7 = (a-d)(g+f)
    
    # C00 = M1 - M2 + M3 + M6
    # C01 = M5 - M3 - M6 - M7
    # C10 = M2 - M4 - M5 + M7
    # C11 = M1 + M4 - M2 + M5
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Block definitions:
    # Block 0: B cols 0,1 -> C cols 0,1
    # Block 1: B cols 2,3 -> C cols 2,3
    # Block 2: B cols 4,5 -> C cols 4,5
    
    for block in range(3):
        base_col = block * 7
        # B column offset
        b_col_off = block * 2
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[0][base_col] = 1; U[3][base_col] = 1
        V[0][base_col] = 1; V[1][base_col] = 1 # B[0,0], B[1,1] -> idx 0*2+0=0, 1*2+1=3?
        # V index j*p+k.
        # B is 2x6.
        # Block 0: B[0,0] is j=0, k=0 -> 0*6+0=0. B[1,1] is j=1, k=1 -> 1*6+1=7.
        # Wait, p=6.
        # V shape (12, 21).
        # V[j*6+k].
        # Block 0:
        # B[0,0] -> 0
        # B[1,1] -> 7
        # Block 1:
        # B[0,2] -> 2
        # B[1,3] -> 9
        # Block 2:
        # B[0,4] -> 4
        # B[1,5] -> 11
        
        V[0][base_col] = 1
        V[1 * 6 + (b_col_off + 1)][base_col] = 1 # j=1, k=b_col_off+1
        
        # W index k*n+i. n=2.
        # C[0,0] -> k=0, i=0 -> 0
        # C[1,1] -> k=1, i=1 -> 1*2+1=3
        # Block 0: C cols 0,1.
        # C[0,0] -> 0
        # C[1,1] -> 3
        # Block 1: C cols 2,3.
        # C[0,2] -> 2
        # C[1,3] -> 5
        # Block 2: C cols 4,5.
        # C[0,4] -> 4
        # C[1,5] -> 7
        
        W[0 * 2 + 0][base_col] = 1
        W[(b_col_off + 1) * 2 + 1][base_col] = 1
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1, C11=-1
        U[2][base_col+1] = 1; U[3][base_col+1] = 1
        V[0][base_col+1] = 1
        # C10: k=0, i=1 -> 1. C11: k=1, i=1 -> 3.
        W[0 * 2 + 1][base_col+1] = 1
        W[(b_col_off + 1) * 2 + 1][base_col+1] = -1
        
        # M3: A00(B11-B00)
        # U: A00=1
        # V: B11=1, B00=-1
        # W: C00=1, C01=-1
        U[0][base_col+2] = 1
        V[1 * 6 + (b_col_off + 1)][base_col+2] = 1
        V[0][base_col+2] = -1
        # C00: 0. C01: 1.
        W[0][base_col+2] = 1
        W[1 * 2 + 0][base_col+2] = -1
        
        # M4: A11(B01-B10)
        # U: A11=1
        # V: B01=1, B10=-1
        # W: C11=1, C10=-1
        U[3][base_col+3] = 1
        V[0 * 6 + (b_col_off + 1)][base_col+3] = 1
        V[1 * 6 + b_col_off][base_col+3] = -1
        # C11: 3. C10: 1.
        W[(b_col_off + 1) * 2 + 1][base_col+3] = 1
        W[0 * 2 + 1][base_col+3] = -1
        
        # M5: (A00+A01)B10
        # U: A00=1, A01=1
        # V: B10=1
        # W: C01=1, C10=-1
        U[0][base_col+4] = 1; U[1][base_col+4] = 1
        V[1 * 6 + b_col_off][base_col+4] = 1
        # C01: 1. C10: 1.
        W[1 * 2 + 0][base_col+4] = 1
        W[0 * 2 + 1][base_col+4] = -1
        
        # M6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C00=1, C01=-1
        U[2][base_col+5] = 1; U[0][base_col+5] = -1
        V[0][base_col+5] = 1
        V[0 * 6 + (b_col_off + 1)][base_col+5] = 1
        # C00: 0. C01: 1.
        W[0][base_col+5] = 1
        W[1 * 2 + 0][base_col+5] = -1
        
        # M7: (A00-A11)(B11+B10)
        # U: A00=1, A11=-1
        # V: B11=1, B10=1
        # W: C01=-1, C10=1
        U[0][base_col+6] = 1; U[3][base_col+6] = -1
        V[1 * 6 + (b_col_off + 1)][base_col+6] = 1
        V[1 * 6 + b_col_off][base_col+6] = 1
        # C01: 1. C10: 1.
        W[1 * 2 + 0][base_col+6] = -1
        W[0 * 2 + 1][base_col+6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
