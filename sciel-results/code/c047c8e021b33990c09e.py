"""Matrix multiplication 2x2x2 rank-minimal exact tensor decomposition (Strassen, rank 7)."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm for 2x2 matrix multiplication, rank 7.
    # U: (n*m, rank) = (4, 7), indexed by row i, col j of A -> i*m+j
    # V: (m*p, rank) = (4, 7), indexed by row j, col k of B -> j*p+k
    # W: (p*n, rank) = (4, 7), indexed by row k, col i of C -> k*n+i  (trace convention)
    #
    # Strassen's 7 products:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U columns (A side coefficients for each product):
    #   col0 (M1): A00=1, A11=1
    #   col1 (M2): A10=1, A11=1
    #   col2 (M3): A00=1
    #   col3 (M4): A11=1
    #   col4 (M5): A00=1, A01=1
    #   col5 (M6): A10=1, A00=-1
    #   col6 (M7): A01=1, A11=-1
    #
    # V columns (B side coefficients for each product):
    #   col0 (M1): B00=1, B11=1
    #   col1 (M2): B00=1
    #   col2 (M3): B01=1, B11=-1
    #   col3 (M4): B10=1, B00=-1
    #   col4 (M5): B11=1
    #   col5 (M6): B00=1, B01=1
    #   col6 (M7): B10=1, B11=1
    #
    # W columns (C side coefficients; W[k*n+i] = coefficient of C[i][k] for product r)
    #   col0 (M1): C00=1, C11=1  -> W[0*2+0]=1, W[1*2+1]=1
    #   col1 (M2): C10=1, C11=-1 -> W[0*2+1]=1, W[1*2+1]=-1
    #   col2 (M3): C01=1, C11=1  -> W[1*2+0]=1, W[1*2+1]=1
    #   col3 (M4): C00=1, C10=1  -> W[0*2+0]=1, W[0*2+1]=1
    #   col4 (M5): C00=-1, C01=1 -> W[0*2+0]=-1, W[1*2+0]=1
    #   col5 (M6): C11=1         -> W[1*2+1]=1
    #   col6 (M7): C00=1         -> W[0*2+0]=1

    n, m, p = DIMENSIONS
    rank = 7

    # U: 4 rows (A indices 00, 01, 10, 11), 7 cols
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ]

    # V: 4 rows (B indices 00, 01, 10, 11), 7 cols
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ]

    # W: 4 rows (C indices in trace order: k*n+i -> 00, 10, 01, 11), 7 cols
    # W[0] = C[0][0], W[1] = C[1][0], W[2] = C[0][1], W[3] = C[1][1]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 (k=0,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0,i=1)
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 0],   # C11 (k=1,i=1)
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
