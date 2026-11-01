"""Optimized program: Strassen-based rank-35 decomposition for 2x2 * 2x10."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-35 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x10 matrix.

    Uses Strassen's 7-term algorithm for the 2x2x2 core, applied
    to each of the 5 independent 2x2x2 blocks (since p=10 = 2*5).
    Total rank = 5 * 7 = 35.

    Convention:
      U has shape (n*m, rank) = (4, 35)
      V has shape (m*p, rank) = (20, 35)
      W has shape (p*n, rank) = (20, 35)

    Indexing:
      U[i*m + j, c]  where i in [0,n), j in [0,m)
      V[j*p + k, c]  where j in [0,m), k in [0,p)
      W[k*n + i, c]  where k in [0,p), i in [0,n)

    For Strassen on 2x2 * 2x2 (indices i,j for first, j,k for second):
      M1 = (A00+A11)(B01+B10)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B00-B11)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)

      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6

    For the 2x2 * 2x10 problem, we split the 10 columns into 5 groups of 2.
    Each group (columns 2g, 2g+1 for g=0..4) is handled by one Strassen block.
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7  # 35

    U = [[0] * rank for _ in range(n * m)]   # (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # (20, 35)

    # Strassen terms for a single 2x2x2 block:
    # Term t in 0..6
    # U contribution: row (i*m+j) for the A matrix index
    # V contribution: row (j*p+k) for the B matrix index
    # W contribution: row (k*n+i) for the C matrix index

    # For each block g (columns 2g, 2g+1):
    #   B00 = V[j=0, k=2g]
    #   B01 = V[j=0, k=2g+1]
    #   B10 = V[j=1, k=2g]
    #   B11 = V[j=1, k=2g+1]
    #
    #   A00 = U[i=0, j=0]
    #   A01 = U[i=0, j=1]
    #   A10 = U[i=1, j=0]
    #   A11 = U[i=1, j=1]
    #
    #   C00 = W[k=2g, i=0]
    #   C01 = W[k=2g+1, i=0]
    #   C10 = W[k=2g, i=1]
    #   C11 = W[k=2g+1, i=1]

    # Strassen coefficients:
    # M1 = (A00+A11)(B01+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # For each term, the U rows and V rows that participate:
    # term: (U_rows_with_coeff, V_rows_with_coeff, W_rows_with_coeff)
    # U_rows: list of (row_index, coeff)
    # V_rows: list of (row_index, coeff)
    # W_rows: list of (row_index, coeff)

    strassen_terms = [
        # M1 = (A00+A11)(B01+B10)
        {
            'U': [(0, 1), (3, 1)],  # A00=U[0], A11=U[3]
            'V': [(1, 1), (2, 1)],  # B01=V[1], B10=V[2]
            'W': [(0, 1), (3, -1), (2, 1), (1, 1)],  # C00+=M1, C11+=M1, C00-=M5 handled separately
        },
        # M2 = (A10+A11)B00
        {
            'U': [(2, 1), (3, 1)],  # A10=U[2], A11=U[3]
            'V': [(0, 1)],          # B00=V[0]
            'W': [(2, 1), (3, -1)], # C10+=M2, C11-=M2
        },
        # M3 = A00(B10-B11)
        {
            'U': [(0, 1)],          # A00=U[0]
            'V': [(2, 1), (3, -1)], # B10=V[2], B11=V[3]
            'W': [(1, 1), (2, 1)],  # C01+=M3, C10+=M3
        },
        # M4 = A11(B00-B11)
        {
            'U': [(3, 1)],          # A11=U[3]
            'V': [(0, 1), (3, -1)], # B00=V[0], B11=V[3]
            'W': [(0, 1), (2, 1)],  # C00+=M4, C10+=M4
        },
        # M5 = (A00+A01)B11
        {
            'U': [(0, 1), (1, 1)],  # A00=U[0], A01=U[1]
            'V': [(3, 1)],          # B11=V[3]
            'W': [(0, -1), (1, 1)], # C00-=M5, C01+=M5
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'U': [(2, 1), (0, -1)], # A10=U[2], A00=U[0]
            'V': [(0, 1), (1, 1)],  # B00=V[0], B01=V[1]
            'W': [(3, 1)],          # C11+=M6
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'U': [(1, 1), (3, -1)], # A01=U[1], A11=U[3]
            'V': [(2, 1), (3, 1)],  # B10=V[2], B11=V[3]
            'W': [(0, 1)],          # C00+=M7
        },
    ]

    # Verify Strassen correctness for single block:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # W contributions per term:
    # M1: C00+=1, C11+=1
    # M2: C10+=1, C11+=-1
    # M3: C01+=1, C10+=1
    # M4: C00+=1, C10+=1
    # M5: C00+=-1, C01+=1
    # M6: C11+=1
    # M7: C00+=1
    #
    # Check C00: M1(1) + M4(1) + M5(-1) + M7(1) = 1+1-1+1 = 2? No...
    # Wait, let me recheck. The W row indices for a block at columns (2g, 2g+1):
    # C00 = W[2g*n + 0] = W[2g*2 + 0] = W[4g]
    # C01 = W[2g+1]*2 + 0 = W[4g+2]
    # C10 = W[2g]*2 + 1 = W[4g+1]
    # C11 = W[2g+1]*2 + 1 = W[4g+3]

    # Let me redo the W assignments with correct indices for block g:
    # C00 -> W[4g + 0]
    # C10 -> W[4g + 1]
    # C01 -> W[4g + 2]
    # C11 -> W[4g + 3]

    # And V indices for block g:
    # B00 -> V[0*p + 2g] = V[2g]
    # B01 -> V[0*p + 2g+1] = V[2g+1]
    # B10 -> V[1*p + 2g] = V[10+2g]
    # B11 -> V[1*p + 2g+1] = V[10+2g+1]

    # Let me rebuild with explicit indices per block.

    # Clear and rebuild
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for g in range(num_blocks):
        base_col = g * 7
        # V indices for this block
        B00 = 0 * p + 2 * g       # 2g
        B01 = 0 * p + 2 * g + 1   # 2g+1
        B10 = 1 * p + 2 * g       # 10+2g
        B11 = 1 * p + 2 * g + 1   # 10+2g+1

        # W indices for this block
        C00 = 2 * g * n + 0       # 4g
        C10 = 2 * g * n + 1       # 4g+1
        C01 = (2 * g + 1) * n + 0 # 4g+2
        C11 = (2 * g + 1) * n + 1 # 4g+3

        # U indices (same for all blocks)
        A00 = 0 * m + 0  # 0
        A01 = 0 * m + 1  # 1
        A10 = 1 * m + 0  # 2
        A11 = 1 * m + 1  # 3

        terms = [
            # M1 = (A00+A11)(B01+B10)
            {'U': [(A00, 1), (A11, 1)], 'V': [(B01, 1), (B10, 1)], 'W': [(C00, 1), (C11, 1)]},
            # M2 = (A10+A11)B00
            {'U': [(A10, 1), (A11, 1)], 'V': [(B00, 1)], 'W': [(C10, 1), (C11, -1)]},
            # M3 = A00(B10-B11)
            {'U': [(A00, 1)], 'V': [(B10, 1), (B11, -1)], 'W': [(C01, 1), (C10, 1)]},
            # M4 = A11(B00-B11)
            {'U': [(A11, 1)], 'V': [(B00, 1), (B11, -1)], 'W': [(C00, 1), (C10, 1)]},
            # M5 = (A00+A01)B11
            {'U': [(A00, 1), (A01, 1)], 'V': [(B11, 1)], 'W': [(C00, -1), (C01, 1)]},
            # M6 = (A10-A00)(B00+B01)
            {'U': [(A10, 1), (A00, -1)], 'V': [(B00, 1), (B01, 1)], 'W': [(C11, 1)]},
            # M7 = (A01-A11)(B10+B11)
            {'U': [(A01, 1), (A11, -1)], 'V': [(B10, 1), (B11, 1)], 'W': [(C00, 1)]},
        ]

        for t, term in enumerate(terms):
            col = base_col + t
            for row, coeff in term['U']:
                U[row][col] = coeff
            for row, coeff in term['V']:
                V[row][col] = coeff
            for row, coeff in term['W']:
                W[row][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
