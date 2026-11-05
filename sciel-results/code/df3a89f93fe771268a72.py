"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 78
    # Structure:
    # C rows 0,1 (2x7) from A rows 0,1 (2x4) * B (4x7)
    #   Split A cols into 2+2.
    #   For each 2x4 part, split B cols into 4+3.
    #   4 part -> two 2x2 Strassen.
    #   3 part -> one 2x2 Strassen + one 2x2x1 schoolbook.
    # C row 2 (1x7) from A row 2 (1x4) * B (4x7)
    #   Schoolbook 1x4x7.

    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_block(i_base, j_base, k_base, offset):
        # Adds a 2x2x2 Strassen block for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]
        # i_base, j_base, k_base are starting indices
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # U: (0,0)+1, (1,1)+1. V: (0,0)+1, (0,1)+1, (1,0)+1, (1,1)+1. W: (0,0)+1, (0,1)+1, (1,0)+1, (1,1)+1?
        # No, Strassen factors are specific.
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i

        # M1
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M2 = (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1

        # M3 = A00(B10-B11)
        U[i_base * m + j_base][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + i_base][col] -= 1
        col += 1

        # M4 = A11(B01-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        W[k_base * n + (i_base+1)][col] -= 1
        col += 1

        # M5 = (A00+A10)B11
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + j_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1

    def add_schoolbook_block(i_base, j_base, k_base, i_size, j_size, k_size):
        # Adds schoolbook factors for a sub-block
        nonlocal col
        for ii in range(i_size):
            for jj in range(j_size):
                for kk in range(k_size):
                    i = i_base + ii
                    j = j_base + jj
                    k = k_base + kk
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # --- Part 1: C rows 0,1 (i in 0,1) ---
    # Split j (0..3) into 0..1 and 2..3
    for j_block in range(2):
        j_base = j_block * 2
        # Split k (0..6) into 0..3 and 4..6
        
        # k 0..3 (4 cols) -> two 2x2x2 Strassen blocks
        add_strassen_block(0, j_base, 0, 0)
        add_strassen_block(0, j_base, 2, 0)
        
        # k 4..6 (3 cols) -> one 2x2x2 Strassen (k 4,5) + one 2x2x1 schoolbook (k 6)
        add_strassen_block(0, j_base, 4, 0)
        add_schoolbook_block(0, j_base, 6, 2, 2, 1)

    # --- Part 2: C row 2 (i=2) ---
    # Schoolbook 1x4x7
    add_schoolbook_block(2, 0, 0, 1, 4, 7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
