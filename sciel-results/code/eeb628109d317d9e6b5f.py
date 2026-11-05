"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + 16 for 4x4x1 (Schoolbook) = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Strassen coefficients for 2x2
    # A, B, C are 2x2 matrices for each of the 7 terms
    A_2x2 = [
        [[1, 0], [0, 1]], # T1
        [[0, 0], [1, 1]], # T2
        [[1, 0], [0, 0]], # T3
        [[0, 0], [0, 1]], # T4
        [[1, 1], [0, 0]], # T5
        [[-1, 0], [1, 0]], # T6
        [[0, 1], [0, -1]]  # T7
    ]
    B_2x2 = [
        [[1, 0], [0, 1]], # T1
        [[1, 0], [0, 0]], # T2
        [[0, 1], [0, -1]], # T3
        [[-1, 0], [1, 0]], # T4
        [[0, 0], [0, 1]], # T5
        [[1, 1], [0, 0]], # T6
        [[0, 0], [1, 1]]  # T7
    ]
    C_2x2 = [
        [[1, 0], [0, 1]], # T1
        [[0, 0], [1, 0]], # T2
        [[0, 1], [0, 0]], # T3
        [[1, 0], [1, 0]], # T4
        [[-1, 0], [0, 0]], # T5
        [[0, 0], [0, 1]], # T6
        [[1, 0], [0, 0]]  # T7
    ]
    
    # Iterate over 7 top-level Strassen terms
    for r in range(7):
        # Iterate over 7 bottom-level Strassen terms
        for t in range(7):
            col = r * 7 + t
            
            # A part
            # Top-level term r determines which blocks of A are used
            # We need to map the 2x2 block structure of the 4x4 matrix
            # A_4x4 is composed of 4 blocks of 2x2: A00, A01, A10, A11
            # Indices: 0-1 are block 0, 2-3 are block 1
            # i = 2*I + u, j = 2*J + v
            
            # The top-level term r defines a linear combination of A-blocks.
            # We can represent this by a 2x2 matrix of block coefficients.
            # A_block_coeffs[r] is a 2x2 matrix where entry (I,J) is the coeff for block A[I][J]
            
            # Let's derive A_block_coeffs from A_2x2[r]
            # A_2x2[r] is the coefficient for the 2x2 "super-elements"
            # Wait, A_2x2[r] IS the coefficient matrix for the 2x2 blocks?
            # Yes, in Strassen, we treat the 2x2 blocks as scalars.
            # So A_2x2[r][I][J] is the coefficient for block A[I][J].
            
            # Similarly for B.
            
            # For each block (I, J) with non-zero coefficient in A_2x2[r]:
            for I in range(2):
                for J in range(2):
                    coeff_A_block = A_2x2[r][I][J]
                    if coeff_A_block == 0: continue
                    
                    # Within this block, we apply the bottom-level term t
                    # The bottom-level term t has A coefficients A_2x2[t][u][v]
                    for u in range(2):
                        for v in range(2):
                            coeff_A_local = A_2x2[t][u][v]
                            if coeff_A_local == 0: continue
                            
                            i = 2 * I + u
                            j = 2 * J + v
                            U[i * m + j][col] += coeff_A_block * coeff_A_local
            
            # B part
            for I in range(2):
                for J in range(2):
                    coeff_B_block = B_2x2[r][I][J]
                    if coeff_B_block == 0: continue
                    
                    for u in range(2):
                        for v in range(2):
                            coeff_B_local = B_2x2[t][u][v]
                            if coeff_B_local == 0: continue
                            
                            # B indices: row j, col k
                            # Block I corresponds to rows 2I..2I+1
                            # Block J corresponds to cols 2J..2J+1
                            j = 2 * I + u
                            k = 2 * J + v
                            V[j * p + k][col] += coeff_B_block * coeff_B_local
            
            # W part (Output C)
            # Output C is 4x4.
            # Top-level term r produces a 2x2 block of C?
            # No, Strassen computes C blocks.
            # C_2x2[r][K][L] is the coefficient for block C[K][L] from top-level term r.
            # Wait, the output of the top-level Strassen is the 2x2 matrix of blocks.
            # Each block is a 2x2 matrix.
            # The bottom-level term t computes the 2x2 matrix within a block.
            
            for K in range(2):
                for L in range(2):
                    coeff_C_block = C_2x2[r][K][L]
                    if coeff_C_block == 0: continue
                    
                    for u in range(2):
                        for v in range(2):
                            coeff_C_local = C_2x2[t][u][v]
                            if coeff_C_local == 0: continue
                            
                            # C indices: row k, col i
                            # Block K corresponds to rows 2K..2K+1
                            # Block L corresponds to cols 2L..2L+1
                            k = 2 * K + u
                            i = 2 * L + v
                            W[k * n + i][col] += coeff_C_block * coeff_C_local

    # --- Part 2: 4x4x1 using Schoolbook (Rank 16) ---
    # Output column 4 (index 4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 16
    for i in range(n):
        for j in range(m):
            col = 49 + i * m + j
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1 # B[j, 4]
            W[4 * n + i][col] = 1 # C[4, i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
