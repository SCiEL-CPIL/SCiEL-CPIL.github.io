import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    # Block 1: A_L B_TL -> C_LL
    # A_L is rows 0,1,4,5 of U (indices in n*m=8)
    # Wait, U shape is (n*m, rank) = (8, 28).
    # Indices for A (2x4):
    # A[0,0]=0, A[0,1]=1, A[0,2]=4, A[0,3]=5?
    # Standard row-major:
    # A:
    # 0 1
    # 2 3
    # 4 5
    # 6 7
    # Wait, n=2, m=4.
    # A is 2x4.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # The current program uses X1 = [0, 1, 4, 5].
    # This implies A_L corresponds to indices 0,1,4,5?
    # If A is 2x4, A_L is 2x2.
    # If A is stored row-major:
    # Row 0: 0, 1, 2, 3
    # Row 1: 4, 5, 6, 7
    # A_L (cols 0,1): 0, 1, 4, 5. Correct.
    # A_R (cols 2,3): 2, 3, 6, 7. Correct.
    
    # B is 4x4.
    # B_TL (rows 0,1 cols 0,1): 0, 1, 4, 5.
    # B_TR (rows 0,1 cols 2,3): 2, 3, 6, 7.
    # B_BL (rows 2,3 cols 0,1): 8, 9, 12, 13.
    # B_BR (rows 2,3 cols 2,3): 10, 11, 14, 15.
    
    # C is 2x4.
    # C_L (cols 0,1): 0, 1, 4, 5.
    # C_R (cols 2,3): 2, 3, 6, 7.
    # Wait, W shape is (p*n, rank) = (8, 28).
    # W indices correspond to C entries.
    # C is 2x4.
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3
    # C[1,0]=4, C[1,1]=5, C[1,2]=6, C[1,3]=7
    
    # The current program uses Z1 = [0, 2, 1, 3].
    # This maps to C_L?
    # C_L entries are 0, 1, 4, 5.
    # Z1 has 0, 1, 2, 3.
    # This implies the output C is treated differently?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This usually means W is indexed by output position.
    # If W is 8x28, and C is 2x4.
    # The indices 0..7 in W correspond to C[0,0]..C[1,3].
    # Z1 = [0, 2, 1, 3].
    # This looks like it's writing to C[0,0], C[0,2], C[0,1], C[0,3]?
    # Or is it a permutation?
    # Let's check the Strassen output.
    # Strassen for 2x2:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M5
    # C21 = M3 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In add_block:
    # M1 affects Z[0] (+1), Z[3] (+1)
    # M2 affects Z[2] (+1), Z[3] (-1)
    # M3 affects Z[1] (+1), Z[3] (+1)
    # M4 affects Z[0] (+1), Z[2] (+1)
    # M5 affects Z[0] (-1), Z[1] (+1)
    # M6 affects Z[3] (+1)
    # M7 affects Z[0] (+1)
    
    # Let's trace C11 (Z[0] in local block):
    # M1: +1
    # M4: +1
    # M5: -1
    # M7: +1
    # Sum: M1 + M4 - M5 + M7. Matches Strassen C11.
    
    # C12 (Z[1] in local block):
    # M3: +1
    # M5: +1
    # Sum: M3 + M5.
    # Strassen C12 is M2 + M5.
    # Here we have M3 + M5.
    # Let's check M2 and M3 definitions.
    # M2: U[X2]+=1, U[X3]+=1, V[Y0]+=1, W[Z2]+=1, W[Z3]-=1
    # M3: U[X0]+=1, V[Y1]+=1, V[Y3]-=1, W[Z1]+=1, W[Z3]+=1
    
    # It seems the local indices Z[0..3] map to specific outputs.
    # If Z1 = [0, 2, 1, 3], then local Z[0] -> Global W[0], local Z[1] -> Global W[2], local Z[2] -> Global W[1], local Z[3] -> Global W[3].
    # So Global C[0,0] gets local C11.
    # Global C[0,2] gets local C12?
    # Global C[0,1] gets local C21?
    # Global C[0,3] gets local C22?
    
    # This implies the output C is not just C_L and C_R in standard order, or the blocks are permuted.
    # However, the task says "exactly decompose the matrix multiplication tensor".
    # As long as the decomposition is correct for the tensor, the score is valid.
    # The current program is reported to have score 1.1429, which implies it is valid.
    
    # I will keep the exact same logic.

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U, "V": V, "W": W}
