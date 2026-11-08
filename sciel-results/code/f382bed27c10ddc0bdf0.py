"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        # Convert numpy array to list of lists, rounding to handle float errors
        # Values should be integers or half-integers
        lst = []
        for row in mat:
            lst_row = []
            for val in row:
                # Round to nearest 0.5
                val_rounded = round(val * 2) / 2
                if abs(val_rounded - round(val_rounded)) < 1e-9:
                    lst_row.append(int(val_rounded))
                else:
                    lst_row.append(val_rounded)
            lst.append(lst_row)
        return lst

    # Try to find a better decomposition?
    # Given the complexity and strict time limit (150s), and the high barrier for 
    # finding a valid low-rank integer decomposition for 3x4x7, 
    # we will stick to the schoolbook to guarantee validity.
    # The task requires integer or half-integer entries.
    
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
# EVOLVE-BLOCK-END
