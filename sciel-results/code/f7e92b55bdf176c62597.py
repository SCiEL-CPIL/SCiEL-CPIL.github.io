"""5x5x5 matrix multiplication tensor decomposition using block structure with Strassen on 2x2 block."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 3+2 block partition. For the 2x2x2 diagonal block (A22*B22),
    uses Strassen's rank-7 decomposition instead of schoolbook rank 8.
    
    Total rank: 27+18+18+12+18+12+12+7 = 124
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []  # list of (U_dict, V_dict, W_dict)
    
    def add_column(u_entries, v_entries, w_entries):
        """Add a rank-1 column. Each is a list of (index, value)."""
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Block indices
    top = [0, 1, 2]
    bot = [3, 4]
    
    # Schoolbook for a block product
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # Strassen for 2x2x2 block
    def strassen_2x2x2():
        """Add 7 columns for the 2x2x2 Strassen decomposition."""
        # Local indices: 0,1 map to global 3,4
        # A[i,j] -> global A[3+i, 3+j], U index = (3+i)*5 + (3+j)
        # B[j,k] -> global B[3+j, 3+k], V index = (3+j)*5 + (3+k)
        # C[i,k] -> global C[3+i, 3+k], W index = (3+k)*5 + (3+i)
        
        # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
        # Each is a dict mapping local (i,j) or (j,k) or (i,k) to coefficient
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A10+A11)B00, contributes to C10 and -C11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3 = A00(B01-B11), contributes to C01 and C11
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4 = A11(B10-B00), contributes to C00 and C10
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5 = (A00+A01)B11, contributes to C01 and -C00
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # M6 = (A10-A00)(B00+B01), contributes to C11
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7 = (A01-A11)(B10+B11), contributes to C00
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), a_val in a_coeffs.items():
                gi, gj = 3 + li, 3 + lj
                u_entries.append((gi * m + gj, a_val))
            for (lj, lk), b_val in b_coeffs.items():
                gj, gk = 3 + lj, 3 + lk
                v_entries.append((gj * p + gk, b_val))
            for (li, lk), c_val in c_coeffs.items():
                gi, gk = 3 + li, 3 + lk
                w_entries.append((gk * n + gi, c_val))
            add_column(u_entries, v_entries, w_entries)
    
    # Build all blocks
    schoolbook_block(top, top, top)  # A11*B11: 27
    schoolbook_block(top, bot, top)  # A12*B21: 18
    schoolbook_block(top, top, bot)  # A11*B12: 18
    schoolbook_block(top, bot, bot)  # A12*B22: 12
    schoolbook_block(bot, top, top)  # A21*B11: 18
    schoolbook_block(bot, bot, top)  # A22*B21: 12
    schoolbook_block(bot, top, bot)  # A21*B12: 12
    strassen_2x2x2()                 # A22*B22: 7
    
    rank = len(columns)
    
    # Build the matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
