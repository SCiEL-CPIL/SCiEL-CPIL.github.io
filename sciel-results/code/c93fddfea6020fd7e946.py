"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Block-structured decomposition using Strassen for 2x2 blocks."""
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Strassen 2x2x2x2 decomposition (rank 7)
    # Local indices: U has 4 (2x2), V has 4 (2x2), W has 4 (2x2)
    # U local: 0=A00, 1=A01, 2=A10, 3=A11
    # V local: 0=B00, 1=B01, 2=B10, 3=B11
    # W local: 0=C00, 1=C01, 2=C10, 3=C11
    def strassen_terms():
        return [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # M1
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),       # M2
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),       # M3
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),       # M4
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),       # M5
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),       # M6
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),       # M7
        ]
    
    # Direct 1x2 * 2x2 -> 1x2 (rank 4)
    # U local: 0=A0, 1=A1
    # V local: 0=B00, 1=B01, 2=B10, 3=B11
    # W local: 0=C0, 1=C1
    def direct_1222_terms():
        return [
            ({0:1}, {0:1}, {0:1}),  # A0*B00 -> C0
            ({0:1}, {1:1}, {1:1}),  # A0*B01 -> C1
            ({1:1}, {2:1}, {0:1}),  # A1*B10 -> C0
            ({1:1}, {3:1}, {1:1}),  # A1*B11 -> C1
        ]
    
    # Build decomposition
    U_rows = n * m  # 12
    V_rows = m * p  # 24
    W_rows = p * n  # 18
    
    all_terms = []
    
    # A: 3x4, split rows [0:2],[2:3], cols [0:2],[2:4]
    # B: 4x6, split rows [0:2],[2:4], cols [0:2],[2:4],[4:6]
    
    a_row_splits = [(0, 2), (2, 1)]  # (start, size)
    a_col_splits = [(0, 2), (2, 2)]
    b_row_splits = [(0, 2), (2, 2)]
    b_col_splits = [(0, 2), (2, 2), (4, 2)]
    
    for (ar_start, ar_size) in a_row_splits:
        for (ac_start, ac_size) in a_col_splits:
            for (bc_start, bc_size) in b_col_splits:
                # B row group matches A col group
                br_start = ac_start
                br_size = ac_size  # should be 2
                
                # Determine which decomposition to use
                if ar_size == 2 and ac_size == 2 and bc_size == 2:
                    terms = strassen_terms()
                elif ar_size == 1 and ac_size == 2 and bc_size == 2:
                    terms = direct_1222_terms()
                else:
                    # Schoolbook fallback
                    terms = []
                    for li in range(ar_size):
                        for lj in range(ac_size):
                            for lk in range(bc_size):
                                terms.append(({li*ac_size + lj: 1}, 
                                              {lj*bc_size + lk: 1}, 
                                              {li*bc_size + lk: 1}))
                
                for (u_loc, v_loc, w_loc) in terms:
                    u_g = {}
                    for local_idx, coeff in u_loc.items():
                        li = local_idx // ac_size
                        lj = local_idx % ac_size
                        global_row = (ar_start + li) * m + (ac_start + lj)
                        u_g[global_row] = coeff
                    
                    v_g = {}
                    for local_idx, coeff in v_loc.items():
                        li = local_idx // bc_size
                        lj = local_idx % bc_size
                        global_row = (br_start + li) * p + (bc_start + lj)
                        v_g[global_row] = coeff
                    
                    w_g = {}
                    for local_idx, coeff in w_loc.items():
                        li = local_idx // bc_size
                        lj = local_idx % bc_size
                        global_row = (bc_start + lj) * n + (ar_start + li)
                        w_g[global_row] = coeff
                    
                    all_terms.append((u_g, v_g, w_g))
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for col, (u_g, v_g, w_g) in enumerate(all_terms):
        for row, coeff in u_g.items():
            U[row][col] = coeff
        for row, coeff in v_g.items():
            V[row][col] = coeff
        for row, coeff in w_g.items():
            W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
