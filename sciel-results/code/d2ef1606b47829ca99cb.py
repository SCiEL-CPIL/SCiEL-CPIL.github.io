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
    Constructs the factor matrices U, V, W for the 5x5x5 matrix multiplication tensor.
    Decomposition strategy:
    - Split indices into I0={0,1,2,3} and I1={4}.
    - Block (I0, I0, I0) uses Strassen 4x4x4 (rank 49).
    - Other blocks use schoolbook multiplication.
    Total rank: 49 + 61 = 110.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We will build the columns of U, V, W as lists of (index, value) pairs.
    # Then convert to dense arrays.
    cols_u = []
    cols_v = []
    cols_w = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        Adds a rank-1 term.
        u_dict: {row_index: coeff}
        v_dict: {col_index: coeff}
        w_dict: {out_index: coeff}
        """
        # Filter out zeros
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        
        if not u_dict or not v_dict or not w_dict:
            return
            
        cols_u.append(u_dict)
        cols_v.append(v_dict)
        cols_w.append(w_dict)

    # --- Strassen 4x4x4 Decomposition ---
    # Strassen 2x2x2 terms (local indices 0,1)
    # T1 = (A00+A11)(B00+B11) -> C00 += T1, C11 += T1
    # T2 = (A10+A11)B00 -> C10 += T2, C11 += -T2
    # T3 = A00(B01-B11) -> C01 += T3, C11 += T3
    # T4 = A11(B10-A00) -> C00 += T4, C10 += T4
    # T5 = (A00+A01)B11 -> C01 += T5, C00 += -T5
    # T6 = (A10-A00)(B00+B01) -> C10 += T6, C01 += -T6
    # T7 = (A01-A11)(B10+B11) -> C11 += T7, C00 += -T7
    
    # Note: The standard Strassen formulas for 2x2 are:
    # M1 = (A11 + A22) (B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - A11) -- Wait, standard is A22 (B21 - A11) is not quite right for all variations.
    # Let's use the explicit rank-7 decomposition for 2x2x2 which is well known.
    
    # Indices for 2x2: 00, 01, 10, 11
    # U indices: (i,j), V indices: (j,k), W indices: (k,i)
    
    S2x2 = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (0,0):-1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # To get 4x4x4 rank 49, we recursively apply Strassen to the 4 quadrants.
    # The 4x4 matrix is composed of four 2x2 blocks.
    # The tensor decomposition is the tensor product of the 2x2 decomposition with itself?
    # No, the rank of $M_{<4,4,4>}$ is 49. This is obtained by applying the rank-7 decomposition to the 4 "macro" blocks.
    # There are 7 macro-terms. Each macro-term is a product of 2x2 matrices.
    # Each 2x2 product can be decomposed into 7 rank-1 terms.
    # So total rank = 7 * 7 = 49.
    
    def add_strassen_4x4(offset_i, offset_j, offset_k):
        # Macro definitions for 4x4 Strassen
        # These are the same structure as S2x2 but operating on 2x2 blocks (quadrants)
        # Quadrant indices: (0,0), (0,1), (1,0), (1,1)
        # Local indices within quadrant: 0, 1
        
        macro_defs = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (0,0):-1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for u_m, v_m, w_m in macro_defs:
            for u_s, v_s, w_s in S2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Expand U
                for (li, lj), val_u_s in u_s.items():
                    for (mi, mj), val_u_m in u_m.items():
                        val = val_u_s * val_u_m
                        if val != 0:
                            gi = offset_i + mi * 2 + li
                            gj = offset_j + mj * 2 + lj
                            u_dict[gi * m + gj] = u_dict.get(gi * m + gj, 0) + val
                
                # Expand V
                for (lj, lk), val_v_s in v_s.items():
                    for (mj, mk), val_v_m in v_m.items():
                        val = val_v_s * val_v_m
                        if val != 0:
                            gj = offset_j + mj * 2 + lj
                            gk = offset_k + mk * 2 + lk
                            v_dict[gj * p + gk] = v_dict.get(gj * p + gk, 0) + val
                            
                # Expand W
                for (li, lk), val_w_s in w_s.items():
                    for (mi, mk), val_w_m in w_m.items():
                        val = val_w_s * val_w_m
                        if val != 0:
                            gi = offset_i + mi * 2 + li
                            gk = offset_k + mk * 2 + lk
                            w_dict[gk * n + gi] = w_dict.get(gk * n + gi, 0) + val
                            
                add_term(u_dict, v_dict, w_dict)

    def add_schoolbook(i_set, j_set, k_set):
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    add_term(
                        {i * m + j: 1.0},
                        {j * p + k: 1.0},
                        {k * n + i: 1.0}
                    )
                    
    # Build decomposition
    # 4x4x4 block
    add_strassen_4x4(0, 0, 0)
    
    # Off-diagonal blocks
    G4 = [0, 1, 2, 3]
    G1 = [4]
    
    add_schoolbook(G4, G4, G1) # 16
    add_schoolbook(G4, G1, G4) # 16
    add_schoolbook(G1, G4, G4) # 16
    add_schoolbook(G4, G1, G1) # 4
    add_schoolbook(G1, G4, G1) # 4
    add_schoolbook(G1, G1, G4) # 4
    add_schoolbook(G1, G1, G1) # 1
    
    rank = len(cols_u)
    
    # Construct numpy arrays
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        for idx, val in cols_u[c].items():
            U[idx, c] = val
        for idx, val in cols_v[c].items():
            V[idx, c] = val
        for idx, val in cols_w[c].items():
            W[idx, c] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
