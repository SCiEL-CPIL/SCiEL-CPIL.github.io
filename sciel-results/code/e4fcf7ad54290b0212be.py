"""5x5x5 matrix multiplication tensor decomposition using 4+1 block structure with recursive Strassen (rank 110)."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# Precompute Strassen 2x2 terms as tuples for speed
_STRASSEN_2x2 = [
    ((0,0,1.0),(1,1,1.0)), ((0,0,1.0),(1,1,1.0)), ((0,0,1.0),(1,1,1.0)),
    ((1,0,1.0),(1,1,1.0)), ((0,0,1.0),), ((1,0,1.0),(1,1,-1.0)),
    ((0,0,1.0),), ((0,1,1.0),(1,1,-1.0)), ((0,1,1.0),(1,1,1.0)),
    ((1,1,1.0),), ((1,0,1.0),(0,0,-1.0)), ((0,0,1.0),(1,0,1.0)),
    ((0,0,1.0),(0,1,1.0)), ((1,1,1.0),), ((0,1,1.0),(0,0,-1.0)),
    ((1,0,1.0),(0,0,-1.0)), ((0,0,1.0),(0,1,1.0)), ((1,1,1.0),),
    ((0,1,1.0),(1,1,-1.0)), ((1,0,1.0),(1,1,1.0)), ((0,0,1.0),),
]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 4+1 block partition. 
    The 4x4x4 diagonal block uses recursive Strassen (Strassen 4x4 composed of 7 Strassen 2x2 blocks).
    Rank of 4x4x4 block: 7 * 7 = 49.
    Off-diagonal blocks use schoolbook.
    Total rank: 49 + 61 = 110.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
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
    
    # Block index groups: 4+1 partition
    # Block 0: rows/cols 0,1,2,3 (size 4)
    # Block 1: row/col 4 (size 1)
    blocks = [[0, 1, 2, 3], [4]]
    
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
                    
    # Standard Strassen 2x2x2 terms for local indices 0,1
    # Returns list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: {(li, lj): val}
    # v_coeffs: {(lj, lk): val}
    # w_coeffs: {(li, lk): val}
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(I_map, J_map, K_map):
        """
        Adds 7 columns for a Strassen 2x2x2 decomposition.
        I_map: dict {local_i: global_i} (keys 0,1)
        J_map: dict {local_j: global_j} (keys 0,1)
        K_map: dict {local_k: global_k} (keys 0,1)
        """
        for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
            u_entries = []
            v_entries = []
            w_entries = []
            
            for (li, lj), val in u_coeffs.items():
                gi = I_map[li]
                gj = J_map[lj]
                u_entries.append((gi * m + gj, val))
                
            for (lj, lk), val in v_coeffs.items():
                gj = J_map[lj]
                gk = K_map[lk]
                v_entries.append((gj * p + gk, val))
                
            for (li, lk), val in w_coeffs.items():
                gi = I_map[li]
                gk = K_map[lk]
                w_entries.append((gk * n + gi, val))
                
            add_column(u_entries, v_entries, w_entries)

    def strassen_4x4x4(offset_i, offset_j, offset_k):
        """
        Adds 49 columns for a Strassen 4x4x4 decomposition.
        Uses 7 Strassen 2x2x2 blocks.
        """
        # Local 0,1,2,3 map to offset+0..3
        # Sub-block 0: [0,1], Sub-block 1: [2,3]
        I0 = [offset_i + 0, offset_i + 1]
        I1 = [offset_i + 2, offset_i + 3]
        J0 = [offset_j + 0, offset_j + 1]
        J1 = [offset_j + 2, offset_j + 3]
        K0 = [offset_k + 0, offset_k + 1]
        K1 = [offset_k + 2, offset_k + 3]
        
        # Strassen 4x4 formulas
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Each Mk is a 2x2 matrix product.
        # We decompose each Mk using Strassen 2x2.
        # For each Mk, we need to define the A, B, C coefficients for the 2x2 "virtual" matrices.
        # Then we combine them with the Strassen 2x2 rank-1 terms.
        
        # Define macro terms: (A_macros, B_macros, C_macros)
        # A_macros: {(si, sj): coeff}
        # B_macros: {(sj, sk): coeff}
        # C_macros: {(si, sk): coeff}
        
        macro_terms = [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for A_macros, B_macros, C_macros in macro_terms:
            # For this macro term, we generate 7 rank-1 columns (from Strassen 2x2)
            for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
                u_entries = []
                v_entries = []
                w_entries = []
                
                # Combine u_coeffs (2x2) with A_macros (2x2 of 2x2s)
                for (li, lj), u_val in u_coeffs.items():
                    # u_val is coefficient for local (li, lj) in the 2x2 virtual A
                    # A_macros gives coefficients for sub-blocks (si, sj)
                    for (si, sj), a_val in A_macros.items():
                        total_val = u_val * a_val
                        if total_val != 0:
                            gi = I0[li] if si == 0 else I1[li]
                            gj = J0[lj] if sj == 0 else J1[lj]
                            u_entries.append((gi * m + gj, total_val))
                            
                # Combine v_coeffs (2x2) with B_macros (2x2 of 2x2s)
                for (lj, lk), v_val in v_coeffs.items():
                    for (sj, sk), b_val in B_macros.items():
                        total_val = v_val * b_val
                        if total_val != 0:
                            gj = J0[lj] if sj == 0 else J1[lj]
                            gk = K0[lk] if sk == 0 else K1[lk]
                            v_entries.append((gj * p + gk, total_val))
                            
                # Combine w_coeffs (2x2) with C_macros (2x2 of 2x2s)
                for (li, lk), w_val in w_coeffs.items():
                    for (si, sk), c_val in C_macros.items():
                        total_val = w_val * c_val
                        if total_val != 0:
                            gi = I0[li] if si == 0 else I1[li]
                            gk = K0[lk] if sk == 0 else K1[lk]
                            w_entries.append((gk * n + gi, total_val))
                            
                add_column(u_entries, v_entries, w_entries)

    # Build all blocks
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                if _deadline_exceeded():
                    break
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 4 and len(j_list) == 4 and len(k_list) == 4:
                    strassen_4x4x4(i_list[0], j_list[0], k_list[0])
                else:
                    schoolbook_block(i_list, j_list, k_list)
            if _deadline_exceeded():
                break
        if _deadline_exceeded():
            break
    
    rank = len(columns)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
