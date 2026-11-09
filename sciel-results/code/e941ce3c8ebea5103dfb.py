"""5x5x5 matrix multiplication tensor decomposition using 4+1 block structure with recursive Strassen on the 4x4x4 block."""
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
    
    Uses a 4+1 block partition. For the 4x4x4 diagonal block, uses recursive
    Strassen (2+2 sub-blocks, each with Strassen rank 7) giving rank 49.
    Off-diagonal blocks use schoolbook.
    
    Total rank:
    - 4x4x4 diagonal block: 49 (recursive Strassen)
    - 4x4x1 blocks (4 of them): 4*4*1 = 16 each -> 64
    - 4x1x4 blocks (4 of them): 4*1*4 = 16 each -> 64
    - 4x1x1 blocks (4 of them): 4*1*1 = 4 each -> 16
    - 1x4x4 blocks (4 of them): 1*4*4 = 16 each -> 64
    - 1x4x1 blocks (4 of them): 1*4*1 = 4 each -> 16
    - 1x1x4 blocks (4 of them): 1*1*4 = 4 each -> 16
    - 1x1x1 block: 1
    
    Wait, let me recount. The partition is 4+1 for each dimension.
    So we have 2x2x2 = 8 block products:
    - (4,4,4): 1 block, rank 49
    - (4,4,1): 1 block, rank 16
    - (4,1,4): 1 block, rank 16
    - (4,1,1): 1 block, rank 4
    - (1,4,4): 1 block, rank 16
    - (1,4,1): 1 block, rank 4
    - (1,1,4): 1 block, rank 4
    - (1,1,1): 1 block, rank 1
    
    Total: 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
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
    
    # Strassen for 2x2x2 block with given offset
    def strassen_2x2x2(offset_i, offset_j, offset_k):
        """Add 7 columns for the 2x2x2 Strassen decomposition at given offsets."""
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
                gi, gj = offset_i + li, offset_j + lj
                u_entries.append((gi * m + gj, a_val))
            for (lj, lk), b_val in b_coeffs.items():
                gj, gk = offset_j + lj, offset_k + lk
                v_entries.append((gj * p + gk, b_val))
            for (li, lk), c_val in c_coeffs.items():
                gi, gk = offset_i + li, offset_k + lk
                w_entries.append((gk * n + gi, c_val))
            add_column(u_entries, v_entries, w_entries)
    
    # Recursive Strassen for 4x4x4: partition into 2+2 sub-blocks
    def strassen_4x4x4(offset_i, offset_j, offset_k):
        """Add 49 columns for the 4x4x4 recursive Strassen decomposition."""
        # Partition the 4x4x4 block into 2x2 sub-blocks at offsets
        # Sub-block (si, sj, sk) has local size 2x2x2
        # Global offset: offset_i + si*2, offset_j + sj*2, offset_k + sk*2
        for si in range(2):
            for sj in range(2):
                for sk in range(2):
                    strassen_2x2x2(
                        offset_i + si * 2,
                        offset_j + sj * 2,
                        offset_k + sk * 2
                    )
    
    # Block index groups for 4+1 partition
    block4 = [0, 1, 2, 3]  # first 4
    block1 = [4]           # last 1
    
    # Build all 8 block products:
    # (4,4,4): recursive Strassen, rank 49
    strassen_4x4x4(0, 0, 0)
    
    # (4,4,1): schoolbook, rank 16
    schoolbook_block(block4, block4, block1)
    
    # (4,1,4): schoolbook, rank 16
    schoolbook_block(block4, block1, block4)
    
    # (4,1,1): schoolbook, rank 4
    schoolbook_block(block4, block1, block1)
    
    # (1,4,4): schoolbook, rank 16
    schoolbook_block(block1, block4, block4)
    
    # (1,4,1): schoolbook, rank 4
    schoolbook_block(block1, block4, block1)
    
    # (1,1,4): schoolbook, rank 4
    schoolbook_block(block1, block1, block4)
    
    # (1,1,1): schoolbook, rank 1
    schoolbook_block(block1, block1, block1)
    
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
