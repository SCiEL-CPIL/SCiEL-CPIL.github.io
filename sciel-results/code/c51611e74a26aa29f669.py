"""5x5x5 matrix multiplication tensor decomposition using 4+1 block structure with recursive Strassen (rank 110)."""
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
    
    Uses a 4+1 block partition.
    For the 4x4x4 block, uses recursive Strassen (rank 49 instead of 64).
    For mixed-size blocks involving the single row/col, uses schoolbook.
    
    Block partition: I = [0,1,2,3], [4]; same for J, K.
    
    2x2x2 = 8 blocks total.
    - 1 block is 4x4x4 -> recursive Strassen (49)
    - 7 blocks involve at least one group of size 1.
    
    Let's count schoolbook terms for the 7 mixed blocks:
    The groups are G0 (size 4) and G1 (size 1).
    Blocks:
    (G0, G0, G1): 4x4x1 = 16 terms
    (G0, G1, G0): 4x1x4 = 16 terms
    (G1, G0, G0): 1x4x4 = 16 terms
    (G0, G1, G1): 4x1x1 = 4 terms
    (G1, G0, G1): 1x4x1 = 4 terms
    (G1, G1, G0): 1x1x4 = 4 terms
    (G1, G1, G1): 1x1x1 = 1 term
    
    Total schoolbook terms = 16 + 16 + 16 + 4 + 4 + 4 + 1 = 61.
    Total rank = 49 (Strassen) + 61 (schoolbook) = 110.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            if val == 0: continue
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            if val == 0: continue
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            if val == 0: continue
            w_d[idx] = w_d.get(idx, 0) + val
        
        # Filter out zeros
        u_d = {k: v for k, v in u_d.items() if v != 0}
        v_d = {k: v for k, v in v_d.items() if v != 0}
        w_d = {k: v for k, v in w_d.items() if v != 0}
        
        if not u_d or not v_d or not w_d:
            return
            
        columns.append((u_d, v_d, w_d))
    
    # Block index groups
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
                    
    # Standard Strassen 2x2x2 terms
    # u: (i,j), v: (j,k), w: (k,i)
    # Indices are local to the 2x2 block: 0, 1
    STRASSEN_2x2_TERMS = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(I_map, J_map, K_map):
        """
        Adds 7 columns for a Strassen 2x2x2 decomposition.
        I_map: list [gi0, gi1]
        J_map: list [gj0, gj1]
        K_map: list [gk0, gk1]
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

    def add_strassen_4x4x4(I_map, J_map, K_map):
        """
        Adds 49 columns for a recursive Strassen 4x4x4 decomposition.
        Partitions I_map, J_map, K_map into two halves.
        """
        I0 = I_map[0:2]
        I1 = I_map[2:4]
        J0 = J_map[0:2]
        J1 = J_map[2:4]
        K0 = K_map[0:2]
        K1 = K_map[2:4]
        
        # High-level Strassen terms for 2x2 blocks of 2x2 matrices
        # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2 blocks
        # B = [[B00, B01], [B10, B11]]
        # C = A * B
        
        # M1 = (A00+A11)(B00+B11)
        # A00+A11 involves indices I0, J0 and I1, J1
        # (A00+A11) = A00 + A11
        # (B00+B11) = B00 + B11
        # Product = A00B00 + A00B11 + A11B00 + A11B11
        # We need to express this as a sum of 7 terms? No, we use the 7 terms of Strassen on the 2x2 of blocks.
        # But the entries of the "block matrix" are 2x2 matrices.
        # So we just apply the Strassen algorithm recursively.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00 + A11 -> indices in I0, J0 and I1, J1
        # V: B00 + B11 -> indices in J0, K0 and J1, K1
        # W: C00 + C11 -> indices in K0, I0 and K1, I1
        
        # We can just call add_strassen_2x2x2 for each of the 4 block products? No, that would be 4*7 = 28 terms, but they share the multipliers.
        # Actually, the recursive Strassen for 4x4x4 uses 7 multipliers of size 2x2x2.
        # Each multiplier is a sum of 2x2x2 Strassen terms? No.
        # Each multiplier is a product of two 2x2 matrices.
        # So we need to decompose each of the 7 high-level products into 7 low-level products? No, that's not right.
        # The high-level product is (A00+A11)(B00+B11).
        # This is a single multiplication of two 2x2 matrices.
        # We can decompose this single multiplication using Strassen (rank 7).
        # So for each of the 7 high-level terms, we add 7 low-level terms.
        # Total 7 * 7 = 49 terms.
        
        # High-level terms definitions (in terms of block indices 0,1):
        # 0: (A00+A11)(B00+B11)
        # 1: (A10+A11)B00
        # 2: A00(B01-B11)
        # 3: A11(B10-B00)
        # 4: (A00+A01)B11
        # 5: (A10-A00)(B00+B01)
        # 6: (A01-A11)(B10+B11)
        
        # We need to map these to the actual indices.
        # A00 is at I0, J0. A11 is at I1, J1. etc.
        
        # Let's define the high-level terms as lists of (coeff, I_block, J_block) for U, etc.
        
        high_level_terms = [
            # (u_coeffs, v_coeffs, w_coeffs) where keys are (I_block, J_block) etc.
            # 0: (A00+A11)(B00+B11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # 1: (A10+A11)B00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # 2: A00(B01-B11)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # 3: A11(B10-B00)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # 4: (A00+A01)B11
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # 5: (A10-A00)(B00+B01)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # 6: (A01-A11)(B10+B11)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in high_level_terms:
            # For this high-level term, we have a "virtual" 2x2 matrix multiplication.
            # U_virtual is a 2x2 matrix where entry (bi, bj) is the sum of u_coeffs[(bi,bj)] * A_block(bi,bj)
            # V_virtual is a 2x2 matrix where entry (bj, bk) is the sum of v_coeffs[(bj,bk)] * B_block(bj,bk)
            # W_virtual is a 2x2 matrix where entry (bk, bi) is the sum of w_coeffs[(bk,bi)] * C_block(bk,bi)
            
            # We need to decompose U_virtual * V_virtual = W_virtual using Strassen.
            # This means we apply the 7 low-level Strassen terms to the virtual matrices.
            
            # The low-level Strassen terms are defined for 2x2 matrices.
            # Here, the "entries" of the virtual matrices are 2x2 blocks.
            # So we need to expand each low-level term.
            
            # Low-level Strassen terms (same as STRASSEN_2x2_TERMS)
            # Let's iterate over them.
            for lu_coeffs, lv_coeffs, lw_coeffs in STRASSEN_2x2_TERMS:
                # lu_coeffs: {(li, lj): val}
                # lv_coeffs: {(lj, lk): val}
                # lw_coeffs: {(li, lk): val}
                
                u_entries = []
                v_entries = []
                w_entries = []
                
                # Expand U
                # U_virtual(li, lj) = sum_{(bi, bj) in lu_coeffs} lu_coeffs[(li, lj)] * (sum_{(hi, hj) in u_coeffs} u_coeffs[(hi, hj)] * A_block(hi, hj))
                # Wait, the indices are getting confusing.
                # Let's use different names.
                # High-level block indices: hi, hj (for A), hj, hk (for B), hk, hi (for C).
                # Low-level indices within block: li, lj, lk.
                
                # U_virtual is a 2x2 matrix of blocks.
                # Entry (li, lj) of U_virtual is a 2x2 matrix.
                # U_virtual(li, lj) = sum_{(hi, hj) in lu_coeffs} lu_coeffs[(li, lj)] * A_block(hi, hj) * (coefficient from high-level u_coeffs? No.)
                
                # Actually, the high-level term defines U_virtual as a linear combination of A_blocks.
                # U_virtual = sum_{(hi, hj)} u_coeffs[(hi, hj)] * A_block(hi, hj)
                # This is a 2x2 matrix (the virtual one).
                # Then we apply Strassen to this virtual 2x2 matrix.
                # The Strassen algorithm computes 7 products of linear combinations of entries of U_virtual.
                # So for each low-level term (which is a product of two linear combinations of entries of U_virtual and V_virtual),
                # we substitute the definition of U_virtual and V_virtual.
                
                # Low-level term: M_l = (sum_{(li, lj)} lu_coeffs[(li, lj)] * U_virtual(li, lj)) * (sum_{(lj, lk)} lv_coeffs[(lj, lk)] * V_virtual(lj, lk))
                # Substitute U_virtual and V_virtual:
                # M_l = (sum_{(li, lj)} lu_coeffs[(li, lj)] * sum_{(hi, hj)} u_coeffs[(hi, hj)] * A_block(hi, hj)) * (sum_{(lj, lk)} lv_coeffs[(lj, lk)] * sum_{(hj, hk)} v_coeffs[(hj, hk)] * B_block(hj, hk))
                # This is a product of two sums. We can expand it into a sum of products of A_blocks and B_blocks.
                # But we want to add columns to the global tensor.
                # Each product A_block(hi, hj) * B_block(hj, hk) is a 2x2x2 tensor.
                # We can decompose each such product using Strassen? No, that's what we are doing.
                # We are decomposing the 4x4x4 tensor.
                # The term M_l is a sum of terms of the form A_block(hi, hj) * B_block(hj, hk).
                # We need to add a column for each such term?
                # No, the column corresponds to the entire product M_l.
                # So U_entry for this column is the sum of all A_block(hi, hj) involved.
                # V_entry is the sum of all B_block(hj, hk) involved.
                # W_entry is the sum of all C_block(hk, hi) involved.
                
                # Let's compute the coefficients for each global index.
                
                # U part:
                # For each (li, lj) in lu_coeffs, and for each (hi, hj) in u_coeffs:
                # coeff = lu_coeffs[(li, lj)] * u_coeffs[(hi, hj)]
                # Global index: I_map[hi*2+li] * m + J_map[hj*2+lj]
                
                # V part:
                # For each (lj, lk) in lv_coeffs, and for each (hj, hk) in v_coeffs:
                # coeff = lv_coeffs[(lj, lk)] * v_coeffs[(hj, hk)]
                # Global index: J_map[hj*2+lj] * p + K_map[hk*2+lk]
                
                # W part:
                # For each (li, lk) in lw_coeffs, and for each (hk, hi) in w_coeffs:
                # coeff = lw_coeffs[(li, lk)] * w_coeffs[(hk, hi)]
                # Global index: K_map[hk*2+lk] * n + I_map[hi*2+li]
                
                # Note: The indices li, lj, lk are 0 or 1.
                # hi, hj, hk are 0 or 1.
                
                for (li, lj), lu_val in lu_coeffs.items():
                    for (hi, hj), u_val in u_coeffs.items():
                        gi = I_map[hi*2 + li]
                        gj = J_map[hj*2 + lj]
                        u_entries.append((gi * m + gj, lu_val * u_val))
                        
                for (lj, lk), lv_val in lv_coeffs.items():
                    for (hj, hk), v_val in v_coeffs.items():
                        gj = J_map[hj*2 + lj]
                        gk = K_map[hk*2 + lk]
                        v_entries.append((gj * p + gk, lv_val * v_val))
                        
                for (li, lk), lw_val in lw_coeffs.items():
                    for (hk, hi), w_val in w_coeffs.items():
                        gi = I_map[hi*2 + li]
                        gk = K_map[hk*2 + lk]
                        w_entries.append((gk * n + gi, lw_val * w_val))
                        
                add_column(u_entries, v_entries, w_entries)

    # Iterate over all blocks
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 4 and len(j_list) == 4 and len(k_list) == 4:
                    # Recursive Strassen block
                    I_map = i_list
                    J_map = j_list
                    K_map = k_list
                    add_strassen_4x4x4(I_map, J_map, K_map)
                else:
                    # Schoolbook block
                    schoolbook_block(i_list, j_list, k_list)
    
    rank = len(columns)
    
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
