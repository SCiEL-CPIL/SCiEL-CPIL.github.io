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
    The 4x4x4 diagonal block uses recursive Strassen (rank 49).
    The remaining 7 blocks use schoolbook multiplication.
    Total rank: 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110.
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
    
    # Standard Strassen 2x2x2 terms
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are dicts mapping (local_i, local_j) to value
    STRASSEN_2x2_TERMS = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_4x4x4(offset_i, offset_j, offset_k):
        """
        Adds 49 columns for a recursive Strassen 4x4x4 decomposition.
        offset_i, offset_j, offset_k: starting global indices (0, 0, 0).
        """
        # Map local 4x4 indices to global
        # Local block (a,b) where a,b in {0,1} corresponds to indices 2a+i, 2b+j
        # U_k[p,q] = sum_{a,b} u_ab^(k) A[2a+p, 2b+q]
        # V_k[r,s] = sum_{c,d} v_cd^(k) B[2c+r, 2d+s]
        
        # Precompute the linear forms for U_k and V_k for each outer term k
        # u_forms[k][p][q] is a dict of global (i,j) -> coeff
        # v_forms[k][r][s] is a dict of global (j,k) -> coeff
        
        u_forms = []
        v_forms = []
        
        for k_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(STRASSEN_2x2_TERMS):
            # u_coeffs maps (a,b) to val, where a,b are block indices {0,1}
            # We need U_k[p,q] for p,q in {0,1}
            # U_k[p,q] = sum_{a,b} u_coeffs[(a,b)] * A[2a+p, 2b+q]
            u_form = [[{} for _ in range(2)] for _ in range(2)]
            for (a, b), val in u_coeffs.items():
                for p in range(2):
                    for q in range(2):
                        gi = offset_i + 2*a + p
                        gj = offset_j + 2*b + q
                        u_form[p][q][gi * m + gj] = u_form[p][q].get(gi * m + gj, 0) + val
            
            # v_coeffs maps (c,d) to val
            # V_k[r,s] = sum_{c,d} v_coeffs[(c,d)] * B[2c+r, 2d+s]
            v_form = [[{} for _ in range(2)] for _ in range(2)]
            for (c, d), val in v_coeffs.items():
                for r in range(2):
                    for s in range(2):
                        gj = offset_j + 2*c + r
                        gk = offset_k + 2*d + s
                        v_form[r][s][gj * p + gk] = v_form[r][s].get(gj * p + gk, 0) + val
            
            u_forms.append(u_form)
            v_forms.append(v_form)
            
        # Now generate the 49 columns
        for k_idx in range(7):
            # For each outer term k, apply inner Strassen
            for l_idx, (u_inner, v_inner, w_inner) in enumerate(STRASSEN_2x2_TERMS):
                # u_inner maps (p,q) to val
                # U'_l = sum_{p,q} u_inner[(p,q)] * U_k[p,q]
                # U_k[p,q] is a linear form in A
                u_entries = []
                for (p, q), val_u in u_inner.items():
                    # Add val_u * U_k[p,q]
                    for idx, val_ak in u_forms[k_idx][p][q].items():
                        u_entries.append((idx, val_u * val_ak))
                
                # v_inner maps (r,s) to val
                # V'_l = sum_{r,s} v_inner[(r,s)] * V_k[r,s]
                v_entries = []
                for (r, s), val_v in v_inner.items():
                    # Add val_v * V_k[r,s]
                    for idx, val_bk in v_forms[k_idx][r][s].items():
                        v_entries.append((idx, val_v * val_bk))
                
                # w_inner maps (i,j) to val (local indices for the 2x2 result M_kl)
                # Contribution to C:
                # C[2e+i, 2f+j] += w_outer[(e,f)] * w_inner[(i,j)] * S_kl
                # Wait, the outer W coefficients w_outer map (e,f) to val.
                # The result of the inner multiplication is a 2x2 matrix M_kl.
                # M_kl[i,j] is the (i,j) entry.
                # The outer term k contributes w_outer[(e,f)] * M_k[i,j] to C[2e+i, 2f+j]?
                # No, M_k is the 2x2 block product.
                # C_block(e,f) = sum_k w_outer[(e,f)] * M_k.
                # M_k is 2x2. So C[2e+i, 2f+j] = sum_k w_outer[(e,f)] * M_k[i,j].
                # And M_k = sum_l M_kl.
                # M_kl[i,j] = sum_{inner_terms} w_inner[(i,j)] * S_l.
                # So C[2e+i, 2f+j] = sum_k sum_l w_outer[(e,f)] * w_inner[(i,j)] * S_kl.
                
                w_entries = []
                outer_w_coeffs = STRASSEN_2x2_TERMS[k_idx][2]
                
                for (e, f), val_wo in outer_w_coeffs.items():
                    for (i, j), val_wi in w_inner.items():
                        gi = offset_i + 2*e + i
                        gk = offset_k + 2*f + j
                        # W index is (k, i) in the tensor, mapped to k*n + i
                        w_entries.append((gk * n + gi, val_wo * val_wi))
                
                add_column(u_entries, v_entries, w_entries)

    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
                    
    # Block index groups: 4+1 partition
    blocks = [[0, 1, 2, 3], [4]]
    
    # Build all 8 blocks
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                if _deadline_exceeded():
                    break
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 4 and len(j_list) == 4 and len(k_list) == 4:
                    add_strassen_4x4x4(i_list[0], j_list[0], k_list[0])
                else:
                    schoolbook_block(i_list, j_list, k_list)
            if _deadline_exceeded():
                break
        if _deadline_exceeded():
            break
    
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
