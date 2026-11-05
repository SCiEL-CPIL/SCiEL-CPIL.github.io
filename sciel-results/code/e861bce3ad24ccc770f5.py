import numpy as np
import time

_START = time.time()
_DEADLINE = 120.0


def construct():
    n, m, p = 4, 5, 5  # A is n x m, B is m x p, C is n x p
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # We decompose the 4x5x5 multiplication into:
    # 1. 4x4x4 core (A[:, 0:4] * B[0:4, 0:4]) -> C[:, 0:4] : 49 terms via nested Strassen
    # 2. 4x4x1 cross (A[:, 0:4] * B[0:4, 4:5]) -> C[:, 4:5] : 16 terms schoolbook
    # 3. 4x1x5 cross (A[:, 4:5] * B[4:5, 0:5]) -> C[:, 0:5] : 20 terms schoolbook
    # Total: 49 + 16 + 20 = 85 terms
    
    # --- Part 1: 4x4x4 via nested Strassen (49 terms) ---
    # 4x4 matrices split into 2x2 blocks. Each 2x2x2 uses Strassen (7 mults).
    # Outer Strassen combines the 4 blocks with 7 mults.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Outer Strassen 2x2 block scheme for 4x4 (indices of 2x2 blocks: 0=TL,1=TR,2=BL,3=BR)
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01], [B10, B11]] (each 2x2)
    # C = A*B, C blocks computed via Strassen on the block-level 2x2x2
    
    outer_strassen_A = {
        0: [0, 1],   # M1 = (A00+A11)
        1: [1, 3],   # M2 = (A10+A11)
        2: [0],      # M3 = A00
        3: [1],      # M4 = A11
        4: [0, 2],   # M5 = (A00+A01)
        5: [1, 2],   # M6 = (A10-A00) -- wait, let me use standard Strassen
        6: [2, 3],   # M7 = (A11-A10)
    }
    # Standard Strassen for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B11)
    # M4 = A22 * (B12 - B22)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A22 - A21) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me define the 7 Strassen products for 2x2x2 explicitly:
    # Each product multiplies a linear form in A (2x2) by a linear form in B (2x2)
    # and contributes to C (2x2) with signed entries.
    
    strassen_2x2 = [
        # (A_form, B_form, C_contrib)
        # A_form: dict {(li, lk): coeff}, B_form: dict {(lk, lj): coeff}, C_contrib: dict {(li, lj): coeff}
        # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)*B00 -> C10 += 1, C11 += -1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00*(B01-B11) -> C01 += 1, C11 += 1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = A11*(B10-B00) -> C00 += 1, C10 += 1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)*B11 -> C00 += -1, C01 += 1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)*(B00+B01) -> C11 += 1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)*(B10+B11) -> C00 += 1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For the outer Strassen, we treat the 4x4 as 2x2 blocks of size 2.
    # Block (bi, bk) of A: rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    # Block (bk, bj) of B: rows 2*bk..2*bk+1, cols 2*bj..2*bj+1
    # Block (bi, bj) of C: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    # Outer Strassen 7 products for block-level 2x2x2:
    outer_products = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: dict {block_idx: sign}, B_blocks: dict {block_idx: sign}, C_blocks: dict {block_idx: sign}
        # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
        # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2 = (A20+A21)*B00 -> C20 += 1, C21 += -1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # M3 = A00*(B10-B00) -> C01 += 1, C11 += 1
        ({0:1}, {2:1, 0:-1}, {1:1, 3:1}),
        # M4 = A11*(B01-B11) -> C00 += 1, C20 += 1
        ({3:1}, {1:1, 3:-1}, {0:1, 2:1}),
        # M5 = (A00+A01)*B11 -> C00 += -1, C01 += 1
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # M6 = (A20-A00)*(B00+B01) -> C11 += 1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # M7 = (A01-A11)*(B20+B21) -> C00 += 1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # For each outer product (7), for each inner Strassen term (7), we get one rank-1 term.
    # Total: 7 * 7 = 49 terms.
    
    def block_to_global_A(bi, bk, li, lk):
        """Convert block (bi,bk) local (li,lk) to global A index."""
        gi = 2 * bi + li
        gk = 2 * bk + lk
        return gi, gk
    
    def block_to_global_B(bk, bj, lk, lj):
        """Convert block (bk,bj) local (lk,lj) to global B index."""
        gk = 2 * bk + lk
        gj = 2 * bj + lj
        return gk, gj
    
    def block_to_global_C(bi, bj, li, lj):
        """Convert block (bi,bj) local (li,lj) to global C index."""
        gi = 2 * bi + li
        gj = 2 * bj + lj
        return gi, gj
    
    for outer_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_products):
        for inner_idx, (a_inner, b_inner, c_inner) in enumerate(strassen_2x2):
            u_vec = [0.0] * (n * m)  # 4*5 = 20
            v_vec = [0.0] * (m * p)  # 5*5 = 25
            w_vec = [0.0] * (p * n)  # 5*4 = 20 (W index: k*n + i)
            
            # Build U vector
            for ab_idx, ab_sign in a_blocks.items():
                ab_i = ab_idx // 2
                ab_k = ab_idx % 2
                for (li, lk), inner_sign in a_inner.items():
                    gi, gk = block_to_global_A(ab_i, ab_k, li, lk)
                    if gk < 4:  # only columns 0-3 for the core
                        u_vec[gi * m + gk] += ab_sign * inner_sign
            
            # Build V vector
            for bb_idx, bb_sign in b_blocks.items():
                bb_k = bb_idx // 2
                bb_j = bb_idx % 2
                for (lk, lj), inner_sign in b_inner.items():
                    gk, gj = block_to_global_B(bb_k, bb_j, lk, lj)
                    if gk < 4:  # only rows 0-3 for the core
                        v_vec[gk * p + gj] += bb_sign * inner_sign
            
            # Build W vector (index: k*n + i where k is col of C, i is row)
            for cb_idx, cb_sign in c_blocks.items():
                cb_i = cb_idx // 2
                cb_j = cb_idx % 2
                for (li, lj), inner_sign in c_inner.items():
                    gi, gj = block_to_global_C(cb_i, cb_j, li, lj)
                    # W index: gj * n + gi (col * n + row)
                    w_vec[gj * n + gi] += cb_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
    
    # --- Part 2: 4x4x1 cross terms (A[:, 0:4] * B[0:4, 4:5] -> C[:, 4:5]) ---
    # Schoolbook: 4 * 4 = 16 terms
    for i in range(4):
        for k in range(4):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + k] = 1.0
            v_vec[k * p + 4] = 1.0
            w_vec[4 * n + i] = 1.0  # col=4, row=i
            add_rank1(u_vec, v_vec, w_vec)
    
    # --- Part 3: 4x1x5 cross terms (A[:, 4:5] * B[4:5, 0:5] -> C[:, 0:5]) ---
    # Schoolbook: 4 * 5 = 20 terms
    for i in range(4):
        for j in range(5):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[i * m + 4] = 1.0
            v_vec[4 * p + j] = 1.0
            w_vec[j * n + i] = 1.0
            add_rank1(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    
    # Safety: if somehow no terms, add one
    if rank == 0:
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[0] = 1.0
        v_vec[0] = 1.0
        w_vec[0] = 1.0
        add_rank1(u_vec, v_vec, w_vec)
    
    U = np.array(U_cols).T  # shape (n*m, rank)
    V = np.array(V_cols).T  # shape (m*p, rank)
    W = np.array(W_cols).T  # shape (p*n, rank)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
