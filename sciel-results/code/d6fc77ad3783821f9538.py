"""Matrix multiplication 4,5,6 with Strassen-based rank reduction."""
import numpy as np

DIMENSIONS = [4, 5, 6]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    
    Strategy: Split middle dimension 5 into 4+1.
    - C = A_top(4x4) @ B_top(4x6) + A_bottom(4x1) @ B_bottom(1x6)
    - For A_top @ B_top: split B_top into B_tl(4x4) + B_tr(4x2)
      - A_top @ B_tl: use Strassen rank 49 (4x4 x 4x4)
      - A_top @ B_tr: schoolbook rank 32 (4x4 x 4x2)
    - A_bottom @ B_bottom: schoolbook rank 24 (4x1 x 1x6)
    Total rank: 49 + 32 + 24 = 105
    """
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(list(u_vec))
        V_cols.append(list(v_vec))
        W_cols.append(list(w_vec))
    
    def zero_u():
        return [0] * (n * m)
    def zero_v():
        return [0] * (m * p)
    def zero_w():
        return [0] * (p * n)
    
    # Strassen 2x2 terms: (a_coeffs, b_coeffs, c_coeffs)
    # a = [a00, a01, a10, a11], b = [b00, b01, b10, b11], c = [c00, c01, c10, c11]
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1: (a00+a11)(b00+b11)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2: (a10+a11)b00
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3: a00(b01-b11)
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4: a11(b10-b00)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5: (a00+a10)b11
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6: (a11-a10)(b00+b10)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7: (a01-a11)(b10+b11)
    ]
    
    # Outer Strassen for 4x4 blocks (same structure as 2x2)
    # X_s = sum_b x_coeff[b] * A_block[b], Y_s = sum_b y_coeff[b] * B_block[b]
    # Z_s = X_s @ Y_s, contributes to C blocks with z_coeff
    outer_strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([0, 0, -1, 1], [1, 0, 1, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # Block element positions within 4x4 matrix
    # Block 0 (top-left): rows 0-1, cols 0-1
    # Block 1 (top-right): rows 0-1, cols 2-3
    # Block 2 (bottom-left): rows 2-3, cols 0-1
    # Block 3 (bottom-right): rows 2-3, cols 2-3
    
    def block_elements(base_row, base_col):
        """Return [(r, c)] for 2x2 block starting at (base_row, base_col)"""
        return [(base_row + dr, base_col + dc) for dr in range(2) for dc in range(2)]
    
    # For 4x4 @ 4x4 Strassen (rank 49):
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks of 2x2 matrices
    
    for s in range(7):
        x_coeff = outer_strassen[s][0]  # coeffs for A blocks [A00, A01, A10, A11]
        y_coeff = outer_strassen[s][1]  # coeffs for B blocks [B00, B01, B10, B11]
        z_coeff = outer_strassen[s][2]  # coeffs for C blocks [C00, C01, C10, C11]
        
        # X_s is a 2x2 matrix where X_s[pi,pj] = sum_b x_coeff[b] * A_block[b][pi,pj]
        # Y_s is a 2x2 matrix where Y_s[pi,pj] = sum_b y_coeff[b] * B_block[b][pi,pj]
        
        # Apply inner Strassen to compute Z_s = X_s @ Y_s
        for t in range(7):
            a_c = strassen_2x2[t][0]  # coeffs for X_s elements [x00,x01,x10,x11]
            b_c = strassen_2x2[t][1]  # coeffs for Y_s elements [y00,y01,y10,y11]
            c_c = strassen_2x2[t][2]  # coeffs for Z_s elements [z00,z01,z10,z11]
            
            # Build U vector for this term: linear combo of A elements
            u_vec = zero_u()
            # X_s[pi,pj] = sum_b x_coeff[b] * A_block[b][pi,pj]
            # The "a" part of inner Strassen: sum_{pi,pj} a_c[pi*2+pj] * X_s[pi,pj]
            # = sum_{pi,pj} a_c[pi*2+pj] * sum_b x_coeff[b] * A_block[b][pi,pj]
            # = sum_b x_coeff[b] * sum_{pi,pj} a_c[pi*2+pj] * A_block[b][pi,pj]
            
            for b in range(4):
                if x_coeff[b] == 0:
                    continue
                # A_block[b] is at position determined by b
                br = (b // 2) * 2  # base row
                bc = (b % 2) * 2   # base col
                for pi in range(2):
                    for pj in range(2):
                        idx = pi * 2 + pj
                        if a_c[idx] == 0:
                            continue
                        # A element at (br+pi, bc+pj) in the 4x4 sub-matrix
                        # In the full A (4x5), this is A[br+pi, bc+pj]
                        ai = br + pi
                        aj = bc + pj
                        coeff = x_coeff[b] * a_c[idx]
                        u_vec[ai * m + aj] += coeff
            
            # Build V vector: linear combo of B elements
            v_vec = zero_v()
            for b in range(4):
                if y_coeff[b] == 0:
                    continue
                br = (b // 2) * 2
                bc = (b % 2) * 2
                for pi in range(2):
                    for pj in range(2):
                        idx = pi * 2 + pj
                        if b_c[idx] == 0:
                            continue
                        # B element at (br+pi, bc+pj) in the 4x4 sub-matrix (B_top_left)
                        # In the full B (5x6), this is B[br+pi, bc+pj]
                        bi = br + pi
                        bj = bc + pj
                        coeff = y_coeff[b] * b_c[idx]
                        v_vec[bi * p + bj] += coeff
            
            # Build W vector: contribution to C elements
            w_vec = zero_w()
            # Z_s[pi,pj] contributes to C blocks with z_coeff
            # Z_s[pi,pj] is at C position: block determined by (pi,pj)
            # C_block[pi*2+pj_base, ...]
            # Z_s element at (pi, pj) belongs to C block at position (pi, pj) in the 2x2 block grid
            # C_block index = pi*2 + pj (same as A/B block indexing)
            for pi in range(2):
                for pj in range(2):
                    z_idx = pi * 2 + pj
                    if c_c[z_idx] == 0:
                        continue
                    # Z_s[pi,pj] is in C block at (pi, pj)
                    c_block_idx = pi * 2 + pj
                    if z_coeff[c_block_idx] == 0:
                        continue
                    # C block c_block_idx is at position:
                    cr = (c_block_idx // 2) * 2
                    cc = (c_block_idx % 2) * 2
                    # C element at (cr+pi, cc+pj) in the 4x4 sub-matrix
                    # In the full C (4x6), this is C[cr+pi, cc+pj]
                    ci = cr + pi
                    ck = cc + pj
                    coeff = c_c[z_idx] * z_coeff[c_block_idx]
                    # W index: k*n + i
                    w_vec[ck * n + ci] += coeff
            
            add_term(u_vec, v_vec, w_vec)
    
    # Part 2: A_top(4x4) @ B_tr(4x2), schoolbook rank 32
    # A_top[i,j] = A[i,j] for i in 0..3, j in 0..3
    # B_tr[j,k] = B[j,k] for j in 0..3, k in 4..5
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_vec = zero_u()
                v_vec = zero_v()
                w_vec = zero_w()
                u_vec[i * m + j] = 1
                v_vec[j * p + (4 + k)] = 1
                w_vec[(4 + k) * n + i] = 1
                add_term(u_vec, v_vec, w_vec)
    
    # Part 3: A_bottom(4x1) @ B_bottom(1x6), schoolbook rank 24
    # A_bottom[i,4] = A[i,4] for i in 0..3
    # B_bottom[4,k] = B[4,k] for k in 0..5
    # C[i,k] += A[i,4] * B[4,k]
    for i in range(4):
        for k in range(6):
            u_vec = zero_u()
            v_vec = zero_v()
            w_vec = zero_w()
            u_vec[i * m + 4] = 1
            v_vec[4 * p + k] = 1
            w_vec[k * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
    
    return {"U": U, "V": V, "W": W}
