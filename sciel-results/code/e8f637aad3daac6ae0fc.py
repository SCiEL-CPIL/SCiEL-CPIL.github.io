"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 49 for 4x4x4 (Double Strassen) + Rank 16 for 4x4x1 (Schoolbook)
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    t = 0
    
    # Helper to add a single rank-1 term
    # u_list: list of (index, coeff)
    # v_list: list of (index, coeff)
    # w_list: list of (index, coeff)
    def add_term(u_list, v_list, w_list, current_t):
        for idx, c in u_list:
            U[idx, current_t] += c
        for idx, c in v_list:
            V[idx, current_t] += c
        for idx, c in w_list:
            W[idx, current_t] += c

    # Helper to add Strassen 2x2x2 decomposition
    # a_comb: 2x2 list of lists of (global_idx, coeff)
    # b_comb: 2x2 list of lists of (global_idx, coeff)
    # c_comb: 2x2 list of lists of (global_idx, coeff)
    def add_strassen_2x2(a_comb, b_comb, c_comb, current_t):
        # Strassen terms for Z = XY
        # m1 = (x00+x11)(y00+y11) -> z00 += m1, z11 += m1
        # m2 = (x10+x11)y00 -> z10 += m2, z11 -= m2
        # m3 = x00(y01-y11) -> z01 += m3, z11 += m3
        # m4 = x11(y10-y00) -> z00 += m4, z10 += m4
        # m5 = (x00+x01)y11 -> z00 -= m5, z01 += m5
        # m6 = (x10-x00)(y00+y01) -> z11 += m6
        # m7 = (x01-x11)(y10+y11) -> z00 += m7
        
        # We need to expand each product of linear forms.
        
        # m1
        # (a00 + a11)(b00 + b11)
        # a00*b00, a00*b11, a11*b00, a11*b11
        # All contribute to c00 (+1) and c11 (+1)
        for u in a_comb[0][0]:
            for v in b_comb[0][0]:
                add_term([u], [v], c_comb[0][0] + c_comb[1][1], current_t)
                current_t += 1
        for u in a_comb[0][0]:
            for v in b_comb[1][1]:
                add_term([u], [v], c_comb[0][0] + c_comb[1][1], current_t)
                current_t += 1
        for u in a_comb[1][1]:
            for v in b_comb[0][0]:
                add_term([u], [v], c_comb[0][0] + c_comb[1][1], current_t)
                current_t += 1
        for u in a_comb[1][1]:
            for v in b_comb[1][1]:
                add_term([u], [v], c_comb[0][0] + c_comb[1][1], current_t)
                current_t += 1
                
        # m2
        # (a10 + a11)b00
        # a10*b00, a11*b00
        # Contribute to c10 (+1) and c11 (-1)
        for u in a_comb[1][0]:
            for v in b_comb[0][0]:
                add_term([u], [v], c_comb[1][0] + [(idx, -c) for idx, c in c_comb[1][1]], current_t)
                current_t += 1
        for u in a_comb[1][1]:
            for v in b_comb[0][0]:
                add_term([u], [v], c_comb[1][0] + [(idx, -c) for idx, c in c_comb[1][1]], current_t)
                current_t += 1

        # m3
        # a00(b01 - b11)
        # a00*b01, a00*(-b11)
        # Contribute to c01 (+1) and c11 (+1)
        for u in a_comb[0][0]:
            for v in b_comb[0][1]:
                add_term([u], [v], c_comb[0][1] + c_comb[1][1], current_t)
                current_t += 1
        for u in a_comb[0][0]:
            for v in b_comb[1][1]:
                add_term([u], [(-idx, -c) for idx, c in [v]], c_comb[0][1] + c_comb[1][1], current_t)
                current_t += 1
                # Note: v is (idx, c). We want -v.
                # My add_term takes lists of tuples.
                # So for -b11, the v_list should be [(idx, -c)]
        
        # Let's refine the helper to be more flexible or just inline the logic.
        # The current add_term adds to the same t.
        # I need to be careful with the coefficients.
        
        # Let's restart the inner logic of add_strassen_2x2 with explicit coefficient handling.
        
        # Actually, the above logic is getting messy with the list modifications.
        # Let's just write out the 7 Strassen terms explicitly for the 2x2 case.
        # There are 7 terms. Each term is a product of two linear forms.
        # Linear form A: sum of a_i * A_i
        # Linear form B: sum of b_j * B_j
        # Product: sum_{i,j} a_i b_j A_i B_j
        # Each A_i B_j is a rank-1 term in U,V.
        # The result contributes to W with coefficients.
        
        pass # Re-implementing below

    # Re-implementing add_strassen_2x2 cleanly
    def add_strassen_2x2_clean(a_comb, b_comb, c_comb, start_t):
        t = start_t
        
        # Helper to expand product of two linear forms and add to W
        # u_list: list of (idx, coeff)
        # v_list: list of (idx, coeff)
        # w_list: list of (idx, coeff)
        def expand_and_add(u_list, v_list, w_list, t_val):
            for i, (u_idx, u_c) in enumerate(u_list):
                for j, (v_idx, v_c) in enumerate(v_list):
                    # Add term
                    U[u_idx, t_val] += u_c
                    V[v_idx, t_val] += v_c
                    for w_idx, w_c in w_list:
                        W[w_idx, t_val] += w_c
                    t_val += 1
            return t_val

        # m1 = (a00+a11)(b00+b11)
        # u = a00+a11, v = b00+b11
        # w = c00 (+1), c11 (+1)
        u_list = a_comb[0][0] + a_comb[1][1]
        v_list = b_comb[0][0] + b_comb[1][1]
        w_list = c_comb[0][0] + c_comb[1][1]
        t = expand_and_add(u_list, v_list, w_list, t)

        # m2 = (a10+a11)b00
        # u = a10+a11, v = b00
        # w = c10 (+1), c11 (-1)
        u_list = a_comb[1][0] + a_comb[1][1]
        v_list = b_comb[0][0]
        w_list = c_comb[1][0] + [(idx, -c) for idx, c in c_comb[1][1]]
        t = expand_and_add(u_list, v_list, w_list, t)

        # m3 = a00(b01-b11)
        # u = a00, v = b01-b11
        # w = c01 (+1), c11 (+1)
        u_list = a_comb[0][0]
        v_list = b_comb[0][1] + [(idx, -c) for idx, c in b_comb[1][1]]
        w_list = c_comb[0][1] + c_comb[1][1]
        t = expand_and_add(u_list, v_list, w_list, t)

        # m4 = a11(b10-b00)
        # u = a11, v = b10-b00
        # w = c00 (+1), c10 (+1)
        u_list = a_comb[1][1]
        v_list = b_comb[1][0] + [(idx, -c) for idx, c in b_comb[0][0]]
        w_list = c_comb[0][0] + c_comb[1][0]
        t = expand_and_add(u_list, v_list, w_list, t)

        # m5 = (a00+a01)b11
        # u = a00+a01, v = b11
        # w = c00 (-1), c01 (+1)
        u_list = a_comb[0][0] + a_comb[0][1]
        v_list = b_comb[1][1]
        w_list = [(idx, -c) for idx, c in c_comb[0][0]] + c_comb[0][1]
        t = expand_and_add(u_list, v_list, w_list, t)

        # m6 = (a10-a00)(b00+b01)
        # u = a10-a00, v = b00+b01
        # w = c11 (+1)
        u_list = a_comb[1][0] + [(idx, -c) for idx, c in a_comb[0][0]]
        v_list = b_comb[0][0] + b_comb[0][1]
        w_list = c_comb[1][1]
        t = expand_and_add(u_list, v_list, w_list, t)

        # m7 = (a01-a11)(b10+b11)
        # u = a01-a11, v = b10+b11
        # w = c00 (+1)
        u_list = a_comb[0][1] + [(idx, -c) for idx, c in a_comb[1][1]]
        v_list = b_comb[1][0] + b_comb[1][1]
        w_list = c_comb[0][0]
        t = expand_and_add(u_list, v_list, w_list, t)

        return t

    # Define global indices for A, B, C blocks
    # A: 4x4, B: 4x5 (we use 4x4 part), C: 5x4 (we use 4x4 part)
    # Block (br, bc) of size 2x2
    # A[br*2+i, bc*2+j] -> idx (br*2+i)*4 + (bc*2+j)
    # B[br*2+i, bc*2+j] -> idx (br*2+i)*5 + (bc*2+j)  (Note: B is 4x5, so col is 0..3 for this part)
    # C[br*2+i, bc*2+j] -> idx (bc*2+j)*4 + (br*2+i)  (Note: C is 5x4, so row is 0..3 for this part)
    
    def get_a_comb(br, bc, coeff=1.0):
        # Returns list of (idx, coeff) for A[br*2..br*2+1, bc*2..bc*2+1]
        res = []
        for i in range(2):
            for j in range(2):
                idx = (br*2+i)*4 + (bc*2+j)
                res.append((idx, coeff))
        return res

    def get_b_comb(br, bc, coeff=1.0):
        res = []
        for i in range(2):
            for j in range(2):
                idx = (br*2+i)*5 + (bc*2+j)
                res.append((idx, coeff))
        return res

    def get_c_comb(br, bc, coeff=1.0):
        res = []
        for i in range(2):
            for j in range(2):
                # C row = bc*2+i, C col = br*2+j ?
                # C = A B. A is 4x4, B is 4x4. C is 4x4.
                # C[i,k] = sum_j A[i,j] B[j,k].
                # In block form:
                # C[br, bc] = sum_... 
                # Wait, the block indices for C correspond to (row_block, col_block).
                # Row block br, Col block bc.
                # C[br*2+i, bc*2+j].
                # W index = k*n + i.
                # Here k is the column of C (which is bc*2+j).
                # i is the row of C (which is br*2+i).
                idx = (bc*2+j)*4 + (br*2+i)
                res.append((idx, coeff))
        return res

    # Construct the 7 Strassen combinations for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    a1 = get_a_comb(0,0,1) + get_a_comb(1,1,1)
    b1 = get_b_comb(0,0,1) + get_b_comb(1,1,1)
    c1 = get_c_comb(0,0,1) + get_c_comb(1,1,1)
    # Wait, the c_comb in add_strassen_2x2 represents the *output* of the 2x2 multiplication.
    # For M1, the output is added to C00 and C11.
    # So we need to pass the C combinations that correspond to C00 and C11.
    # But add_strassen_2x2 expects a_comb, b_comb, c_comb for a *single* 2x2 product.
    # The c_comb passed should be the *local* 2x2 output.
    # Then the *global* placement is handled by the caller?
    # No, my add_strassen_2x2_clean takes a_comb, b_comb, c_comb which are *global* indices.
    # But for M1, the "c_comb" is not a single block, but a sum of blocks C00 and C11?
    # No. M1 is a 2x2 matrix.
    # M1 is added to C00 (with +1) and C11 (with +1).
    # So the "c_comb" for the 2x2 product M1 should be:
    # c[0,0] -> global C00[0,0], C00[0,1], C00[1,0], C00[1,1]
    # c[1,1] -> global C11[0,0], C11[0,1], C11[1,0], C11[1,1]
    # c[0,1] -> empty
    # c[1,0] -> empty
    
    # So we need to construct the c_comb list for each M_k appropriately.
    
    def make_c_comb_for_m1():
        # c00 -> C00, c11 -> C11
        return [
            [get_c_comb(0,0,1), []],
            [[], get_c_comb(1,1,1)]
        ]
    # This structure is 2x2 list of lists of (idx, coeff).
    # c_comb[0][0] is list for C00.
    # c_comb[1][1] is list for C11.
    
    # Let's re-verify the structure expected by add_strassen_2x2_clean.
    # It accesses a_comb[0][0], etc.
    # So a_comb is a 2x2 grid of lists.
    
    # M1
    a_comb_1 = [
        [get_a_comb(0,0,1), []],
        [[], get_a_comb(1,1,1)]
    ]
    b_comb_1 = [
        [get_b_comb(0,0,1), []],
        [[], get_b_comb(1,1,1)]
    ]
    c_comb_1 = [
        [get_c_comb(0,0,1), []],
        [[], get_c_comb(1,1,1)]
    ]
    t = add_strassen_2x2_clean(a_comb_1, b_comb_1, c_comb_1, t)

    # M2 = (A10+A11)B00
    # Output: C10 (+1), C11 (-1)
    a_comb_2 = [
        [[], []],
        [get_a_comb(1,0,1), get_a_comb(1,1,1)]
    ]
    b_comb_2 = [
        [get_b_comb(0,0,1), []],
        [[], []]
    ]
    c_comb_2 = [
        [[], []],
        [get_c_comb(1,0,1), get_c_comb(1,1,-1)]
    ]
    t = add_strassen_2x2_clean(a_comb_2, b_comb_2, c_comb_2, t)

    # M3 = A00(B01-B11)
    # Output: C01 (+1), C11 (+1)
    a_comb_3 = [
        [get_a_comb(0,0,1), []],
        [[], []]
    ]
    b_comb_3 = [
        [[], get_b_comb(0,1,1)],
        [[], get_b_comb(1,1,-1)]
    ]
    c_comb_3 = [
        [[], get_c_comb(0,1,1)],
        [[], get_c_comb(1,1,1)]
    ]
    t = add_strassen_2x2_clean(a_comb_3, b_comb_3, c_comb_3, t)

    # M4 = A11(B10-B00)
    # Output: C00 (+1), C10 (+1)
    a_comb_4 = [
        [[], []],
        [[], get_a_comb(1,1,1)]
    ]
    b_comb_4 = [
        [get_b_comb(0,0,-1), []],
        [[], get_b_comb(1,0,1)]
    ]
    c_comb_4 = [
        [get_c_comb(0,0,1), []],
        [get_c_comb(1,0,1), []]
    ]
    t = add_strassen_2x2_clean(a_comb_4, b_comb_4, c_comb_4, t)

    # M5 = (A00+A01)B11
    # Output: C00 (-1), C01 (+1)
    a_comb_5 = [
        [get_a_comb(0,0,1), get_a_comb(0,1,1)],
        [[], []]
    ]
    b_comb_5 = [
        [[], []],
        [[], get_b_comb(1,1,1)]
    ]
    c_comb_5 = [
        [get_c_comb(0,0,-1), get_c_comb(0,1,1)],
        [[], []]
    ]
    t = add_strassen_2x2_clean(a_comb_5, b_comb_5, c_comb_5, t)

    # M6 = (A10-A00)(B00+B01)
    # Output: C11 (+1)
    a_comb_6 = [
        [get_a_comb(0,0,-1), []],
        [get_a_comb(1,0,1), []]
    ]
    b_comb_6 = [
        [get_b_comb(0,0,1), get_b_comb(0,1,1)],
        [[], []]
    ]
    c_comb_6 = [
        [[], []],
        [[], get_c_comb(1,1,1)]
    ]
    t = add_strassen_2x2_clean(a_comb_6, b_comb_6, c_comb_6, t)

    # M7 = (A01-A11)(B10+B11)
    # Output: C00 (+1)
    a_comb_7 = [
        [[], get_a_comb(0,1,1)],
        [[], get_a_comb(1,1,-1)]
    ]
    b_comb_7 = [
        [[], []],
        [get_b_comb(1,0,1), get_b_comb(1,1,1)]
    ]
    c_comb_7 = [
        [get_c_comb(0,0,1), []],
        [[], []]
    ]
    t = add_strassen_2x2_clean(a_comb_7, b_comb_7, c_comb_7, t)

    # Now add the 4x4x1 part (column 4 of B)
    # C[i,4] = sum_j A[i,j] B[j,4]
    for i in range(4):
        for j in range(4):
            U[i*4 + j, t] = 1.0
            V[j*5 + 4, t] = 1.0
            W[4*4 + i, t] = 1.0
            t += 1

    # t should be 65
    assert t == 65, f"Expected 65 terms, got {t}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
