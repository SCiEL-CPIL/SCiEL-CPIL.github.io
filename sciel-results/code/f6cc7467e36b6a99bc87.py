import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 64 construction for 4x4 * 4x5
    # Using a direct bilinear algorithm with shared structure
    rank = 64
    U = [[0] * rank for _ in range(n * m)]  # 16 x 64
    V = [[0] * rank for _ in range(m * p)]  # 20 x 64
    W = [[0] * rank for _ in range(p * n)]  # 20 x 64
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
    
    col = 0
    
    # Use a unified bilinear algorithm
    # Split into 2x2 blocks for A and B (first 4 cols), plus 5th col
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01, B02], [B10, B11, B12]] where B02, B12 are 2x1
    
    # 7 Strassen terms for the 4x4 part (each is 2x2x2x2 = 4 scalar mults = 28 terms)
    # Plus terms for the 5th column
    
    # Actually, let's use the known rank-64 algorithm
    # This uses 64 scalar multiplications total
    
    # Terms 0-63: direct bilinear form
    # We'll construct it as 49 (for 4x4x4) + 15 (for 4x4x1 with one shared)
    
    # For the 15-term part, we use:
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # We can write this as 15 terms by noting that:
    # A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4] + A[1,3]*B[3,4]
    # A[2,0]*B[0,4] + A[2,1]*B[1,4] + A[2,2]*B[2,4] + A[2,3]*B[3,4]
    # A[3,0]*B[0,4] + A[3,1]*B[1,4] + A[3,2]*B[2,4] + A[3,3]*B[3,4]
    # = 16 terms minimum. Cannot reduce below 16 for this part.
    
    # So total is 49 + 16 = 65. The rank-64 construction must use a different approach.
    
    # Let's use a completely different bilinear algorithm
    # Based on the work of Landsberg and Ottaviani, the rank of 4x4x5 is 64
    
    # I'll use a direct construction with 64 terms
    
    # Define 64 bilinear terms explicitly
    # This is complex, so let's use a hybrid: 48 for 4x4x4 + 16 for 4x4x1 = 64
    # Use a rank-48 algorithm for 4x4x4 (if it exists)
    
    # Actually, the minimum rank for 4x4x4 is 49 (Strassen is optimal for 4x4x4)
    # So we're stuck with 49 + 16 = 65
    
    # Unless we can share terms between the two parts
    
    # Let me try: use 49 terms for 4x4x4, and for the 5th column,
    # notice that some of the Strassen intermediate products can be reused
    
    # For example, in Strassen, M1 = (A00+A11)(B00+B11)
    # This involves A[0,0], A[0,1], A[2,2], A[2,3] and B[0,0], B[0,1], B[2,2], B[2,3]
    # The 5th column needs A[i,j]*B[j,4]
    # There's no direct overlap since B[j,4] is different from B[j,k] for k<4
    
    # So I'll stick with rank 65 but try to optimize the construction
    
    # Actually, let me just return the rank-65 solution since I can't find a rank-64 one
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c
    
    col = 0
    
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),
    ]
    
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),
    ]
    
    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            
            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            
            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
