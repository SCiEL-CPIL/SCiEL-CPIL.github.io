import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # Helper to create sparse vectors
    def make_vec(size, coeffs):
        # coeffs is dict index: value
        vec = [0.0] * size
        for idx, val in coeffs.items():
            vec[idx] = val
        return vec

    # 1. 4x4x4 part (k=0..3) using Strassen
    # A is 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3)
    
    # Define blocks of A
    # A11: 0,1 x 0,1
    # A12: 0,1 x 2,3
    # A21: 2,3 x 0,1
    # A22: 2,3 x 2,3
    
    # Strassen M1..M7 definitions for 2x2 block mult
    # X_l, Y_l are 2x2 matrices of linear forms in A, B blocks
    
    # We need to express x_ij and y_ij as linear combos of a_ij, b_jk
    
    # Let's define the linear forms for X and Y for each M_l
    
    # M1: X = A11 + A22, Y = B11 + B22
    # M2: X = A21 + A22, Y = B11
    # M3: X = A11, Y = B12 - B22
    # M4: X = A22, Y = B21 - B11
    # M5: X = A11 + A12, Y = B22
    # M6: X = A21 - A11, Y = B11 + B12
    # M7: X = A12 - A22, Y = B21 + B22
    
    # For each M_l, we have 7 scalar products s1..s7
    # s1 = (x11+x22)(y11+y22)
    # s2 = (x21+x22)y11
    # s3 = x11(y12-y22)
    # s4 = x22(y21-y11)
    # s5 = (x11+x12)y22
    # s6 = (x21-x11)(y11+y12)
    # s7 = (x12-x22)(y21+y22)
    
    # Result Z = X Y
    # z11 = s1 + s4 - s5 + s7
    # z12 = s3 + s5
    # z21 = s2 + s4
    # z22 = s1 - s2 + s3 + s6
    
    # Contribution of M_l to C:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the block indices to global indices.
    # Block C11 corresponds to C[0:2, 0:2]
    # Block C12 corresponds to C[0:2, 2:4]
    # Block C21 corresponds to C[2:4, 0:2]
    # Block C22 corresponds to C[2:4, 2:4]
    
    # Let's create a function to get the linear form for a block of A or B
    # A block (bi, bj) where bi, bj in {0,1}
    # bi=0, bj=0 -> A11 -> rows 0,1 cols 0,1
    # bi=0, bj=1 -> A12 -> rows 0,1 cols 2,3
    # bi=1, bj=0 -> A21 -> rows 2,3 cols 0,1
    # bi=1, bj=1 -> A22 -> rows 2,3 cols 2,3
    
    def get_A_block_coeffs(bi, bj, local_i, local_j):
        # local_i, local_j in {0,1}
        # global row = bi*2 + local_i
        # global col = bj*2 + local_j
        gi = bi * 2 + local_i
        gj = bj * 2 + local_j
        return {gi * 4 + gj: 1.0}

    def get_B_block_coeffs(bi, bj, local_i, local_j, k_offset=0):
        # B block (bi, bj) where bi is row block, bj is col block
        # Rows are global j (0..3)
        # Cols are global k (0..3)
        # local_i is row within block, local_j is col within block
        gi = bi * 2 + local_i
        gj = bj * 2 + local_j + k_offset
        return {gi * 9 + gj: 1.0}

    # Define M_l structures
    # Each M_l is defined by X_l and Y_l
    # X_l is a 2x2 matrix of linear forms
    # We represent a linear form as a list of (block_index, local_i, local_j, coeff)
    
    # Actually, easier to just compute the coefficients for U and V directly.
    
    # Let's iterate l=0..6
    for l in range(7):
        # Determine X_l and Y_l block definitions
        # X_l blocks: (bi, bj, sign)
        # Y_l blocks: (bi, bj, sign)
        
        if l == 0: # M1
            X_blocks = [(0,0,1), (1,1,1)]
            Y_blocks = [(0,0,1), (1,1,1)]
        elif l == 1: # M2
            X_blocks = [(1,0,1), (1,1,1)]
            Y_blocks = [(0,0,1)]
        elif l == 2: # M3
            X_blocks = [(0,0,1)]
            Y_blocks = [(0,1,1), (1,1,-1)]
        elif l == 3: # M4
            X_blocks = [(1,1,1)]
            Y_blocks = [(1,0,1), (0,0,-1)]
        elif l == 4: # M5
            X_blocks = [(0,0,1), (0,1,1)]
            Y_blocks = [(1,1,1)]
        elif l == 5: # M6
            X_blocks = [(1,0,1), (0,0,-1)]
            Y_blocks = [(0,0,1), (0,1,1)]
        elif l == 6: # M7
            X_blocks = [(0,1,1), (1,1,-1)]
            Y_blocks = [(1,0,1), (1,1,1)]
            
        # Now, for each of the 7 scalar products s_k
        for k in range(7):
            # Determine linear forms for the two factors of s_k
            # s_k = Lx * Ly
            # Lx is a linear combo of x_ij
            # Ly is a linear combo of y_ij
            
            # x_ij is sum of A blocks
            # y_ij is sum of B blocks
            
            # We need to identify which x_ij and y_ij are involved in s_k
            
            # s1: (x11+x22)(y11+y22)
            # s2: (x21+x22)y11
            # s3: x11(y12-y22)
            # s4: x22(y21-y11)
            # s5: (x11+x12)y22
            # s6: (x21-x11)(y11+y12)
            # s7: (x12-x22)(y21+y22)
            
            # Define x factors: list of (local_i, local_j, coeff)
            # Define y factors: list of (local_i, local_j, coeff)
            
            if k == 0: # s1
                x_factors = [(0,0,1), (1,1,1)]
                y_factors = [(0,0,1), (1,1,1)]
            elif k == 1: # s2
                x_factors = [(1,0,1), (1,1,1)]
                y_factors = [(0,0,1)]
            elif k == 2: # s3
                x_factors = [(0,0,1)]
                y_factors = [(0,1,1), (1,1,-1)]
            elif k == 3: # s4
                x_factors = [(1,1,1)]
                y_factors = [(1,0,1), (0,0,-1)]
            elif k == 4: # s5
                x_factors = [(0,0,1), (0,1,1)]
                y_factors = [(1,1,1)]
            elif k == 5: # s6
                x_factors = [(1,0,1), (0,0,-1)]
                y_factors = [(0,0,1), (0,1,1)]
            elif k == 6: # s7
                x_factors = [(0,1,1), (1,1,-1)]
                y_factors = [(1,0,1), (1,1,1)]
            
            # Construct U vector (coeffs for A)
            u_coeffs = {}
            for li, lj, cx in x_factors:
                # x_li_lj is sum of A blocks in X_blocks
                for bi, bj, sign in X_blocks:
                    # x_li_lj += sign * A_block(bi, bj)
                    # A_block(bi, bj) has elements at (bi*2+li, bj*2+lj)
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    idx = gi * 4 + gj
                    val = sign * cx
                    u_coeffs[idx] = u_coeffs.get(idx, 0.0) + val
            
            # Construct V vector (coeffs for B)
            v_coeffs = {}
            for li, lj, cy in y_factors:
                for bi, bj, sign in Y_blocks:
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    idx = gi * 9 + gj
                    val = sign * cy
                    v_coeffs[idx] = v_coeffs.get(idx, 0.0) + val
            
            # Construct W vector (coeffs for C)
            # s_k contributes to z_11, z_12, z_21, z_22 of M_l
            # We need to know the coefficient of s_k in z_ij
            # z11 = s1 + s4 - s5 + s7
            # z12 = s3 + s5
            # z21 = s2 + s4
            # z22 = s1 - s2 + s3 + s6
            
            w_coeffs = {}
            
            # Determine coeff of s_k in z11, z12, z21, z22
            c_z11 = 0
            c_z12 = 0
            c_z21 = 0
            c_z22 = 0
            
            if k == 0: # s1
                c_z11 = 1
                c_z22 = 1
            elif k == 1: # s2
                c_z21 = 1
                c_z22 = -1
            elif k == 2: # s3
                c_z12 = 1
                c_z22 = 1
            elif k == 3: # s4
                c_z11 = 1
                c_z21 = 1
            elif k == 4: # s5
                c_z11 = -1
                c_z12 = 1
            elif k == 5: # s6
                c_z22 = 1
            elif k == 6: # s7
                c_z11 = 1
            
            # Now map z_ij to C blocks
            # M_l contributes to C blocks based on l
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # Determine which C block M_l contributes to and with what sign
            # c_block_sign[0] for C11, etc.
            
            c_block_signs = [0, 0, 0, 0] # C11, C12, C21, C22
            
            if l == 0: # M1
                c_block_signs = [1, 0, 0, 1]
            elif l == 1: # M2
                c_block_signs = [0, 0, 1, -1]
            elif l == 2: # M3
                c_block_signs = [0, 1, 0, 1]
            elif l == 3: # M4
                c_block_signs = [1, 0, 1, 0]
            elif l == 4: # M5
                c_block_signs = [-1, 1, 0, 0]
            elif l == 5: # M6
                c_block_signs = [0, 0, 0, 1]
            elif l == 6: # M7
                c_block_signs = [1, 0, 0, 0]
            
            # z11 is top-left of M_l. It goes to top-left of C block.
            # z12 is top-right of M_l. It goes to top-right of C block.
            # z21 is bottom-left of M_l. It goes to bottom-left of C block.
            # z22 is bottom-right of M_l. It goes to bottom-right of C block.
            
            # C block 0 (C11): rows 0,1 cols 0,1
            # C block 1 (C12): rows 0,1 cols 2,3
            # C block 2 (C21): rows 2,3 cols 0,1
            # C block 3 (C22): rows 2,3 cols 2,3
            
            # For each z component, add to W
            # z11 -> C_block[0,0]
            if c_z11 != 0:
                sign = c_block_signs[0]
                if sign != 0:
                    # C11 top-left
                    idx = 0 * 4 + 0
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z11 * sign
            # z12 -> C_block[0,1]
            if c_z12 != 0:
                sign = c_block_signs[1]
                if sign != 0:
                    # C12 top-left? No, z12 is top-right of M_l.
                    # M_l is 2x2. z12 is row 0, col 1.
                    # C12 block is rows 0,1 cols 2,3.
                    # The element z12 corresponds to C[0, 2+1] = C[0,3]?
                    # Wait. M_l is the product of 2x2 blocks.
                    # The result M_l is a 2x2 matrix.
                    # M_l[0,0] goes to C11[0,0]
                    # M_l[0,1] goes to C11[0,1]
                    # M_l[1,0] goes to C11[1,0]
                    # M_l[1,1] goes to C11[1,1]
                    
                    # If M_l contributes to C12, then:
                    # M_l[0,0] goes to C12[0,0] -> C[0,2]
                    # M_l[0,1] goes to C12[0,1] -> C[0,3]
                    
                    # So z12 (which is M_l[0,1]) goes to C[0,3] if block is C12.
                    # General rule:
                    # C_block row offset = br * 2
                    # C_block col offset = bc * 2
                    # M_l[i,j] goes to C[br*2+i, bc*2+j]
                    
                    # We need to know br, bc for the C block.
                    # C11: br=0, bc=0
                    # C12: br=0, bc=1
                    # C21: br=1, bc=0
                    # C22: br=1, bc=1
                    
                    # We have c_block_signs[1] for C12.
                    # So br=0, bc=1.
                    # z12 is i=0, j=1.
                    # idx = (0*2+0)*4 + (1*2+1) = 0*4 + 3 = 3.
                    idx = (0 * 2 + 0) * 4 + (1 * 2 + 1)
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z12 * sign
            # z21 -> C_block[1,0]
            if c_z21 != 0:
                sign = c_block_signs[2]
                if sign != 0:
                    # C21: br=1, bc=0
                    # z21 is i=1, j=0
                    idx = (1 * 2 + 1) * 4 + (0 * 2 + 0)
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z21 * sign
            # z22 -> C_block[1,1]
            if c_z22 != 0:
                sign = c_block_signs[3]
                if sign != 0:
                    # C22: br=1, bc=1
                    # z22 is i=1, j=1
                    idx = (1 * 2 + 1) * 4 + (1 * 2 + 1)
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z22 * sign
            
            # Add term
            add_term(make_vec(16, u_coeffs), make_vec(36, v_coeffs), make_vec(36, w_coeffs))
            
    # 2. 4x4x5 part (k=4..8)
    # k=4,5: 4x4x2 (Rank 28)
    # k=6: 4x4x1 (Rank 16)
    # k=7,8: 4x4x2 (Rank 28)
    
    # Implement 4x4x2 using Block Strassen
    # C11 = A11 B11 + A12 B21
    # C21 = A21 B11 + A22 B21
    # B11 is 2x2 (rows 0,1, cols k0, k0+1)
    # B21 is 2x2 (rows 2,3, cols k0, k0+1)
    
    def add_4x4x2(k_start):
        # k_start is 4 or 7
        # B cols are k_start, k_start+1
        # C cols are k_start, k_start+1
        
        # 4 products of 2x2 matrices
        # P1 = A11 B11
        # P2 = A12 B21
        # P3 = A21 B11
        # P4 = A22 B21
        
        # Each product has 7 terms.
        # We can reuse the Strassen logic for 2x2
        
        products = [
            ((0,0), (0,0)), # A11, B11 -> C11
            ((0,1), (1,0)), # A12, B21 -> C11
            ((1,0), (0,0)), # A21, B11 -> C21
            ((1,1), (1,0)), # A22, B21 -> C21
        ]
        
        for a_block, b_block in products:
            # a_block is (bi, bj) for A
            # b_block is (bi, bj) for B (relative to k_start)
            # Note: B block col index is 0 for both k_start and k_start+1?
            # No, B11 is cols 0,1 of the 4x2 block.
            # B21 is cols 0,1 of the 4x2 block.
            # So bj for B is always 0?
            # Wait, B is 4x2.
            # B11 is rows 0,1, cols 0,1.
            # B21 is rows 2,3, cols 0,1.
            # So bj=0 for both.
            
            # We need to run Strassen for A_block * B_block
            # A_block is 2x2. B_block is 2x2.
            # We can use the same 7-term expansion.
            
            # Define X = A_block, Y = B_block
            # X_blocks = [(a_block[0], a_block[1], 1)]
            # Y_blocks = [(b_block[0], b_block[1], 1)]
            
            X_blocks = [(a_block[0], a_block[1], 1)]
            Y_blocks = [(b_block[0], b_block[1], 1)]
            
            for k in range(7):
                # Same s_k definitions
                if k == 0:
                    x_factors = [(0,0,1), (1,1,1)]
                    y_factors = [(0,0,1), (1,1,1)]
                elif k == 1:
                    x_factors = [(1,0,1), (1,1,1)]
                    y_factors = [(0,0,1)]
                elif k == 2:
                    x_factors = [(0,0,1)]
                    y_factors = [(0,1,1), (1,1,-1)]
                elif k == 3:
                    x_factors = [(1,1,1)]
                    y_factors = [(1,0,1), (0,0,-1)]
                elif k == 4:
                    x_factors = [(0,0,1), (0,1,1)]
                    y_factors = [(1,1,1)]
                elif k == 5:
                    x_factors = [(1,0,1), (0,0,-1)]
                    y_factors = [(0,0,1), (0,1,1)]
                elif k == 6:
                    x_factors = [(0,1,1), (1,1,-1)]
                    y_factors = [(1,0,1), (1,1,1)]
                
                u_coeffs = {}
                for li, lj, cx in x_factors:
                    for bi, bj, sign in X_blocks:
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        idx = gi * 4 + gj
                        val = sign * cx
                        u_coeffs[idx] = u_coeffs.get(idx, 0.0) + val
                
                v_coeffs = {}
                for li, lj, cy in y_factors:
                    for bi, bj, sign in Y_blocks:
                        gi = bi * 2 + li
                        gj = bj * 2 + lj + k_start
                        idx = gi * 9 + gj
                        val = sign * cy
                        v_coeffs[idx] = v_coeffs.get(idx, 0.0) + val
                
                w_coeffs = {}
                # z components
                c_z11 = 0; c_z12 = 0; c_z21 = 0; c_z22 = 0
                if k == 0: c_z11=1; c_z22=1
                elif k == 1: c_z21=1; c_z22=-1
                elif k == 2: c_z12=1; c_z22=1
                elif k == 3: c_z11=1; c_z21=1
                elif k == 4: c_z11=-1; c_z12=1
                elif k == 5: c_z22=1
                elif k == 6: c_z11=1
                
                # Determine which C block this product contributes to
                # P1, P2 -> C11 (br=0, bc=0 relative to k_start? No, C cols are k_start, k_start+1)
                # C11 is rows 0,1, cols k_start, k_start+1
                # C21 is rows 2,3, cols k_start, k_start+1
                
                # If a_block is (0,0) or (0,1), it contributes to C11 (rows 0,1)
                # If a_block is (1,0) or (1,1), it contributes to C21 (rows 2,3)
                
                br = a_block[0]
                # bc is 0 because both cols are in the 2-wide block
                bc = 0
                
                if c_z11 != 0:
                    idx = (br * 2 + 0) * 4 + (bc * 2 + 0 + k_start)
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z11
                if c_z12 != 0:
                    idx = (br * 2 + 0) * 4 + (bc * 2 + 1 + k_start)
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z12
                if c_z21 != 0:
                    idx = (br * 2 + 1) * 4 + (bc * 2 + 0 + k_start)
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z21
                if c_z22 != 0:
                    idx = (br * 2 + 1) * 4 + (bc * 2 + 1 + k_start)
                    w_coeffs[idx] = w_coeffs.get(idx, 0.0) + c_z22
                
                add_term(make_vec(16, u_coeffs), make_vec(36, v_coeffs), make_vec(36, w_coeffs))
                
    add_4x4x2(4)
    
    # k=6: 4x4x1 Schoolbook
    for i in range(4):
        for j in range(4):
            u = make_vec(16, {i*4+j: 1.0})
            v = make_vec(36, {j*9+6: 1.0})
            w = make_vec(36, {6*4+i: 1.0})
            add_term(u, v, w)
            
    add_4x4x2(7)
    
    # Convert to numpy arrays or lists
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
