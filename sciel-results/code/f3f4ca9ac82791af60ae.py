import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def zero_lf():
        return [0.0] * 16

    def add_lf(a, b, sign=1):
        return [x + sign * y for x, y in zip(a, b)]

    def sub_lf(a, b):
        return [x - y for x, y in zip(a, b)]

    def get_a_lf(br, bc, lr, lc):
        lf = zero_lf()
        # Index mapping: Row i, Col j -> i * 4 + j
        # Block (br, bc) local (lr, lc) -> Global (2*br+lr, 2*bc+lc)
        idx = (2 * br + lr) * 4 + (2 * bc + lc)
        lf[idx] = 1.0
        return lf

    def get_b_lf(br, bc, lr, lc):
        lf = zero_lf()
        idx = (2 * br + lr) * 4 + (2 * bc + lc)
        lf[idx] = 1.0
        return lf

    # Pre-compute A and B block entries as 2x2 matrices of 16-dim vectors
    A_blocks = [[None for _ in range(2)] for _ in range(2)]
    B_blocks = [[None for _ in range(2)] for _ in range(2)]
    for br in range(2):
        for bc in range(2):
            A_blocks[br][bc] = [[zero_lf() for _ in range(2)] for _ in range(2)]
            B_blocks[br][bc] = [[zero_lf() for _ in range(2)] for _ in range(2)]
            for lr in range(2):
                for lc in range(2):
                    A_blocks[br][bc][lr][lc] = get_a_lf(br, bc, lr, lc)
                    B_blocks[br][bc][lr][lc] = get_b_lf(br, bc, lr, lc)

    def add_block_lf(X, Y, sign=1):
        return [
            [add_lf(X[i][j], Y[i][j], sign) for j in range(2)]
            for i in range(2)
        ]

    def sub_block_lf(X, Y):
        return [
            [sub_lf(X[i][j], Y[i][j]) for j in range(2)]
            for i in range(2)
        ]

    # Define Outer Strassen Linear Forms (2x2 matrices of 16-dim vectors)
    # L_k and R_k
    outer_L = []
    outer_R = []
    
    # k=0: M1 = (A00+A11)(B00+B11)
    L0 = add_block_lf(A_blocks[0][0], A_blocks[1][1])
    R0 = add_block_lf(B_blocks[0][0], B_blocks[1][1])
    outer_L.append(L0)
    outer_R.append(R0)
    
    # k=1: M2 = (A10+A11)B00
    L1 = add_block_lf(A_blocks[1][0], A_blocks[1][1])
    R1 = B_blocks[0][0]
    outer_L.append(L1)
    outer_R.append(R1)
    
    # k=2: M3 = A00(B01-B11)
    L2 = A_blocks[0][0]
    R2 = sub_block_lf(B_blocks[0][1], B_blocks[1][1])
    outer_L.append(L2)
    outer_R.append(R2)
    
    # k=3: M4 = A11(B10-B00)
    L3 = A_blocks[1][1]
    R3 = sub_block_lf(B_blocks[1][0], B_blocks[0][0])
    outer_L.append(L3)
    outer_R.append(R3)
    
    # k=4: M5 = (A00+A01)B11
    L4 = add_block_lf(A_blocks[0][0], A_blocks[0][1])
    R4 = B_blocks[1][1]
    outer_L.append(L4)
    outer_R.append(R4)
    
    # k=5: M6 = (A10-A00)(B00+B01)
    L5 = sub_block_lf(A_blocks[1][0], A_blocks[0][0])
    R5 = add_block_lf(B_blocks[0][0], B_blocks[0][1])
    outer_L.append(L5)
    outer_R.append(R5)
    
    # k=6: M7 = (A01-A11)(B10+B11)
    L6 = sub_block_lf(A_blocks[0][1], A_blocks[1][1])
    R6 = add_block_lf(B_blocks[1][0], B_blocks[1][1])
    outer_L.append(L6)
    outer_R.append(R6)

    # Inner Strassen Decomposition Parameters
    # For a 2x2 multiplication X*Y, the 7 terms are defined by linear forms on X and Y
    # and coefficients for the result Z.
    # We define a function to generate the 7 (U_lf, V_lf, Z_coeffs) for given X_block, Y_block
    def get_inner_terms(X_block, Y_block):
        terms = []
        
        # Term 0: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
        u = add_lf(X_block[0][0], X_block[1][1])
        v = add_lf(Y_block[0][0], Y_block[1][1])
        z_coeffs = [[1, 0], [0, 1]]
        terms.append((u, v, z_coeffs))
        
        # Term 1: (X10+X11)Y00 -> Z10 += 1, Z11 -= 1
        u = add_lf(X_block[1][0], X_block[1][1])
        v = Y_block[0][0]
        z_coeffs = [[0, 0], [1, -1]]
        terms.append((u, v, z_coeffs))
        
        # Term 2: X00(Y01-Y11) -> Z01 += 1, Z11 += 1
        u = X_block[0][0]
        v = sub_lf(Y_block[0][1], Y_block[1][1])
        z_coeffs = [[0, 1], [0, 1]]
        terms.append((u, v, z_coeffs))
        
        # Term 3: X11(Y10-Y00) -> Z00 += 1, Z10 += 1
        u = X_block[1][1]
        v = sub_lf(Y_block[1][0], Y_block[0][0])
        z_coeffs = [[1, 0], [1, 0]]
        terms.append((u, v, z_coeffs))
        
        # Term 4: (X00+X01)Y11 -> Z00 -= 1, Z01 += 1
        u = add_lf(X_block[0][0], X_block[0][1])
        v = Y_block[1][1]
        z_coeffs = [[-1, 1], [0, 0]]
        terms.append((u, v, z_coeffs))
        
        # Term 5: (X10-X00)(Y00+Y01) -> Z11 += 1
        u = sub_lf(X_block[1][0], X_block[0][0])
        v = add_lf(Y_block[0][0], Y_block[0][1])
        z_coeffs = [[0, 0], [0, 1]]
        terms.append((u, v, z_coeffs))
        
        # Term 6: (X01-X11)(Y10+Y11) -> Z00 += 1
        u = sub_lf(X_block[0][1], X_block[1][1])
        v = add_lf(Y_block[1][0], Y_block[1][1])
        z_coeffs = [[1, 0], [0, 0]]
        terms.append((u, v, z_coeffs))
        
        return terms

    col = 0
    for k in range(7):
        L_k = outer_L[k]
        R_k = outer_R[k]
        
        inner_terms = get_inner_terms(L_k, R_k)
        
        for l in range(7):
            u_lf, v_lf, z_coeffs = inner_terms[l]
            
            # u_lf and v_lf are already 16-dim vectors (linear forms of A and B)
            # z_coeffs is a 2x2 matrix of coefficients for the block result
            # We need to expand z_coeffs to a 16-dim vector for W
            
            w_lf = zero_lf()
            for br in range(2):
                for bc in range(2):
                    coeff = z_coeffs[br][bc]
                    if coeff != 0:
                        for lr in range(2):
                            for lc in range(2):
                                idx = (2 * br + lr) * 4 + (2 * bc + lc)
                                w_lf[idx] += coeff
            
            # Add to U, V, W
            for i in range(16):
                if u_lf[i] != 0:
                    U[i][col] += u_lf[i]
                if v_lf[i] != 0:
                    V[i][col] += v_lf[i]
                if w_lf[i] != 0:
                    W[i][col] += w_lf[i]
            
            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
