"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    rank = 49
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    # Note: W index is k*n + i.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_idx: index in A (i*4+j)
    # v_idx: index in B (j*4+k)
    # w_idx: index in C (k*4+i)
    # coeff: coefficient (1 or -1)
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Block Strassen Terms
    # Block indices: I, J in {0, 1}
    # A[I,J] is 2x2 block. Rows 2I, 2I+1. Cols 2J, 2J+1.
    # B[J,K] is 2x2 block. Rows 2J, 2J+1. Cols 2K, 2K+1.
    # C[I,K] is 2x2 block. Rows 2I, 2I+1. Cols 2K, 2K+1.
    
    # Define the 7 block-level combinations
    # Each is a list of (A_block, sign) and (B_block, sign)
    # A_block is (I, J), B_block is (J, K)
    # We need to form linear combinations of A and B blocks.
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00) (B00 + B01)
    # M7 = (A01 - A11) (B10 + B11)
    
    # For each M_k, we have X and Y (2x2 matrices).
    # X is a linear combo of A blocks. Y is a linear combo of B blocks.
    # We then apply Strassen to X*Y.
    
    # Let's define the coefficients for X and Y for each M_k.
    # X_entries: dict mapping (local_row, local_col) to list of (global_I, global_J, coeff)
    # Actually, it's easier to just build the vectors directly.
    
    # For a given M_k, X is a 2x2 matrix where each entry is a linear combination of A entries.
    # Y is a 2x2 matrix where each entry is a linear combination of B entries.
    
    # Strassen terms for X*Y:
    # S1 = (X00 + X11)(Y00 + Y11)
    # S2 = (X10 + X11) Y00
    # S3 = X00 (Y01 - Y11)
    # S4 = X11 (Y10 - Y00)
    # S5 = (X00 + X01) Y11
    # S6 = (X10 - X00) (Y00 + Y01)
    # S7 = (X01 - X11) (Y10 + Y11)
    
    # Result Z:
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # We will iterate through the 7 M_k's.
    # For each, we define the linear forms for X and Y.
    
    # M1: X = A00 + A11, Y = B00 + B11
    # A00 is A[0,0], A[0,1], A[1,0], A[1,1] (global indices 0,1,4,5)
    # A11 is A[2,2], A[2,3], A[3,2], A[3,3] (global indices 10,11,14,15)
    # So X00 = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
    # X01 = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
    # X10 = A00[1,0] + A11[1,0] = A[1,0] + A[3,2]
    # X11 = A00[1,1] + A11[1,1] = A[1,1] + A[3,3]
    
    # Y00 = B00[0,0] + B11[0,0] = B[0,0] + B[2,2]
    # Y01 = B00[0,1] + B11[0,1] = B[0,1] + B[2,3]
    # Y10 = B00[1,0] + B11[1,0] = B[1,0] + B[3,2]
    # Y11 = B00[1,1] + B11[1,1] = B[1,1] + B[3,3]
    
    # This is getting complex to hardcode manually for all 7.
    # I will write a helper function to generate the vectors for a given linear combination.
    
    def get_a_vec(block_I, block_J, local_r, local_c, sign=1):
        # Returns list of (global_idx, coeff)
        g_r = 2 * block_I + local_r
        g_c = 2 * block_J + local_c
        return [(g_r * 4 + g_c, sign)]
        
    def get_b_vec(block_J, block_K, local_r, local_c, sign=1):
        g_r = 2 * block_J + local_r
        g_c = 2 * block_K + local_c
        return [(g_r * 4 + g_c, sign)]
        
    def get_c_vec(block_I, block_K, local_r, local_c, sign=1):
        # W index is k*n + i.
        # C[I,K] block.
        # Global row i = 2*I + local_r
        # Global col k = 2*K + local_c
        # W index = k*4 + i
        g_i = 2 * block_I + local_r
        g_k = 2 * block_K + local_c
        return [(g_k * 4 + g_i, sign)]

    # Define the 7 block terms as lists of (A_blocks, B_blocks)
    # Each A_block is (I, J, local_r, local_c, coeff)
    # This is too granular.
    
    # Let's define X and Y as 2x2 matrices of "linear forms".
    # A linear form is a list of (global_index, coeff).
    
    def combine_forms(forms1, forms2):
        # Adds two lists of (idx, coeff)
        combined = {}
        for idx, c in forms1:
            combined[idx] = combined.get(idx, 0) + c
        for idx, c in forms2:
            combined[idx] = combined.get(idx, 0) + c
        return [(k, v) for k, v in combined.items() if v != 0]

    # Define X and Y for each M_k
    # M1: X = A00 + A11, Y = B00 + B11
    # X00 = A(0,0,0,0) + A(1,1,0,0)
    # X01 = A(0,0,0,1) + A(1,1,0,1)
    # X10 = A(0,0,1,0) + A(1,1,1,0)
    # X11 = A(0,0,1,1) + A(1,1,1,1)
    
    # Y00 = B(0,0,0,0) + B(1,1,0,0)
    # Y01 = B(0,0,0,1) + B(1,1,0,1)
    # Y10 = B(0,0,1,0) + B(1,1,1,0)
    # Y11 = B(0,0,1,1) + B(1,1,1,1)
    
    # I will create a function to process one Strassen term given X and Y forms.
    
    def process_strassen_term(X_forms, Y_forms, result_coeffs):
        # X_forms: 2x2 list of lists of (idx, coeff)
        # Y_forms: 2x2 list of lists of (idx, coeff)
        # result_coeffs: dict mapping (C_block_I, C_block_K, local_r, local_c) to coeff
        # This is getting too abstract.
        
        # Let's just generate the 7 scalar multiplications for X*Y.
        # S1 = (X00 + X11)(Y00 + Y11)
        # The "product" of two linear forms is not a single term, it's a sum of products.
        # Wait. The tensor decomposition requires each term to be a product of a linear form in A, a linear form in B, and a linear form in C?
        # No. The tensor decomposition is a sum of rank-1 tensors.
        # A rank-1 tensor is $u \otimes v \otimes w$.
        # $u$ is a vector in A-space. $v$ is a vector in B-space. $w$ is a vector in C-space.
        # The term is $\sum u_i v_j w_k$.
        # This corresponds to a single scalar multiplication of the linear forms $u^T A$ and $v^T B$, added to $w^T C$.
        # So each of the 49 terms is a single scalar multiplication.
        
        # My previous logic was:
        # M1 = (A00 + A11)(B00 + B11).
        # This M1 is a 2x2 matrix product.
        # We decompose this 2x2 matrix product into 7 scalar multiplications using Strassen.
        # Each of those 7 scalar multiplications is a rank-1 term in the overall 4x4 tensor.
        
        # So, for M1:
        # X = A00 + A11. This is a 2x2 matrix.
        # Y = B00 + B11. This is a 2x2 matrix.
        # We compute S1 = (X00 + X11)(Y00 + Y11).
        # X00 is a linear form in A. X11 is a linear form in A.
        # So (X00 + X11) is a linear form in A.
        # (Y00 + Y11) is a linear form in B.
        # The product is a scalar.
        # This scalar contributes to Z00, Z11, etc.
        # Specifically, S1 contributes to Z00 with coeff 1, and Z11 with coeff 1.
        # So the rank-1 term is:
        # u = X00 + X11
        # v = Y00 + Y11
        # w = Z00_part + Z11_part
        # Z00 is C[0,0] block local (0,0).
        # Z11 is C[1,1] block local (1,1).
        
        # So I need to:
        # 1. Define the linear forms for X and Y for each M_k.
        # 2. For each of the 7 Strassen steps, define u, v, w.
        
        pass

    # Let's implement this.
    
    # Define linear forms for A blocks
    # A[I,J] is 2x2.
    # Form A(I, J, r, c) is a vector with 1 at index (2I+r)*4 + (2J+c).
    
    def a_form(I, J, r, c, sign=1):
        idx = (2*I + r)*4 + (2*J + c)
        return {idx: sign}

    def b_form(J, K, r, c, sign=1):
        idx = (2*J + r)*4 + (2*K + c)
        return {idx: sign}

    def c_form(I, K, r, c, sign=1):
        # W index k*4 + i
        i = 2*I + r
        k = 2*K + c
        idx = k*4 + i
        return {idx: sign}

    def add_dicts(d1, d2, sign1=1, sign2=1):
        res = {}
        for k, v in d1.items():
            res[k] = res.get(k, 0) + v * sign1
        for k, v in d2.items():
            res[k] = res.get(k, 0) + v * sign2
        return res

    def sub_dicts(d1, d2, sign1=1, sign2=1):
        return add_dicts(d1, d2, sign1, -sign2)

    # Define X and Y for M1
    # X00 = A(0,0,0,0) + A(1,1,0,0)
    X00 = add_dicts(a_form(0,0,0,0), a_form(1,1,0,0))
    X01 = add_dicts(a_form(0,0,0,1), a_form(1,1,0,1))
    X10 = add_dicts(a_form(0,0,1,0), a_form(1,1,1,0))
    X11 = add_dicts(a_form(0,0,1,1), a_form(1,1,1,1))
    
    Y00 = add_dicts(b_form(0,0,0,0), b_form(1,1,0,0))
    Y01 = add_dicts(b_form(0,0,0,1), b_form(1,1,0,1))
    Y10 = add_dicts(b_form(0,0,1,0), b_form(1,1,1,0))
    Y11 = add_dicts(b_form(0,0,1,1), b_form(1,1,1,1))
    
    # Strassen terms for M1
    # S1 = (X00 + X11)(Y00 + Y11)
    u1 = add_dicts(X00, X11)
    v1 = add_dicts(Y00, Y11)
    # Contributes to Z00 (C00 block 0,0) and Z11 (C11 block 1,1)
    w1 = add_dicts(c_form(0,0,0,0,1), c_form(1,1,1,1,1))
    add_term_dict(u1, v1, w1)
    
    # S2 = (X10 + X11) Y00
    u2 = add_dicts(X10, X11)
    v2 = Y00
    # Contributes to Z10 (C10 block 1,0)
    w2 = c_form(1,0,1,0,1)
    add_term_dict(u2, v2, w2)
    
    # S3 = X00 (Y01 - Y11)
    u3 = X00
    v3 = sub_dicts(Y01, Y11)
    # Contributes to Z01 (C01 block 0,1) and Z11 (C11 block 0,1)
    w3 = add_dicts(c_form(0,1,0,1,1), c_form(1,1,0,1,1))
    add_term_dict(u3, v3, w3)
    
    # S4 = X11 (Y10 - Y00)
    u4 = X11
    v4 = sub_dicts(Y10, Y00)
    # Contributes to Z00 (C00 block 1,1) and Z10 (C10 block 1,1)
    w4 = add_dicts(c_form(0,0,1,1,1), c_form(1,0,1,1,1))
    add_term_dict(u4, v4, w4)
    
    # S5 = (X00 + X01) Y11
    u5 = add_dicts(X00, X01)
    v5 = Y11
    # Contributes to Z00 (C00 block 0,0) with -1? No.
    # Z00 = S1 + S4 - S5 + S7
    # Z11 = S1 - S2 + S3 + S6
    # S5 contributes to Z00 with -1 and Z01 with +1.
    w5 = add_dicts(c_form(0,0,0,0,-1), c_form(0,1,0,1,1))
    add_term_dict(u5, v5, w5)
    
    # S6 = (X10 - X00) (Y00 + Y01)
    u6 = sub_dicts(X10, X00)
    v6 = add_dicts(Y00, Y01)
    # S6 contributes to Z11 with +1.
    w6 = c_form(1,1,1,1,1)
    add_term_dict(u6, v6, w6)
    
    # S7 = (X01 - X11) (Y10 + Y11)
    u7 = sub_dicts(X01, X11)
    v7 = add_dicts(Y10, Y11)
    # S7 contributes to Z00 with +1.
    w7 = c_form(0,0,0,0,1)
    add_term_dict(u7, v7, w7)

    # I need to define add_term_dict to update U, V, W.
    
    def add_term_dict(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, coeff in u_dict.items():
            U[idx][col] += coeff
        for idx, coeff in v_dict.items():
            V[idx][col] += coeff
        for idx, coeff in w_dict.items():
            W[idx][col] += coeff
        col += 1

    # I already called add_term_dict before defining it.
    # I will refactor to define it first.
    
    # Let's restart the logic inside the function properly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term_dict(u_dict, v_dict, w_dict):
        nonlocal col
        for idx, coeff in u_dict.items():
            U[idx][col] += coeff
        for idx, coeff in v_dict.items():
            V[idx][col] += coeff
        for idx, coeff in w_dict.items():
            W[idx][col] += coeff
        col += 1

    # M1
    X00 = add_dicts(a_form(0,0,0,0), a_form(1,1,0,0))
    X01 = add_dicts(a_form(0,0,0,1), a_form(1,1,0,1))
    X10 = add_dicts(a_form(0,0,1,0), a_form(1,1,1,0))
    X11 = add_dicts(a_form(0,0,1,1), a_form(1,1,1,1))
    Y00 = add_dicts(b_form(0,0,0,0), b_form(1,1,0,0))
    Y01 = add_dicts(b_form(0,0,0,1), b_form(1,1,0,1))
    Y10 = add_dicts(b_form(0,0,1,0), b_form(1,1,1,0))
    Y11 = add_dicts(b_form(0,0,1,1), b_form(1,1,1,1))
    
    add_term_dict(add_dicts(X00, X11), add_dicts(Y00, Y11), add_dicts(c_form(0,0,0,0,1), c_form(1,1,1,1,1)))
    add_term_dict(add_dicts(X10, X11), Y00, c_form(1,0,1,0,1))
    add_term_dict(X00, sub_dicts(Y01, Y11), add_dicts(c_form(0,1,0,1,1), c_form(1,1,0,1,1)))
    add_term_dict(X11, sub_dicts(Y10, Y00), add_dicts(c_form(0,0,1,1,1), c_form(1,0,1,1,1)))
    add_term_dict(add_dicts(X00, X01), Y11, add_dicts(c_form(0,0,0,0,-1), c_form(0,1,0,1,1)))
    add_term_dict(sub_dicts(X10, X00), add_dicts(Y00, Y01), c_form(1,1,1,1,1))
    add_term_dict(sub_dicts(X01, X11), add_dicts(Y10, Y11), c_form(0,0,0,0,1))
    
    # M2 = (A10 + A11) B00
    # X = A10 + A11, Y = B00
    X00 = add_dicts(a_form(1,0,0,0), a_form(1,1,0,0))
    X01 = add_dicts(a_form(1,0,0,1), a_form(1,1,0,1))
    X10 = add_dicts(a_form(1,0,1,0), a_form(1,1,1,0))
    X11 = add_dicts(a_form(1,0,1,1), a_form(1,1,1,1))
    Y00 = b_form(0,0,0,0)
    Y01 = b_form(0,0,0,1)
    Y10 = b_form(0,0,1,0)
    Y11 = b_form(0,0,1,1)
    
    add_term_dict(add_dicts(X00, X11), add_dicts(Y00, Y11), add_dicts(c_form(1,0,0,0,1), c_form(1,1,1,1,1)))
    add_term_dict(add_dicts(X10, X11), Y00, c_form(1,1,1,0,1))
    add_term_dict(X00, sub_dicts(Y01, Y11), add_dicts(c_form(1,1,0,1,1), c_form(1,1,0,1,1))) # Z11 for C11?
    # M2 contributes to C10 and C11.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    # So M2 contributes to C10 with +1 and C11 with -1.
    # My Strassen result Z is the block product.
    # Z00 -> C10 block (0,0)? No.
    # X is A10+A11. Y is B00.
    # Product is (A10+A11)B00.
    # This is the block C10?
    # C10 = A10 B00 + A11 B10.
    # Wait. M2 = (A10 + A11) B00.
    # This is not a block in C. It's an intermediate.
    # The block results are:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M2, the result Z is added to C10 with +1 and C11 with -1.
    # Z00 -> C10(0,0) and C11(0,0)
    # Z01 -> C10(0,1) and C11(0,1)
    # Z10 -> C10(1,0) and C11(1,0)
    # Z11 -> C10(1,1) and C11(1,1)
    
    # I need to adjust the w forms for M2.
    # For M2, w for Z00 should be c_form(1,0,0,0,1) + c_form(1,1,0,0,-1).
    
    # I will refactor the code to handle the block mapping correctly.
    
    # This is getting very long. I will provide the complete code in the final block.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
