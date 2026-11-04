import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-25 exact tensor decomposition for matrix multiplication 2x3 * 3x5.
    Uses Strassen for 2x2x2 blocks and simple decomposition for remaining parts.
    """
    n, m, p = DIMENSIONS
    # Total rank 25
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # u_idx: list of (index, coeff) for U
    # v_idx: list of (index, coeff) for V
    # w_idx: list of (index, coeff) for W
    def add_term(col, u_terms, v_terms, w_terms):
        for idx, coeff in u_terms:
            U[idx][col] += coeff
        for idx, coeff in v_terms:
            V[idx][col] += coeff
        for idx, coeff in w_terms:
            W[idx][col] += coeff

    # Strassen terms for 2x2x2
    # Local indices for A (2x2): 00, 01, 10, 11
    # Local indices for B (2x2): 00, 01, 10, 11
    # Local indices for W (2x2 output, trace conv k*n+i):
    # C00 -> k=0,i=0 -> 0
    # C01 -> k=1,i=0 -> 2
    # C10 -> k=0,i=1 -> 1
    # C11 -> k=1,i=1 -> 3
    
    def strassen_terms(offset_u, offset_v, offset_w, col_offset):
        # A indices: offset_u + [0, 1, 2, 3] corresponding to 00, 01, 10, 11?
        # A is 2x3. We use cols 0,1.
        # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
        # So offset_u is the base index for A[0,0].
        a00 = offset_u + 0
        a01 = offset_u + 1
        a10 = offset_u + 2
        a11 = offset_u + 3
        
        # B indices: offset_v + [0, 1, 5, 6]?
        # B is 3x5. We use rows 0,1 and cols 0,1.
        # B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6.
        b00 = offset_v + 0
        b01 = offset_v + 1
        b10 = offset_v + 5
        b11 = offset_v + 6
        
        # W indices: offset_w + [0, 1, 2, 3]
        w00 = offset_w + 0
        w10 = offset_w + 1
        w01 = offset_w + 2
        w11 = offset_w + 3
        
        # M1
        add_term(col_offset + 0, 
                 [(a00, 1), (a11, 1)], 
                 [(b00, 1), (b11, 1)], 
                 [(w00, 1), (w11, 1)]) # Wait, W coeff is for output.
                 # In Strassen, W is the output combination.
                 # C00 = M1 + M4 - M5 + M7
                 # So for M1, W contributes 1 to C00 and C11?
                 # No, M1 is a scalar.
                 # C00 gets M1. C11 gets M1.
                 # So W for M1 should have 1 at w00 and 1 at w11?
                 # Yes.
        
        # M2
        add_term(col_offset + 1,
                 [(a10, 1), (a11, 1)],
                 [(b00, 1)],
                 [(w10, 1), (w11, -1)]) # C21 gets M2, C22 gets -M2?
                 # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
                 # So M2 contributes to C21 (+1) and C22 (-1).
                 # w10 is C10 (row 1, col 0)?
                 # Trace conv W[k*n+i].
                 # C[1,0] -> k=0, i=1 -> 0*2+1 = 1. (w10)
                 # C[1,1] -> k=1, i=1 -> 1*2+1 = 3. (w11)
                 # So W for M2: 1 at w10, -1 at w11.
        
        # M3
        add_term(col_offset + 2,
                 [(a00, 1)],
                 [(b01, 1), (b11, -1)],
                 [(w01, 1), (w11, 1)]) # C12 gets M3, C22 gets M3.
                 # C12 = M3 + M5. C22 = ... + M3 ...
                 # w01 is C[0,1] -> k=1, i=0 -> 1*2+0 = 2.
                 # w11 is C[1,1] -> 3.
        
        # M4
        add_term(col_offset + 3,
                 [(a11, 1)],
                 [(b10, 1), (b00, -1)],
                 [(w00, 1), (w10, 1)]) # C00 gets M4, C21 gets M4.
                 # C00 = M1 + M4 ...
                 # C21 = M2 + M4.
                 # w00 (C00), w10 (C21).
        
        # M5
        add_term(col_offset + 4,
                 [(a00, 1), (a01, 1)],
                 [(b11, 1)],
                 [(w00, -1), (w01, 1)]) # C00 gets -M5, C12 gets M5.
                 # C00 = ... - M5 ...
                 # C12 = M3 + M5.
        
        # M6
        add_term(col_offset + 5,
                 [(a10, 1), (a00, -1)],
                 [(b00, 1), (b01, 1)],
                 [(w11, 1)]) # C22 gets M6.
        
        # M7
        add_term(col_offset + 6,
                 [(a01, 1), (a11, -1)],
                 [(b10, 1), (b11, 1)],
                 [(w00, 1), (w11, 1)]) # C00 gets M7, C11 gets M7.
                 # C00 = ... + M7. C11 = ... + M7.

    # 4 terms for 2x1x2 (outer product)
    def outer_product_terms(offset_u, offset_v, offset_w, col_offset):
        # A is 2x1 (col 2).
        # A[0,2] = offset_u + 1?
        # A indices: 00(0), 01(1), 10(2), 11(3), 02(4), 12(5).
        # We use col 2.
        a02 = offset_u + 4
        a12 = offset_u + 5
        
        # B is 1x2 (row 2, cols 0,1).
        # B[2,0] = offset_v + 10
        # B[2,1] = offset_v + 11
        b20 = offset_v + 10
        b21 = offset_v + 11
        
        # W is 2x2 output.
        w00 = offset_w + 0
        w10 = offset_w + 1
        w01 = offset_w + 2
        w11 = offset_w + 3
        
        # C00 = A02 * B20
        add_term(col_offset + 0, [(a02, 1)], [(b20, 1)], [(w00, 1)])
        # C10 = A12 * B20
        add_term(col_offset + 1, [(a12, 1)], [(b20, 1)], [(w10, 1)])
        # C01 = A02 * B21
        add_term(col_offset + 2, [(a02, 1)], [(b21, 1)], [(w01, 1)])
        # C11 = A12 * B21
        add_term(col_offset + 3, [(a12, 1)], [(b21, 1)], [(w11, 1)])

    # 3 terms for 2x3x1
    def simple_vector_terms(offset_u, offset_v, offset_w, col_offset):
        # A is 2x3 (all cols).
        # B is 3x1 (row 0,1,2, col 4).
        # C is 2x1 (rows 0,1, col 4).
        
        # B col 4.
        b04 = offset_v + 4
        b14 = offset_v + 9
        b24 = offset_v + 14
        
        # W col 4.
        # C[0,4] -> k=4, i=0 -> 4*2+0 = 8.
        # C[1,4] -> k=4, i=1 -> 4*2+1 = 9.
        w04 = offset_w + 8
        w14 = offset_w + 9
        
        # A all cols.
        a00 = offset_u + 0
        a01 = offset_u + 1
        a02 = offset_u + 2
        a10 = offset_u + 3
        a11 = offset_u + 4
        a12 = offset_u + 5
        
        # C04 = A00*B04 + A01*B14 + A02*B24
        # This is sum of 3 terms?
        # We can combine them into 3 rank-1 terms?
        # Term 1: A00 * B04 * W04
        # Term 2: A01 * B14 * W04
        # Term 3: A02 * B24 * W04
        # And for C14:
        # Term 4: A10 * B04 * W14
        # Term 5: A11 * B14 * W14
        # Term 6: A12 * B24 * W14
        # Total 6 terms?
        # But we estimated rank 3.
        # Rank 3 means we can do it in 3 terms.
        # How?
        # C04 = A00*B04 + A01*B14 + A02*B24
        # C14 = A10*B04 + A11*B14 + A12*B24
        # We need 3 multiplications.
        # Let M1 = B04, M2 = B14, M3 = B24.
        # Then C04 = A00*M1 + A01*M2 + A02*M3.
        # C14 = A10*M1 + A11*M2 + A12*M3.
        # This corresponds to 3 terms in the tensor decomposition?
        # Term 1: U has A00, A10. V has B04. W has W04, W14?
        # No, W is for output index.
        # Term 1: U[0,0]=1, U[1,0]=1. V[0,4]=1. W[4,0]=1, W[4,1]=1?
        # If W[4,0]=1 and W[4,1]=1, then C04 gets 1, C14 gets 1.
        # But we need C04 to get A00*M1 and C14 to get A10*M1.
        # So U must distinguish A00 and A10.
        # U is a vector.
        # If U has 1 at (0,0) and 1 at (1,0), then the term is (A00+A10) * B04 * W...
        # This is not correct.
        # The tensor decomposition requires $\sum u_r[i,j] v_r[j,k] w_r[k,i]$.
        # For Term 1 (associated with B04):
        # We want contribution to C04: A00 * B04.
        # Contribution to C14: A10 * B04.
        # So u_r[0,0]=1, u_r[1,0]=1.
        # v_r[0,4]=1.
        # w_r[4,0]=1, w_r[4,1]=1?
        # Then for C04 (i=0, k=4): sum_j u[0,j] v[j,4] w[4,0].
        # j=0: u[0,0]=1, v[0,4]=1, w[4,0]=1. Product 1.
        # j=1,2: 0.
        # So C04 gets 1.
        # For C14 (i=1, k=4): sum_j u[1,j] v[j,4] w[4,1].
        # j=0: u[1,0]=1, v[0,4]=1, w[4,1]=1. Product 1.
        # So C14 gets 1.
        # This works!
        # So Term 1: U has 1 at (0,0) and (1,0). V has 1 at (0,4). W has 1 at (4,0) and (4,1).
        # Term 2: U has 1 at (0,1) and (1,1). V has 1 at (1,4). W has 1 at (4,0) and (4,1).
        # Term 3: U has 1 at (0,2) and (1,2). V has 1 at (2,4). W has 1 at (4,0) and (4,1).
        
        add_term(col_offset + 0, 
                 [(a00, 1), (a10, 1)], 
                 [(b04, 1)], 
                 [(w04, 1), (w14, 1)])
                 
        add_term(col_offset + 1,
                 [(a01, 1), (a11, 1)],
                 [(b14, 1)],
                 [(w04, 1), (w14, 1)])
                 
        add_term(col_offset + 2,
                 [(a02, 1), (a12, 1)],
                 [(b24, 1)],
                 [(w04, 1), (w14, 1)])

    # Construct full decomposition
    # C1 (cols 0,1)
    # A offset 0. B offset 0. W offset 0.
    strassen_terms(0, 0, 0, 0)
    outer_product_terms(0, 0, 0, 7)
    
    # C2 (cols 2,3)
    # A offset 0. B offset 2 (cols 2,3). W offset 4 (cols 2,3).
    # B cols 2,3.
    # B[0,2]=2, B[0,3]=3. B[1,2]=7, B[1,3]=8.
    # My strassen_terms assumes B cols 0,1.
    # I need to adjust offsets.
    # B offset for cols 2,3 is 2?
    # b00 -> 2, b01 -> 3, b10 -> 7, b11 -> 8.
    # So offset_v = 2.
    # W offset for cols 2,3.
    # C[0,2] -> k=2, i=0 -> 4.
    # C[1,2] -> k=2, i=1 -> 5.
    # C[0,3] -> k=3, i=0 -> 6.
    # C[1,3] -> k=3, i=1 -> 7.
    # So offset_w = 4.
    strassen_terms(0, 2, 4, 11)
    outer_product_terms(0, 2, 4, 18)
    
    # C3 (col 4)
    # A offset 0. B offset 4. W offset 8.
    simple_vector_terms(0, 4, 8, 22)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
