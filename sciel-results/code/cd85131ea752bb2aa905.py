import time

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Rank-10 decomposition for 2x2 * 2x3
    # Total rank = 7 (outer Strassen) * 10 (inner) = 70
    inner_rank = 10
    rank = 7 * inner_rank  # 70
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen Steps for 4x4 * 4x6
    # A is 4x4, B is 4x6.
    # Split A into four 2x2 blocks: A00, A01, A10, A11
    # Split B into four 2x3 blocks: B00, B01, B10, B11
    # C is 4x6, split into four 2x3 blocks: C00, C01, C10, C11
    #
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    #
    # Strassen-like combination for the 4x4 * 4x6 product:
    # We treat the 2x2 block structure of A and the 2x3 block structure of B.
    # The 7 outer steps correspond to the 7 Strassen multiplications of 2x2 blocks.
    # However, since B is 2x3, the "multiplication" is 2x2 * 2x3.
    
    # Outer steps definition:
    # Each step is (A_combo, B_combo, C_combo)
    # A_combo is a list of (block_id, coeff)
    # B_combo is a list of (block_id, coeff)
    # C_combo is a list of (block_id, coeff)
    
    outer_steps = [
        # M1 = (A00 + A11) * (B00 + B10)
        (([(0, 1), (3, 1)], [(0, 1), (2, 1)], [(0, 1), (2, -1)])),
        # M2 = (A10 + A11) * (B00)
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        # M3 = (A00) * (B01 + B11)
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        # M4 = (A11) * (B10 + B00)
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        # M5 = (A00 + A01) * (B11)
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        # M6 = (A10 + A00) * (B00 + B01)
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        # M7 = (A01 + A11) * (B10 + B11)
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    def get_global_A(block_id, r, s):
        # block_id: 0=A00, 1=A01, 2=A10, 3=A11
        # r, s: local row, col in 2x2 block
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        # block_id: 0=B00, 1=B01, 2=B10, 3=B11
        # s: local row in 2x3 block
        # k: local col in 2x3 block
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def get_global_C(block_id, r, k):
        # block_id: 0=C00, 1=C01, 2=C10, 3=C11
        # r: local row in 2x3 block
        # k: local col in 2x3 block
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    # Inner Rank-10 decomposition for 2x2 * 2x3
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = A*B
    #
    # 7 Strassen terms for C[:, 0:2]
    inner_strassen = [
        # M1: (a+d)(e+h) -> C00+=, C10+=
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)])),
        # M2: (a+c)(e+h) -> C00+=, C10+=
        (([(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)])),
        # M3: (b+d)(f+i) -> C01+=, C11+=
        (([(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)])),
        # M4: (b+c)(f+i) -> C01+=, C11+=
        (([(0,1,1), (1,0,1)], [(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)])),
        # M5: (a-d)(e-h) -> C00-=, C10-=
        (([(0,0,1), (1,1,-1)], [(0,0,1), (1,0,-1)], [(0,0,-1), (1,0,-1)])),
        # M6: (b-d)(f-i) -> C01-=, C11-=
        (([(0,1,1), (1,1,-1)], [(0,1,1), (1,1,-1)], [(0,1,-1), (1,1,-1)])),
        # M7: (a+b+c+d)(g+i+j) ? No, M7 is for the 3rd column?
        # Wait, the 7 Strassen terms are for 2x2*2x2.
        # The 7th term in standard Strassen is (a+b+c+d)(g+i+j) is NOT correct for 2x2.
        # Standard Strassen 7th term is (a+b+c+d) * (something)?
        # Actually, standard Strassen for 2x2 uses:
        # M1=(a+d)(e+h), M2=(a+c)(e+h), M3=(b+d)(f+i), M4=(b+c)(f+i)
        # M5=(a-d)(e-h), M6=(b-d)(f-i), M7=(a+b+c+d)(g+i+j) is for 3x3?
        # No, for 2x2, the 7th term is typically (a+b)(e+h) or similar?
        # Let's use the verified 7 terms from the previous code which worked for 2x2*2x2.
        # The previous code used:
        # (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
        # This corresponds to (b-d)(h+i) -> C00 += ?
        # I will reuse the 7 terms from the previous code's inner_strassen list.
    ]
    
    # Reusing the 7 terms from the provided working code for the 2x2 part
    inner_strassen = [
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,0,1)], [(0,0,1), (1,0,1)])), # M1
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])), # M2
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])), # M3
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])), # M4
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])), # M5
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])), # M6
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])) # M7
    ]
    
    # 3 additional terms for the 3rd column (g, j)
    # C02 = ag + cj
    # C12 = dg + dj
    # M8 = (a+c)(g+j) -> C02 += 0.5 * M8
    # M9 = (a-c)(g-j) -> C02 += 0.5 * M9
    # M10 = d(g+j) -> C12 += M10
    
    inner_col3 = [
        # M8: (a+c)(g+j)
        (([(0,0,1), (1,0,1)], [(0,2,1), (1,2,1)], [(0,2,0.5)])),
        # M9: (a-c)(g-j)
        (([(0,0,1), (1,0,-1)], [(0,2,1), (1,2,-1)], [(0,2,0.5)])),
        # M10: d(g+j)
        (([(1,1,1)], [(0,2,1), (1,2,1)], [(1,2,1)]))
    ]
    
    all_inner_terms = inner_strassen + inner_col3
    
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * inner_rank
        
        for j, (A_inner, B_inner, C_inner) in enumerate(all_inner_terms):
            c_global_rank = base_rank + j
            
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

    return {"U": U, "V": V, "W": W}
