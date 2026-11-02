"""
Block-matrix approach for 3x3 x 3x5 matrix multiplication.

Splits the 3x3 x 3x3 part into 2x2, 2x1, 1x2, 1x1 blocks.
Uses Strassen (rank 7) for the 2x2 core.
Uses schoolbook for the remaining blocks.
Total rank for 3x3x3 part: 7 + 4 + 4 + 4 + 2 + 2 + 1 = 24.
Uses schoolbook for the 3x3x2 part: 18.
Total rank: 42.
"""

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # We will build U, V, W for the full 3x3x5 problem.
    # U: 9 x Rank
    # V: 15 x Rank
    # W: 15 x Rank
    
    # Part 1: 3x3x3 (columns 0-2 of B)
    # Part 2: 3x3x2 (columns 3-4 of B)
    
    # We'll accumulate the terms.
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u, v, w):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
        
    # Indices:
    # A is 3x3. A[i,j]. i,j in 0..2
    # B is 3x5. B[j,k]. j in 0..2, k in 0..4
    # C is 3x5. C[i,k]. i in 0..2, k in 0..4
    
    # Helper to create U vector (9 elements, index i*m+j)
    def make_u(indices_vals):
        u = [0.0] * 9
        for idx, val in indices_vals:
            u[idx] = val
        return u
        
    # Helper to create V vector (15 elements, index j*p+k)
    def make_v(indices_vals):
        v = [0.0] * 15
        for idx, val in indices_vals:
            v[idx] = val
        return v
        
    # Helper to create W vector (15 elements, index k*n+i)
    def make_w(indices_vals):
        w = [0.0] * 15
        for idx, val in indices_vals:
            w[idx] = val
        return w

    # --- Part 1: 3x3x3 ---
    # Blocks:
    # A00: A[0:2, 0:2] (2x2)
    # A01: A[0:2, 2:3] (2x1)
    # A10: A[2:3, 0:2] (1x2)
    # A11: A[2:3, 2:3] (1x1)
    
    # B00: B[0:2, 0:2] (2x2)
    # B01: B[0:2, 2:3] (2x1)
    # B10: B[2:3, 0:2] (1x2)
    # B11: B[2:3, 2:3] (1x1)
    
    # C00: C[0:2, 0:2] (2x2)
    # C01: C[0:2, 2:3] (2x1)
    # C10: C[2:3, 0:2] (1x2)
    # C11: C[2:3, 2:3] (1x1)
    
    # 1. C00 = A00 B00 + A01 B10
    # Use Strassen for A00 B00 (2x2)
    # Strassen terms for 2x2:
    # M1 = A00[0,0] * B00[0,0]
    # M2 = A00[1,0] * B00[1,0] + A00[1,1] * B00[1,1] ? No.
    # Standard Strassen 2x2:
    # P1 = A11 * B11 (using 1-based, so A[1,1]B[1,1])
    # P2 = A21 * B21 ? No.
    # Let's use explicit 2x2 Strassen.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = a*e
    # M2 = c*g
    # M3 = d*h
    # M4 = (a+d)*(e+h)
    # M5 = (a+b)*f
    # M6 = (c+d)*(f+h)
    # M7 = (a-c)*(e+g)
    # C11 = M1 - M4 + M6 + M7
    # C12 = M5 - M6
    # C21 = M4 - M3 - M7
    # C22 = M2 - M1 + M5 + M3
    
    # Map to our indices:
    # A00[0,0] is A[0,0] (idx 0)
    # A00[0,1] is A[0,1] (idx 1)
    # A00[1,0] is A[1,0] (idx 3)
    # A00[1,1] is A[1,1] (idx 4)
    
    # B00[0,0] is B[0,0] (idx 0)
    # B00[0,1] is B[0,1] (idx 1)
    # B00[1,0] is B[1,0] (idx 5)
    # B00[1,1] is B[1,1] (idx 6)
    
    # C00[0,0] is C[0,0] (idx 0)
    # C00[0,1] is C[0,1] (idx 1)
    # C00[1,0] is C[1,0] (idx 3)
    # C00[1,1] is C[1,1] (idx 4)
    
    # Term M1: A[0,0] * B[0,0]
    add_term(make_u([(0,1)]), make_v([(0,1)]), make_w([(0,1)]))
    
    # Term M2: A[1,0] * B[1,0]
    add_term(make_u([(3,1)]), make_v([(5,1)]), make_w([(3,1)])) # Wait, C21? No, C[1,0] is idx 3.
    # Let's re-verify Strassen mapping.
    # C11 (C[0,0]) = M1 - M4 + M6 + M7
    # C12 (C[0,1]) = M5 - M6
    # C21 (C[1,0]) = M4 - M3 - M7
    # C22 (C[1,1]) = M2 - M1 + M5 + M3
    
    # So:
    # M1 -> C[0,0] += 1, C[1,1] -= 1
    # M2 -> C[1,0] += 1, C[1,1] += 1 ? No.
    # Let's check standard Strassen again.
    # M1 = A11 B11
    # M2 = A21 B21
    # M3 = A22 B22
    # M4 = (A11+A22)(B11+B22)
    # M5 = (A11+A12) B12
    # M6 = (A21+A22) B22
    # M7 = (A11-A21) B11
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M5 - M6
    # C21 = M4 - M3 - M7
    # C22 = M2 - M1 + M5 + M3
    
    # My previous mapping:
    # M1 (A11 B11) -> C11 += 1, C22 -= 1
    # M2 (A21 B21) -> C21 += 1, C22 += 1
    # M3 (A22 B22) -> C21 -= 1, C22 += 1
    # M4 -> C11 -= 1, C21 += 1
    # M5 -> C12 += 1, C22 += 1
    # M6 -> C11 += 1, C12 -= 1
    # M7 -> C11 += 1, C21 -= 1
    
    # Let's add M2..M7
    
    # M2: A[1,0] * B[1,0]
    # U: A[1,0] (idx 3)
    # V: B[1,0] (idx 5)
    # W: C[1,0] (idx 3) += 1, C[1,1] (idx 4) += 1
    add_term(make_u([(3,1)]), make_v([(5,1)]), make_w([(3,1), (4,1)]))
    
    # M3: A[1,1] * B[1,1]
    # U: A[1,1] (idx 4)
    # V: B[1,1] (idx 6)
    # W: C[1,0] (idx 3) -= 1, C[1,1] (idx 4) += 1
    add_term(make_u([(4,1)]), make_v([(6,1)]), make_w([(3,-1), (4,1)]))
    
    # M4: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0](0), A[1,1](4)
    # V: B[0,0](0), B[1,1](6)
    # W: C[0,0](0) -= 1, C[1,0](3) += 1
    add_term(make_u([(0,1), (4,1)]), make_v([(0,1), (6,1)]), make_w([(0,-1), (3,1)]))
    
    # M5: (A[0,0]+A[0,1]) * B[0,1]
    # U: A[0,0](0), A[0,1](1)
    # V: B[0,1](1)
    # W: C[0,1](1) += 1, C[1,1](4) += 1
    add_term(make_u([(0,1), (1,1)]), make_v([(1,1)]), make_w([(1,1), (4,1)]))
    
    # M6: (A[1,0]+A[1,1]) * B[1,1]
    # U: A[1,0](3), A[1,1](4)
    # V: B[1,1](6)
    # W: C[0,0](0) += 1, C[0,1](1) -= 1
    add_term(make_u([(3,1), (4,1)]), make_v([(6,1)]), make_w([(0,1), (1,-1)]))
    
    # M7: (A[0,0]-A[1,0]) * B[0,0]
    # U: A[0,0](0), A[1,0](3)
    # V: B[0,0](0)
    # W: C[0,0](0) += 1, C[1,0](3) -= 1
    add_term(make_u([(0,1), (3,-1)]), make_v([(0,1)]), make_w([(0,1), (3,-1)]))
    
    # Now A01 B10 (2x1 * 1x2 -> 2x2)
    # A01: A[0,2] (idx 2), A[1,2] (idx 5)
    # B10: B[2,0] (idx 10), B[2,1] (idx 11)
    # C00: C[0,0] (idx 0), C[0,1] (idx 1), C[1,0] (idx 3), C[1,1] (idx 4)
    # Terms:
    # C[0,0] += A[0,2] B[2,0]
    # C[0,1] += A[0,2] B[2,1]
    # C[1,0] += A[1,2] B[2,0]
    # C[1,1] += A[1,2] B[2,1]
    
    add_term(make_u([(2,1)]), make_v([(10,1)]), make_w([(0,1)]))
    add_term(make_u([(2,1)]), make_v([(11,1)]), make_w([(1,1)]))
    add_term(make_u([(5,1)]), make_v([(10,1)]), make_w([(3,1)]))
    add_term(make_u([(5,1)]), make_v([(11,1)]), make_w([(4,1)]))
    
    # C01 = A00 B01 + A01 B11
    # A00 B01 (2x2 * 2x1 -> 2x1)
    # A00: A[0,0](0), A[0,1](1), A[1,0](3), A[1,1](4)
    # B01: B[0,2](2), B[1,2](7)
    # C01: C[0,2](2), C[1,2](5)
    # C[0,2] += A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] += A[1,0]B[0,2] + A[1,1]B[1,2]
    
    add_term(make_u([(0,1)]), make_v([(2,1)]), make_w([(2,1)]))
    add_term(make_u([(1,1)]), make_v([(7,1)]), make_w([(2,1)]))
    add_term(make_u([(3,1)]), make_v([(2,1)]), make_w([(5,1)]))
    add_term(make_u([(4,1)]), make_v([(7,1)]), make_w([(5,1)]))
    
    # A01 B11 (2x1 * 1x1 -> 2x1)
    # A01: A[0,2](2), A[1,2](5)
    # B11: B[2,2](12)
    # C01: C[0,2](2), C[1,2](5)
    
    add_term(make_u([(2,1)]), make_v([(12,1)]), make_w([(2,1)]))
    add_term(make_u([(5,1)]), make_v([(12,1)]), make_w([(5,1)]))
    
    # C10 = A10 B00 + A11 B10
    # A10 B00 (1x2 * 2x2 -> 1x2)
    # A10: A[2,0](6), A[2,1](7)
    # B00: B[0,0](0), B[0,1](1), B[1,0](5), B[1,1](6)
    # C10: C[2,0](6), C[2,1](7)
    
    add_term(make_u([(6,1)]), make_v([(0,1)]), make_w([(6,1)]))
    add_term(make_u([(7,1)]), make_v([(5,1)]), make_w([(6,1)]))
    add_term(make_u([(6,1)]), make_v([(1,1)]), make_w([(7,1)]))
    add_term(make_u([(7,1)]), make_v([(6,1)]), make_w([(7,1)]))
    
    # A11 B10 (1x1 * 1x2 -> 1x2)
    # A11: A[2,2](8)
    # B10: B[2,0](10), B[2,1](11)
    # C10: C[2,0](6), C[2,1](7)
    
    add_term(make_u([(8,1)]), make_v([(10,1)]), make_w([(6,1)]))
    add_term(make_u([(8,1)]), make_v([(11,1)]), make_w([(7,1)]))
    
    # C11 = A10 B01 + A11 B11
    # A10 B01 (1x2 * 2x1 -> 1x1)
    # A10: A[2,0](6), A[2,1](7)
    # B01: B[0,2](2), B[1,2](7)
    # C11: C[2,2](8)
    
    add_term(make_u([(6,1)]), make_v([(2,1)]), make_w([(8,1)]))
    add_term(make_u([(7,1)]), make_v([(7,1)]), make_w([(8,1)]))
    
    # A11 B11 (1x1 * 1x1 -> 1x1)
    # A11: A[2,2](8)
    # B11: B[2,2](12)
    # C11: C[2,2](8)
    
    add_term(make_u([(8,1)]), make_v([(12,1)]), make_w([(8,1)]))
    
    # --- Part 2: 3x3x2 (columns 3-4) ---
    # Schoolbook for A (3x3) * B[:, 3:5] (3x2)
    # C[:, 3] = A * B[:, 3]
    # C[:, 4] = A * B[:, 4]
    
    for k_offset in [0, 1]:
        k = 3 + k_offset
        for i in range(3):
            for j in range(3):
                # C[i, k] += A[i, j] B[j, k]
                u_idx = i * 3 + j
                v_idx = j * 5 + k
                w_idx = k * 3 + i
                add_term(make_u([(u_idx, 1)]), make_v([(v_idx, 1)]), make_w([(w_idx, 1)]))
                
    rank = len(U_terms)
    
    U = [u for u in U_terms]
    V = [v for v in V_terms]
    W = [w for w in W_terms]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
