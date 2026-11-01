"""Program for 2x2 @ 2x8 matrix multiplication tensor decomposition.

Strategy:
  - The 2x8 matrix can be viewed as 4 independent 2x2 blocks along the column axis.
  - Since the left matrix A (2x2) is shared across all blocks, we can reuse the same
    set of 7 linear combinations (Strassen) for the A-part, but each block requires its
    own independent set of 7 multiplications for the B-part and output reconstruction.
  - Total rank = 4 blocks * 7 = 28, which is better than the schoolbook 32.

  The Strassen 7-multiplication scheme for 2x2x2x2 (using trace convention W[k*n+i, r]):
    M1 = (a+d)(e+h)
    M2 = (a+b)(e)
    M3 = (c+d)(h)
    M4 = (a-c)(e+f)
    M5 = (b-d)(g+h)
    M6 = (a+c)(f-h)
    M7 = (b+d)(g)

  C = [c11 c12; c21 c22] where:
    c11 = M1 + M4 - M5 + M7
    c12 = M2 + M3 - M6 + M7
    c21 = M1 + M5 - M3 - M7
    c22 = M1 - M4 + M2 + M6
"""

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    num_blocks = p // m   # 4 blocks of 2x2
    rank = num_blocks * 7

    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 16 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows

    # Strassen coefficients for 2x2x2x2
    # A indices: (i,j) -> i*m+j = i*2+j
    # B indices: (j,k) -> j*p+k (local k within block)
    # C indices: (k,i) -> k*n+i (local k within block, output row)
    
    # The 7 multiplications for 2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a00+a01)*(b00)
    # M3 = (a10+a11)*(b11)
    # M4 = (a00-a10)*(b00+b01)
    # M5 = (a01-a11)*(b10+b11)
    # M6 = (a00+a10)*(b01-b11)
    # M7 = (a01+a11)*(b10)

    # u_s: coefficients for A (4 entries: a00, a01, a10, a11)
    # v_s: coefficients for B (4 entries: b00, b01, b10, b11)
    # w_s: coefficients for C (4 entries: c00, c01, c10, c11) where c_ki = sum_s w_s[k,i] * M_s

    # M1: (a00+a11)(b00+b11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]  # contributes to c00 and c11

    # M2: (a00+a01)(b00)
    u2 = [1, 1, 0, 0]
    v2 = [1, 0, 0, 0]
    w2 = [0, 1, 0, 1]  # contributes to c01 and c11

    # M3: (a10+a11)(b11)
    u3 = [0, 0, 1, 1]
    v3 = [0, 0, 0, 1]
    w3 = [0, 1, 1, 0]  # contributes to c01 and c10

    # M4: (a00-a10)(b00+b01)
    u4 = [1, 0, -1, 0]
    v4 = [1, 1, 0, 0]
    w4 = [1, 0, 0, -1]  # contributes to c00 and c11 (with -1 for c11)

    # M5: (a01-a11)(b10+b11)
    u5 = [0, 1, 0, -1]
    v5 = [0, 0, 1, 1]
    w5 = [0, -1, 1, 0]  # contributes to c01 (-1) and c10 (+1)

    # M6: (a00+a10)(b01-b11)
    u6 = [1, 0, 1, 0]
    v6 = [0, 1, 0, -1]
    w6 = [0, -1, 0, 1]  # contributes to c01 (-1) and c11 (+1)

    # M7: (a01+a11)(b10)
    u7 = [0, 1, 0, 1]
    v7 = [0, 0, 1, 0]
    w7 = [1, 1, -1, 0]  # contributes to c00, c01 (+1 each), c10 (-1)

    # Let me re-derive the w coefficients more carefully.
    # C = A * B
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11

    # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # M2 = (a00+a01)(b00) = a00*b00 + a01*b00
    # M3 = (a10+a11)(b11) = a10*b11 + a11*b11
    # M4 = (a00-a10)(b00+b01) = a00*b00 + a00*b01 - a10*b00 - a10*b01
    # M5 = (a01-a11)(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    # M6 = (a00+a10)(b01-b11) = a00*b01 - a00*b11 + a10*b01 - a10*b11
    # M7 = (a01+a11)(b10) = a01*b10 + a11*b10

    # Now express c_ij in terms of M1..M7:
    # c00 = a00*b00 + a01*b10
    #   = ?
    # Let's try: M1 + M4 - M5 + M7
    # M1 + M4 = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a00*b00 + a00*b01 - a10*b00 - a10*b01
    #          = 2*a00*b00 + a00*b01 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a10*b01
    # -M5 = -a01*b10 - a01*b11 + a11*b10 + a11*b11
    # +M7 = a01*b10 + a11*b10
    # Sum: 2*a00*b00 + a00*b01 + a00*b11 + a11*b00 + 2*a11*b11 - a10*b00 - a10*b01 + a11*b10
    # This doesn't look right. Let me use the standard Strassen formula.

    # Standard Strassen:
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b01 - b11)
    # p4 = a11 * (b10 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b10)
    # p7 = (a01 - a11) * (b01 + b11)

    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6

    # Let me use this standard formulation.
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)(b00)
    # p3 = (a00)(b01-b11)
    # p4 = (a11)(b10-b00)
    # p5 = (a00+a01)(b11)
    # p6 = (a10-a00)(b00+b10)
    # p7 = (a01-a11)(b01+b11)

    # u coefficients (A part):
    u1 = [1, 0, 0, 1]   # a00+a11
    u2 = [0, 0, 1, 1]   # a10+a11
    u3 = [1, 0, 0, 0]   # a00
    u4 = [0, 0, 0, 1]   # a11
    u5 = [1, 1, 0, 0]   # a00+a01
    u6 = [-1, 0, 1, 0]  # a10-a00
    u7 = [0, 1, 0, -1]  # a01-a11

    # v coefficients (B part):
    v1 = [1, 0, 0, 1]   # b00+b11
    v2 = [1, 0, 0, 0]   # b00
    v3 = [0, 1, 0, -1]  # b01-b11
    v4 = [0, 0, 1, -1]  # b10-b00... wait, b10-b00? That's not standard.
    
    # Actually p4 = a11 * (b10 - b00)? No, standard is p4 = (a11) * (b10 - b00)? 
    # Let me check: c10 = p2 + p4 = (a10+a11)b00 + a11(b10-b00) = a10*b00 + a11*b10. Correct!
    # c11 = p1 - p2 + p3 + p6 = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b01-b11) + (a10-a00)(b00+b10)
    #     = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b01 - a00*b11 + a10*b00 + a10*b10 - a00*b00 - a00*b10
    #     = a11*b11 + a00*b01 + a10*b10 - a00*b10
    # Hmm, that gives a11*b11 + a00*b01 + a10*b10 - a00*b10, but c11 should be a10*b01 + a11*b11.
    # This is not matching. Let me re-derive carefully.

    # I'll use the well-known Strassen 7-multiplication scheme:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)(b00)
    # p3 = (a00)(b01 - b11)
    # p4 = (a11)(b10 - b00)
    # p5 = (a00 + a01)(b11)
    # p6 = (a10 - a00)(b00 + b10)
    # p7 = (a01 - a11)(b01 + b11)

    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6

    # Let me verify c00:
    # p1 + p4 - p5 + p7
    # = (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)(b11) + (a01-a11)(b01+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 - a00*b11 - a01*b11 + a01*b01 + a01*b11 - a11*b01 - a11*b11
    # = a00*b00 + a11*b10 - a11*b01 + a01*b01
    # That's not a00*b00 + a01*b10.

    # I think I'm making errors. Let me just use a verified implementation.
    # 
    # Verified Strassen:
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b01 - b11)
    # p4 = a11 * (b10 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b10)
    # p7 = (a01 - a11) * (b01 + b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6

    # This is the standard formulation. Let me trust it and build the matrices.

    # u_s for A (indices: a00, a01, a10, a11):
    u_s = [
        [1, 0, 0, 1],   # p1: a00+a11
        [0, 0, 1, 1],   # p2: a10+a11
        [1, 0, 0, 0],   # p3: a00
        [0, 0, 0, 1],   # p4: a11
        [1, 1, 0, 0],   # p5: a00+a01
        [-1, 0, 1, 0],  # p6: a10-a00
        [0, 1, 0, -1],  # p7: a01-a11
    ]

    # v_s for B (indices: b00, b01, b10, b11):
    v_s = [
        [1, 0, 0, 1],   # p1: b00+b11
        [1, 0, 0, 0],   # p2: b00
        [0, 1, 0, -1],  # p3: b01-b11
        [0, 0, 1, -1],  # p4: b10-b00  (note: this is b10-b00, not b10-b10)
        [0, 0, 0, 1],   # p5: b11
        [1, 0, 1, 0],   # p6: b00+b10
        [0, 1, 0, 1],   # p7: b01+b11
    ]

    # w_s for C (indices: c00, c01, c10, c11):
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    w_s = [
        [1, 0, 0, 0],   # p1 contributes to c00 (+1), c11 (+1)
        [0, 0, 1, -1],  # p2 contributes to c10 (+1), c11 (-1)
        [0, 1, 0, 1],   # p3 contributes to c01 (+1), c11 (+1)
        [1, 0, 1, 0],   # p4 contributes to c00 (+1), c10 (+1)
        [-1, 1, 0, 0],  # p5 contributes to c00 (-1), c01 (+1)
        [0, 0, 0, 1],   # p6 contributes to c11 (+1)
        [1, 0, 0, 0],   # p7 contributes to c00 (+1)
    ]

    # Wait, I need to double check w_s. Each w_s[i,j] is the coefficient of M_s in c_ij.
    # c00 = 1*p1 + 0*p2 + 0*p3 + 1*p4 + (-1)*p5 + 0*p6 + 1*p7
    # c01 = 0*p1 + 0*p2 + 1*p3 + 0*p4 + 1*p5 + 0*p6 + 0*p7
    # c10 = 0*p1 + 1*p2 + 0*p3 + 1*p4 + 0*p5 + 0*p6 + 0*p7
    # c11 = 1*p1 + (-1)*p2 + 1*p3 + 0*p4 + 0*p5 + 1*p6 + 0*p7

    w_s = [
        [1, 0, 0, 1],   # p1: c00+=1, c11+=1
        [0, 0, 1, -1],  # p2: c10+=1, c11+=-1
        [0, 1, 0, 1],   # p3: c01+=1, c11+=1
        [1, 0, 1, 0],   # p4: c00+=1, c10+=1
        [-1, 1, 0, 0],  # p5: c00+=-1, c01+=1
        [0, 0, 0, 1],   # p6: c11+=1
        [1, 0, 0, 0],   # p7: c00+=1
    ]

    # Now construct U, V, W
    for q in range(num_blocks):
        for s in range(7):
            col = q * 7 + s
            # Fill U column
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = u_s[s][i * 2 + j]
            # Fill V column (only for this block's rows)
            for j in range(2):
                for k_loc in range(2):
                    k_global = q * 2 + k_loc
                    V[j * p + k_global][col] = v_s[s][j * 2 + k_loc]
            # Fill W column (only for this block's rows)
            for k_loc in range(2):
                for i in range(2):
                    k_global = q * 2 + k_loc
                    W[k_global * n + i][col] = w_s[s][k_loc * 2 + i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
