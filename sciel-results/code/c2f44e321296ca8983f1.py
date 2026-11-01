"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition using Strassen for 4x (2x2x2) and 1x (2x2x1)
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a multiplication term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col_idx
        if col_idx >= rank:
            return
        for u_idx, val in u_coeffs.items():
            if val != 0:
                U[u_idx][col_idx] += val
        for v_idx, val in v_coeffs.items():
            if val != 0:
                V[v_idx][col_idx] += val
        for w_idx, val in w_coeffs.items():
            if val != 0:
                W[w_idx][col_idx] += val
        col_idx += 1

    # Process 4 blocks of 2 columns each
    for block in range(4):
        k1 = block * 2
        k2 = block * 2 + 1
        
        # Strassen products for A * B_block
        # A = [[a, b], [c, d]]
        # B_block = [[B0k1, B0k2], [B1k1, B1k2]]
        
        # P1 = (a+d)(B0k1 + B1k2)
        u1 = {0: 1, 3: 1} # a=A[0,0] idx 0, d=A[1,1] idx 3
        v1 = {k1: 1, 9 + k2: 1} # B0k1 idx k1, B1k2 idx 9+k2
        w1 = {2*k1 + 0: 1, 2*k1 + 1: 1, 2*k2 + 0: -1, 2*k2 + 1: -1}
        # C0k1 += P1, C1k1 += P1, C0k2 -= P1, C1k2 -= P1
        # Wait, let's re-verify W coefficients from formulas
        # C0k1 = P1 + P4 - P5 + P7
        # C0k2 = P5 - P1 - P4 + P6
        # C1k1 = P1 - P2 + P3 - P7
        # C1k2 = P2 - P3 + P1 - P6
        
        # So for P1:
        # W[2*k1+0] += 1
        # W[2*k1+1] += 1
        # W[2*k2+0] += -1
        # W[2*k2+1] += -1
        w1 = {2*k1 + 0: 1, 2*k1 + 1: 1, 2*k2 + 0: -1, 2*k2 + 1: -1}
        add_term(u1, v1, w1)
        
        # P2 = (c+d)B0k1
        u2 = {2: 1, 3: 1} # c=A[1,0] idx 2, d=A[1,1] idx 3
        v2 = {k1: 1}
        w2 = {2*k1 + 1: -1, 2*k2 + 1: 1}
        # C1k1 -= P2, C1k2 += P2
        w2 = {2*k1 + 1: -1, 2*k2 + 1: 1}
        add_term(u2, v2, w2)
        
        # P3 = a(B1k1 - B1k2)
        u3 = {0: 1}
        v3 = {9 + k1: 1, 9 + k2: -1}
        w3 = {2*k1 + 1: 1, 2*k2 + 1: -1}
        # C1k1 += P3, C1k2 -= P3
        w3 = {2*k1 + 1: 1, 2*k2 + 1: -1}
        add_term(u3, v3, w3)
        
        # P4 = d(B0k1 - B1k1)
        u4 = {3: 1}
        v4 = {k1: 1, 9 + k1: -1}
        w4 = {2*k1 + 0: 1, 2*k2 + 0: -1}
        # C0k1 += P4, C0k2 -= P4
        w4 = {2*k1 + 0: 1, 2*k2 + 0: -1}
        add_term(u4, v4, w4)
        
        # P5 = (a+b)B1k2
        u5 = {0: 1, 1: 1}
        v5 = {9 + k2: 1}
        w5 = {2*k1 + 0: -1, 2*k2 + 0: 1}
        # C0k1 -= P5, C0k2 += P5
        w5 = {2*k1 + 0: -1, 2*k2 + 0: 1}
        add_term(u5, v5, w5)
        
        # P6 = (c-a)(B0k1 + B1k1)
        u6 = {2: 1, 0: -1}
        v6 = {k1: 1, 9 + k1: 1}
        w6 = {2*k2 + 0: 1, 2*k2 + 1: -1}
        # C0k2 += P6, C1k2 -= P6
        w6 = {2*k2 + 0: 1, 2*k2 + 1: -1}
        add_term(u6, v6, w6)
        
        # P7 = (b-d)(B0k2 + B1k2)
        u7 = {1: 1, 3: -1}
        v7 = {k2: 1, 9 + k2: 1}
        w7 = {2*k1 + 0: 1, 2*k1 + 1: -1}
        # C0k1 += P7, C1k1 -= P7
        w7 = {2*k1 + 0: 1, 2*k1 + 1: -1}
        add_term(u7, v7, w7)

    # Process last column k=8
    k8 = 8
    # Q1 = a * B0k8
    u_q1 = {0: 1}
    v_q1 = {k8: 1}
    w_q1 = {2*k8 + 0: 1}
    add_term(u_q1, v_q1, w_q1)
    
    # Q2 = b * B1k8
    u_q2 = {1: 1}
    v_q2 = {9 + k8: 1}
    w_q2 = {2*k8 + 0: 1}
    add_term(u_q2, v_q2, w_q2)
    
    # Q3 = c * B0k8
    u_q3 = {2: 1}
    v_q3 = {k8: 1}
    w_q3 = {2*k8 + 1: 1}
    add_term(u_q3, v_q3, w_q3)
    
    # Q4 = d * B1k8
    u_q4 = {3: 1}
    v_q4 = {9 + k8: 1}
    w_q4 = {2*k8 + 1: 1}
    add_term(u_q4, v_q4, w_q4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
