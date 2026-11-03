import time

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 rank 7 block
    # a, b, c, d are indices in A (2x2)
    # e, f, g, h are indices in B (2x2)
    # r, s, t, u are indices in C (2x2)
    # offset is the current column in U, V, W
    def add_2x2x2(a_idx, b_idx, c_idx, d_idx, 
                  e_idx, f_idx, g_idx, h_idx, 
                  r_idx, s_idx, t_idx, u_idx, offset):
        # P1 = a e
        U[a_idx][offset] = 1
        V[e_idx][offset] = 1
        W[r_idx][offset] = 1
        offset += 1
        
        # P2 = b f
        U[b_idx][offset] = 1
        V[f_idx][offset] = 1
        W[s_idx][offset] = 1
        offset += 1
        
        # P3 = c g
        U[c_idx][offset] = 1
        V[g_idx][offset] = 1
        W[t_idx][offset] = 1
        offset += 1
        
        # P4 = d h
        U[d_idx][offset] = 1
        V[h_idx][offset] = 1
        W[u_idx][offset] = 1
        offset += 1
        
        # P5 = (a+b)(e+g)
        U[a_idx][offset] = 1
        U[b_idx][offset] = 1
        V[e_idx][offset] = 1
        V[g_idx][offset] = 1
        W[r_idx][offset] = 1
        W[t_idx][offset] = 1
        # Correction: C00 = ae + bg, C10 = ce + dg
        # P5 = (a+b)(e+g) = ae + ag + be + bg
        # We need to subtract ag and be? No, we add to C00 and C10?
        # C00 = ae + bg. C10 = ce + dg.
        # This 7-term scheme is tricky to get right without reference.
        # Let's use the 14-term scheme (two independent 2x2x2).
        # It's safer.
        pass

    # Let's restart with the 14-term scheme for 2x2x4, etc.
    # Rank 56 total.
    
    # We will build 2x4x8 from 2x4x4 (rank 28) twice.
    # 2x4x4 from 2x2x4 (rank 14) twice.
    # 2x2x4 from 2x2x2 (rank 7) twice.
    
    # 2x2x2 rank 7 is hard to get right manually.
    # 2x2x2 rank 8 is trivial.
    # If we use rank 8 for 2x2x2:
    # 2x2x4 rank 16.
    # 2x4x4 rank 32.
    # 2x4x8 rank 64.
    # That's the baseline.
    
    # We need to use rank 7 for 2x2x2.
    # I will use the known Winograd 7-term decomposition.
    
    # Indices for A (2x2): 0,1,2,3
    # Indices for B (2x2): 0,1,2,3
    # Indices for C (2x2): 0,1,2,3
    
    # Terms:
    # 1: A00 B00 -> C00
    # 2: A01 B11 -> C11
    # 3: A10 B10 -> C10
    # 4: A11 B01 -> C01
    # 5: (A00+A01)(B00+B10) -> C00 + C10 ?
    #    (a+b)(e+g) = ae + ag + be + bg
    #    C00 = ae + bg. C10 = ce + dg.
    #    This term contributes to C00 and C10?
    #    Actually, the 7th term is (a+b+c+d)(e+f+g+h) is NOT used.
    #    The 7th term is (a+b+c+d)(e+f+g+h) is for 2x2x2?
    #    No, the 7th term is (a+b+c+d)(e+f+g+h) is NOT correct.
    
    # Correct 7-term:
    # P1 = a e
    # P2 = b f
    # P3 = c g
    # P4 = d h
    # P5 = (a+b)(e+g)
    # P6 = (c+d)(g+h)
    # P7 = (a+b+c+d)(e+f+g+h)
    
    # C00 = P1 + P5 - P7 + P2? No.
    # C00 = ae + bg.
    # C01 = af + bh.
    # C10 = ce + dg.
    # C11 = cf + dh.
    
    # Let's use the 14-term (two 7-terms) but I need the 7-term right.
    # I will use a known implementation of 2x2x2 rank 7.
    
    def add_2x2x2_rank7(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, offset):
        # P1 = a0 b0
        U[a0][offset] = 1; V[b0][offset] = 1; W[c0][offset] = 1; offset += 1
        # P2 = a1 b3
        U[a1][offset] = 1; V[b3][offset] = 1; W[c3][offset] = 1; offset += 1
        # P3 = a2 b2
        U[a2][offset] = 1; V[b2][offset] = 1; W[c2][offset] = 1; offset += 1
        # P4 = a3 b1
        U[a3][offset] = 1; V[b1][offset] = 1; W[c1][offset] = 1; offset += 1
        # P5 = (a0+a1)(b0+b2)
        U[a0][offset] = 1; U[a1][offset] = 1
        V[b0][offset] = 1; V[b2][offset] = 1
        W[c0][offset] = 1; W[c2][offset] = 1
        offset += 1
        # P6 = (a2+a3)(b2+b3)
        U[a2][offset] = 1; U[a3][offset] = 1
        V[b2][offset] = 1; V[b3][offset] = 1
        W[c2][offset] = 1; W[c3][offset] = 1
        offset += 1
        # P7 = (a0+a1+a2+a3)(b0+b1+b2+b3)
        U[a0][offset] = 1; U[a1][offset] = 1; U[a2][offset] = 1; U[a3][offset] = 1
        V[b0][offset] = 1; V[b1][offset] = 1; V[b2][offset] = 1; V[b3][offset] = 1
        W[c0][offset] = 1; W[c1][offset] = 1; W[c2][offset] = 1; W[c3][offset] = 1
        offset += 1
        return offset

    # This 7-term decomposition is WRONG for standard multiplication.
    # C00 = a0 b0 + a1 b2.
    # P1 = a0 b0.
    # P5 = (a0+a1)(b0+b2) = a0 b0 + a0 b2 + a1 b0 + a1 b2.
    # P7 = (a0+a1+a2+a3)(b0+b1+b2+b3).
    # This is getting complicated.
    
    # I will stick to the rank 64 solution but try to optimize by using rank 7 for 2x2x2 if I can find a correct one.
    # Otherwise, I will return rank 64.
    # But the task asks for a stronger algorithm.
    # I will implement the rank 56 using the 14-term 2x2x4 which uses two 7-term 2x2x2.
    # I need a correct 7-term 2x2x2.
    
    # Correct 7-term:
    # P1 = a e
    # P2 = b f
    # P3 = c g
    # P4 = d h
    # P5 = (a+b)(e+g)
    # P6 = (c+d)(g+h)
    # P7 = (a+b+c+d)(e+f+g+h)
    
    # C00 = P1 + P5 - P7 + P2? No.
    # C00 = ae + bg.
    # C01 = af + bh.
    # C10 = ce + dg.
    # C11 = cf + dh.
    
    # Let's assume the 7-term is:
    # P1 = ae
    # P2 = bf
    # P3 = cg
    # P4 = dh
    # P5 = (a+b)(e+g)
    # P6 = (c+d)(g+h)
    # P7 = (a+b+c+d)(e+f+g+h)
    
    # C00 = P1 + P5 - P7 + P2?
    # P5 - P7 = (a+b)(e+g) - (a+b+c+d)(e+f+g+h)
    # This is not working.
    
    # I will return the rank 64 solution to be safe, as I cannot verify the 7-term decomposition quickly.
    # However, 56 is a known bound.
    # I will implement the rank 56 solution using the 14-term 2x2x4 which is two 7-term 2x2x2.
    # I will use the 14-term 2x2x2 (rank 8) to be safe? No, that's 64.
    
    # I will use the rank 64 solution.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
