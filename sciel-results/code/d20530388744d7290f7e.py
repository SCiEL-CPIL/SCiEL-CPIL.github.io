def construct():
    n, m, p = 2, 2, 14
    rank = 49  # Rank of 4x4 matrix multiplication
    
    # We embed 2x2 * 2x14 into 4x4 * 4x4
    # A is 2x2, B is 2x14 (treat as 2x4 with 4 blocks of 2x2, but we have 7 blocks)
    # Actually, 2x14 = 2x(2*7), so we have 7 blocks of 2x2
    # We can embed into 4x4 by using a 4x4 matrix where the top-left 2x2 is A
    # and the right side has B embedded
    
    # The 4x4 * 4x4 = 4x4 multiplication has rank 49
    # We use the known 49-term decomposition
    
    # For the embedding:
    # A_4x4 = [[A, 0], [0, 0]] where A is 2x2
    # B_4x4 = [[B1, B2, B3, B4, ...], ...] but we need to fit 14 columns into 4x4
    
    # Actually, let's use a different approach: 
    # 2x2 * 2x14 can be seen as 7 separate 2x2 * 2x2 multiplications
    # But we can share the A factors!
    
    # The tensor for 2x2 * 2x14 has:
    # - A has 4 entries
    # - B has 28 entries (2x14)
    # - C has 28 entries (14x2, with W[k*n+i] convention)
    
    # We can use a rank-49 decomposition by embedding into 4x4*4x4
    # The 4x4*4x4 multiplication tensor has rank 49
    
    # Let me use the explicit 49-term decomposition for 4x4*4x4
    # and map it to our problem
    
    # Actually, the simplest approach: use the known fact that 
    # rank(2x2x14) <= 7 * rank(2x2x2) = 7*7 = 49 (direct sum)
    # But we can do better by sharing A factors
    
    # Let me try a different embedding: 
    # Treat the 2x14 as a 4x4 matrix with padding
    # B_padded = [[B[:, 0:4], B[:, 4:8]], [B[:, 8:12], B[:, 12:14] padded]]
    # This is getting complicated
    
    # Let me just return the current valid solution since I'm not confident
    # in the 4x4 embedding coefficients
    
    # Actually, let me try to use the 11-term algorithm for 2x2x4
    # and extend it to 2x2x14
    
    # The 11-term algorithm for 2x2*2x4 = 2x4:
    # This is a known result: rank(2x2x4) = 11
    
    # For 2x2x14, we can decompose 14 = 4+4+4+2
    # So rank <= 11 + 11 + 11 + rank(2x2x2) = 11+11+11+7 = 40
    
    # But this requires the 11-term coefficients which I need to get right
    
    # Let me implement the 11-term algorithm for 2x2*2x4
    # and then use it 3 times plus one 7-term for the remaining 2x2
    
    # 2x2*2x4 = 2x4 (11 terms)
    # 2x2*2x2 = 2x2 (7 terms)
    # Total: 3*11 + 7 = 40
    
    # I need the 11-term decomposition for 2x2*2x4
    # Let me derive it or use known coefficients
    
    # After research, the 11-term algorithm for 2x2*2x4 exists
    # Let me implement it carefully
    
    # For now, let me return the safe rank-49 solution
    # The 40-term solution would be a significant improvement but requires
    # correct 11-term coefficients
    
    # Let me try to implement the 11-term algorithm
    # The 11-term decomposition for multiplying 2x2 by 2x4:
    
    # A = [[a, b], [c, d]]
    # B = [[e, f, g, h], [i, j, k, l]]
    # C = A*B = 2x4
    
    # The 11 products are:
    # m1 = (a+d)(e+i)
    # m2 = (c+d)(e+i)  
    # m3 = (a+b)(f+j)
    # m4 = (c+b)(f+j)
    # m5 = (a+d)(g+k)
    # m6 = (c+d)(g+k)
    # m7 = (a+b)(h+l)
    # m8 = (c+b)(h+l)
    # m9 = a(e+i) - m1 + m2  ... this is getting complex
    
    # I'm not confident in the exact 11-term coefficients
    # Let me return the current valid rank-49 solution
    
    num_blocks = p // 2  # 7 blocks of 2 columns each
    rank = 7 * num_blocks  # 7 Strassen terms per block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    col = 0
    for b in range(num_blocks):
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
