import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    # Pad p to 8
    p_pad = 8
    
    U = [[0] * 98 for _ in range(n * m)]
    V = [[0] * 98 for _ in range(m * p_pad)]
    W = [[0] * 98 for _ in range(p_pad * n)]
    
    # Strassen 2x2 coefficients
    # M_s = U_s(M) * V_s(N)
    # P = sum W_s * M_s
    # U_s, V_s, W_s are linear forms
    
    # Define 2x2 Strassen steps
    # Indices for 2x2: 00, 01, 10, 11
    strassen_2x2 = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U is in M, V is in N, W is in P
        # s=1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # Wait, P coeffs are for P00, P01, P10, P11
        # Let's store W as dict of P index to coeff
    ]
    
    # Correct Strassen 2x2:
    # M1: (m00+m11)(n00+n11) -> P00: 1, P11: 1
    # M2: (m10+m11)n00 -> P10: 1, P11: -1
    # M3: m00(n01-n11) -> P01: 1, P11: 1
    # M4: m11(n10-n00) -> P00: 1, P10: 1
    # M5: (m00+m01)n11 -> P00: -1, P01: 1
    # M6: (m10-m00)(n00+n01) -> P11: 1
    # M7: (m01-m11)(n10+n11) -> P00: 1
    
    # Let's define them properly
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Strassen 4x4 steps
    # Each step t produces a 2x2 matrix product
    # We need to map the "block" linear forms to global A/B entries
    # and then apply the 2x2 Strassen decomposition
    
    # Define 4x4 Strassen steps in terms of block indices (r,c)
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    S4 = [
        # (U_blocks, V_blocks)
        # U_blocks is a dict of block (r,c) to coeff
        # But we need to expand to entries
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    
    # P block coeffs for 4x4
    # C00, C01, C10, C11
    W4 = [
        {(0,0): 1, (1,1): 1}, # M1
        {(1,0): 1, (1,1): -1}, # M2
        {(0,1): 1, (1,1): 1}, # M3
        {(0,0): 1, (1,0): 1}, # M4
        {(0,0): -1, (0,1): 1}, # M5
        {(1,1): 1}, # M6
        {(0,0): 1} # M7
    ]
    
    # Helper to expand block form to entry form
    def expand_block_form(block_form, is_A=True):
        # block_form: dict of (r,c) -> coeff
        # Returns dict of (i,j) -> coeff for global matrix
        entry_form = {}
        for (r,c), coeff in block_form.items():
            for u in range(2):
                for v in range(2):
                    i = 2*r + u
                    j = 2*c + v
                    if is_A:
                        key = (i, j)
                    else:
                        key = (j, i) # B is (row, col) = (j, k)
                        # Wait, B blocks are (r,c) where r is row block, c is col block
                        # B_{rc} has entries B_{2r+u, 2c+v}
                        # So key is (2r+u, 2c+v)
                        key = (2*r+u, 2*c+v)
                    entry_form[key] = entry_form.get(key, 0) + coeff
        return entry_form
    
    # Generate terms
    col = 0
    for k_offset in [0, 4]:
        # k_offset 0 for C1 (cols 0-3), 4 for C2 (cols 4-7)
        for t in range(7):
            U_blocks = S4[t][0]
            V_blocks = S4[t][1]
            W_blocks = W4[t]
            
            # Expand block forms to entry forms for A and B
            # A entry form: (i, j) -> coeff
            # B entry form: (j, k) -> coeff
            # Note: B is shared across the 7 Strassen steps for a given k_offset
            # But the linear form in B is different for each t
            
            # For each 2x2 Strassen step s
            for s in range(7):
                U_2x2 = S2[s][0]
                V_2x2 = S2[s][1]
                W_2x2 = S2[s][2]
                
                # Combine U_blocks and U_2x2
                # U_total is a linear form in A entries
                # U_blocks gives coefficients for A_{2r+u, 2c+v}
                # U_2x2 gives coefficients for m_{u,v} within the block product
                
                # The product is a bilinear form? No, it's a linear form in A and a linear form in B.
                # The term is (sum A_{ij} u_{ij}) * (sum B_{jk} v_{jk})
                # So U_total_{ij} is the coefficient of A_{ij} in the first factor.
                
                # How to combine?
                # The first factor is a matrix M_t = sum_{rc} coeff_{rc} A_{rc}
                # M_t is a 2x2 matrix of 2x2 matrices? No, M_t is a 2x2 matrix.
                # Wait, A_{rc} is a 2x2 matrix.
                # So M_t is a 2x2 matrix where each entry is a linear combination of A entries.
                # Specifically, (M_t)_{uv} = sum_{rc} coeff_{rc} * (A_{rc})_{uv}
                # (A_{rc})_{uv} = A_{2r+u, 2c+v}
                
                # So for a fixed (u,v) in 2x2, the coefficient of A_{2r+u, 2c+v} is coeff_{rc}.
                
                # Now apply 2x2 Strassen to M_t * N_t.
                # U_2x2 is a linear form in m_{uv}.
                # So U_total is a linear form in A entries.
                # U_total_{2r+u, 2c+v} = sum_{uv' such that (u',v') in U_2x2} U_2x2[(u',v')] * coeff_{rc}
                # Wait, U_2x2 is a form in the entries of the 2x2 matrix M_t.
                # So if U_2x2 has term m_{uv}, it picks the (u,v)-th entry of M_t.
                # The (u,v)-th entry of M_t is sum_{rc} coeff_{rc} A_{2r+u, 2c+v}.
                
                U_total = {}
                for (u,v), c_u in U_2x2.items():
                    for (r,c), c_b in U_blocks.items():
                        i = 2*r + u
                        j = 2*c + v
                        U_total[(i,j)] = U_total.get((i,j), 0) + c_u * c_b
                        
                V_total = {}
                for (u,v), c_v in V_2x2.items():
                    for (r,c), c_b in V_blocks.items():
                        # B entry is (2r+u, 2c+v + k_offset)
                        j = 2*r + u
                        k = 2*c + v + k_offset
                        V_total[(j,k)] = V_total.get((j,k), 0) + c_v * c_b
                        
                W_total = {}
                # W_2x2 is a form in P entries (0,0), (0,1), (1,0), (1,1)
                # W_blocks is a form in C blocks (0,0), (0,1), (1,0), (1,1)
                # P is the 2x2 result block.
                # P_{uv} contributes to C_{rc} with coeff from W_blocks[rc]
                # And P_{uv} is formed from M_s with coeff W_2x2[(u,v)]
                
                # So for each block (r,c) in C:
                # Coeff of C_{2r+u, 2c+v} is sum_{uv'} W_blocks[(r,c)] * W_2x2[(u',v')]
                # Wait, P is a 2x2 matrix.
                # P_{uv} = sum_s W_2x2_s[(u,v)] M_s
                # C_{rc} = sum_t W_4_t[(r,c)] P_t
                # So C_{2r+u, 2c+v} = sum_t W_4_t[(r,c)] P_t_{uv}
                # = sum_t W_4_t[(r,c)] sum_s W_2x2_s[(u,v)] M_{t,s}
                
                # So for a fixed t, s:
                # Coeff of C_{2r+u, 2c+v} is W_4_t[(r,c)] * W_2x2_s[(u,v)]
                
                for (r,c), c_wc in W_blocks.items():
                    for (u,v), c_wp in W_2x2.items():
                        i = 2*r + u
                        k = 2*c + v + k_offset
                        W_total[(i,k)] = W_total.get((i,k), 0) + c_wc * c_wp
                        
                # Fill U, V, W
                for (i,j), val in U_total.items():
                    U[i*4+j][col] = val
                for (j,k), val in V_total.items():
                    V[j*8+k][col] = val
                for (i,k), val in W_total.items():
                    W[k*4+i][col] = val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
