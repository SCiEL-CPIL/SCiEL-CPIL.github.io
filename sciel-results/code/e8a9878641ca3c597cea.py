import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition for 4x4 matrix multiplication.
    """
    start_time = time.time()
    DEADLINE = 145.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize matrices
    # U: (n*m, rank) -> (16, 49)
    # V: (m*p, rank) -> (16, 49)
    # W: (p*n, rank) -> (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Recursive Strassen construction
    # We define the 7 block-level products for 4x4 (split into 2x2 blocks of 2x2 matrices)
    # Then for each block product, we use Strassen for 2x2 (7 scalar mults).
    
    # Block indices: 0,1 for rows/cols of 2x2 blocks.
    # Global indices: 2*bi + r, 2*bj + c
    
    # Define the 7 terms of Strassen for the 2x2 block matrix multiplication
    # L and R are 2x2 matrices of linear combinations.
    # We represent each element as a dict: {(global_row, global_col): coeff}
    
    def a_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
        
    def b_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)
        
    def c_elem(bi, bj, r, c):
        return (2*bi + r, 2*bj + c)

    # Strassen terms for 2x2 block multiplication
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # Each Pi contributes to C blocks:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We iterate through the 7 block terms
    # For each term, we have L (2x2) and R (2x2)
    # We then apply Strassen to the 2x2 product L*R using 7 scalar multiplications.
    
    # Define the 7 block terms
    # term: (L_dict, R_dict, contrib_dict)
    # L_dict[r][c] = {(row, col): coeff}
    # R_dict[r][c] = {(row, col): coeff}
    # contrib_dict[(pr, pc)] = [(out_row, out_col, coeff)]
    
    terms = []
    
    # Term 0: P1
    L0 = [[{} for _ in range(2)] for _ in range(2)]
    R0 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            ar, ac = a_elem(0, 0, r, c)
            dr, dc = a_elem(1, 1, r, c)
            L0[r][c] = {(ar, ac): 1, (dr, dc): 1}
            er, ec = b_elem(0, 0, r, c)
            hr, hc = b_elem(1, 1, r, c)
            R0[r][c] = {(er, ec): 1, (hr, hc): 1}
    contrib0 = {
        (0,0): [(0,0,1), (2,0,1)],  # C11, C21? No, C11 is block (0,0), C22 is block (1,1).
        # Wait, C11 is top-left block. C22 is bottom-right.
        # C11 = P1 + ...
        # P1 contributes +1 to C11 and +1 to C22.
        # My indices:
        # Block (0,0) is rows 0-1, cols 0-1.
        # Block (1,1) is rows 2-3, cols 2-3.
        # So (0,0) in block coords -> global (0,0) to (1,1).
        # (1,1) in block coords -> global (2,2) to (3,3).
        # The contrib dict maps block result position (pr, pc) to list of global contributions.
        # For P1:
        # Result is 2x2.
        # Element (0,0) of P1 contributes to C11 (block 0,0) and C22 (block 1,1).
        # Element (0,1) of P1 contributes to C11 and C22.
        # ...
        # So for each (pr, pc) in 0..1, 0..1, we add to global block (0,0) and (1,1).
    }
    # Let's generalize the contrib logic.
    # P1 contributes to C11 and C22 with coeff +1.
    # P2 contributes to C21 with +1 and C22 with -1.
    # P3 contributes to C12 with +1 and C22 with +1.
    # P4 contributes to C11 with +1 and C21 with +1.
    # P5 contributes to C11 with -1 and C12 with +1.
    # P6 contributes to C22 with +1.
    # P7 contributes to C11 with +1.
    
    # Block indices for C:
    # C11: (0,0), C12: (0,1), C21: (1,0), C22: (1,1)
    
    contrib_defs = [
        { (0,0): [(0,0,1)], (0,1): [(0,0,1)], (1,0): [(0,0,1)], (1,1): [(0,0,1)],
          # Wait, P1 is a 2x2 matrix.
          # P1[0,0] contributes to C11[0,0], C22[0,0]
          # P1[0,1] contributes to C11[0,1], C22[0,1]
          # etc.
          # So for each (pr, pc), we add to global block (0,0) at (pr,pc) and (1,1) at (pr,pc).
          # My contrib dict should map (pr, pc) to list of (global_block_r, global_block_c, coeff)
          # Then I expand to global indices.
        }
    ]
    
    # Let's just hardcode the contributions for each term's 2x2 result.
    # Term t, result position (pr, pc) -> list of (out_br, out_bc, coeff)
    
    contribs = [
        # T0: P1
        { (0,0): [(0,0,1), (1,1,1)], (0,1): [(0,0,1), (1,1,1)], (1,0): [(0,0,1), (1,1,1)], (1,1): [(0,0,1), (1,1,1)] },
        # T1: P2
        { (0,0): [(1,0,1), (1,1,-1)], (0,1): [(1,0,1), (1,1,-1)], (1,0): [(1,0,1), (1,1,-1)], (1,1): [(1,0,1), (1,1,-1)] },
        # T2: P3
        { (0,0): [(0,1,1), (1,1,1)], (0,1): [(0,1,1), (1,1,1)], (1,0): [(0,1,1), (1,1,1)], (1,1): [(0,1,1), (1,1,1)] },
        # T3: P4
        { (0,0): [(0,0,1), (1,0,1)], (0,1): [(0,0,1), (1,0,1)], (1,0): [(0,0,1), (1,0,1)], (1,1): [(0,0,1), (1,0,1)] },
        # T4: P5
        { (0,0): [(0,0,-1), (0,1,1)], (0,1): [(0,0,-1), (0,1,1)], (1,0): [(0,0,-1), (0,1,1)], (1,1): [(0,0,-1), (0,1,1)] },
        # T5: P6
        { (0,0): [(1,1,1)], (0,1): [(1,1,1)], (1,0): [(1,1,1)], (1,1): [(1,1,1)] },
        # T6: P7
        { (0,0): [(0,0,1)], (0,1): [(0,0,1)], (1,0): [(0,0,1)], (1,1): [(0,0,1)] }
    ]

    # Define L and R for each term
    # L[t][r][c], R[t][r][c]
    
    Ls = []
    Rs = []
    
    for t in range(7):
        L = [[{} for _ in range(2)] for _ in range(2)]
        R = [[{} for _ in range(2)] for _ in range(2)]
        
        if t == 0: # P1
            for r in range(2):
                for c in range(2):
                    L[r][c] = {a_elem(0,0,r,c): 1, a_elem(1,1,r,c): 1}
                    R[r][c] = {b_elem(0,0,r,c): 1, b_elem(1,1,r,c): 1}
        elif t == 1: # P2
            for r in range(2):
                for c in range(2):
                    L[r][c] = {a_elem(1,0,r,c): 1, a_elem(1,1,r,c): 1}
                    R[r][c] = {b_elem(0,0,r,c): 1}
        elif t == 2: # P3
            for r in range(2):
                for c in range(2):
                    L[r][c] = {a_elem(0,0,r,c): 1}
                    R[r][c] = {b_elem(0,1,r,c): 1, b_elem(1,1,r,c): -1}
        elif t == 3: # P4
            for r in range(2):
                for c in range(2):
                    L[r][c] = {a_elem(1,1,r,c): 1}
                    R[r][c] = {b_elem(1,0,r,c): 1, b_elem(0,0,r,c): -1}
        elif t == 4: # P5
            for r in range(2):
                for c in range(2):
                    L[r][c] = {a_elem(0,0,r,c): 1, a_elem(0,1,r,c): 1}
                    R[r][c] = {b_elem(1,1,r,c): 1}
        elif t == 5: # P6
            for r in range(2):
                for c in range(2):
                    L[r][c] = {a_elem(1,0,r,c): 1, a_elem(0,0,r,c): -1}
                    R[r][c] = {b_elem(0,0,r,c): 1, b_elem(0,1,r,c): 1}
        elif t == 6: # P7
            for r in range(2):
                for c in range(2):
                    L[r][c] = {a_elem(0,1,r,c): 1, a_elem(1,1,r,c): -1}
                    R[r][c] = {b_elem(1,0,r,c): 1, b_elem(1,1,r,c): 1}
                    
        Ls.append(L)
        Rs.append(R)
        
    # Now iterate through terms and apply Strassen to each 2x2 product
    for t in range(7):
        L = Ls[t]
        R = Rs[t]
        contrib = contribs[t]
        
        # Extract a,b,c,d and e,f,g,h
        a, b, c, d = L[0][0], L[0][1], L[1][0], L[1][1]
        e, f, g, h = R[0][0], R[0][1], R[1][0], R[1][1]
        
        # Strassen sub-products
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        sub_prods = []
        
        # Helper to add dicts
        def add_dict(d1, d2, sign=1):
            res = {}
            for k, v in d1.items():
                res[k] = res.get(k, 0) + v
            for k, v in d2.items():
                res[k] = res.get(k, 0) + sign * v
            return res
        
        def sub_dict(d1, d2):
            return add_dict(d1, d2, -1)
            
        m1_x = add_dict(a, d)
        m1_y = add_dict(e, h)
        sub_prods.append((m1_x, m1_y))
        
        m2_x = add_dict(c, d)
        m2_y = e
        sub_prods.append((m2_x, m2_y))
        
        m3_x = a
        m3_y = sub_dict(f, h)
        sub_prods.append((m3_x, m3_y))
        
        m4_x = d
        m4_y = sub_dict(g, e)
        sub_prods.append((m4_x, m4_y))
        
        m5_x = add_dict(a, b)
        m5_y = h
        sub_prods.append((m5_x, m5_y))
        
        m6_x = sub_dict(c, a)
        m6_y = add_dict(e, f)
        sub_prods.append((m6_x, m6_y))
        
        m7_x = sub_dict(b, d)
        m7_y = add_dict(g, h)
        sub_prods.append((m7_x, m7_y))
        
        # Strassen result contributions to 2x2 block P
        # P11 = M1 + M4 - M5 + M7
        # P12 = M3 + M5
        # P21 = M2 + M4
        # P22 = M1 - M2 + M3 + M6
        
        # Map sub-product index to contributions to P positions
        # k: (pr, pc, coeff)
        p_contribs = [
            [(0,0,1), (1,1,1)],    # M1
            [(1,0,1), (1,1,-1)],   # M2
            [(0,1,1), (1,1,1)],    # M3
            [(0,0,1), (1,0,1)],    # M4
            [(0,0,-1), (0,1,1)],   # M5
            [(1,1,1)],             # M6
            [(0,0,1)]              # M7
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            x, y = sub_prods[k]
            
            # Update U
            for (row, col), coeff in x.items():
                U[u_idx(row, col)][rank_idx] += coeff
            
            # Update V
            for (row, col), coeff in y.items():
                V[v_idx(row, col)][rank_idx] += coeff
            
            # Update W
            for (pr, pc, sub_coeff) in p_contribs[k]:
                if (pr, pc) in contrib:
                    for (out_br, out_bc, block_coeff) in contrib[(pr, pc)]:
                        # Global row = 2*out_br + pr
                        # Global col = 2*out_bc + pc
                        gr = 2 * out_br + pr
                        gc = 2 * out_bc + pc
                        W[w_idx(gr, gc)][rank_idx] += sub_coeff * block_coeff
                        
    return {"U": U, "V": V, "W": W}

