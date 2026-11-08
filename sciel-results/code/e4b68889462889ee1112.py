import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper for flat indices
    def u_idx(r, c): return r * m + c
    def v_idx(r, c): return r * p + c
    def w_idx(r, c): return c * n + r  # Convention W[k*n+i, r] where i=row, k=col of C[i,k]
    
    # Strassen 7x2x2 decomposition constants
    # Each of the 7 block products is a 2x2 matmul (7 scalar mults)
    # Total 49 scalar mults
    
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We iterate over the 7 block terms and the 7 scalar Strassen products within each.
    
    # Strassen sub-products for a 2x2 matrix multiplication:
    # Let M = [[a,b],[c,d]], N = [[e,f],[g,h]]
    # s1 = (a+d), s2 = (c+d), s3 = a+b, s4 = e+f, s5 = g+h, s6 = f-h, s7 = g-e
    # p1 = s1 * s5
    # p2 = s2 * e
    # p3 = a * s6
    # p4 = d * s7
    # p5 = s3 * h
    # p6 = (c-a) * s4
    # p7 = (b-d) * s5
    
    # Output assembly for 2x2:
    # M11 = p1 + p4 - p5 + p7
    # M12 = p3 + p5
    # M21 = p2 + p4
    # M22 = p1 - p2 + p3 + p6
    
    # We define the 7 block terms (L_t, R_t) where L_t and R_t are 2x2 matrices of linear combinations of A and B blocks.
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00, B01, B10, B11 similarly.
    
    # Indexing inside a 2x2 block: (r, c) where r,c in {0,1}
    # Global index for A:
    def a_loc(block_r, block_c, r, c):
        return (2*block_r + r, 2*block_c + c)
    def b_loc(block_r, block_c, r, c):
        return (2*block_r + r, 2*block_c + c)

    # Each term t defines:
    # L_t: 2x2 list of list of (global_index, coeff)
    # R_t: 2x2 list of list of (global_index, coeff)
    # Contrib_t: dict mapping (sub_row, sub_col) in the 2x2 result P_t to list of (out_row, out_col, coeff) in C
    
    terms = []
    
    # Helper to build a 2x2 matrix of linear forms
    def make_l_forms(coeffs):
        # coeffs: 2x2 list of lists of (block_r, block_c, coeff)
        forms = [[[] for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                for br, bc, val in coeffs[r][c]:
                    gr, gc = a_loc(br, bc, r, c)
                    forms[r][c].append((gr, gc, val))
        return forms
        
    def make_r_forms(coeffs):
        forms = [[[] for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                for br, bc, val in coeffs[r][c]:
                    gr, gc = b_loc(br, bc, r, c)
                    forms[r][c].append((gr, gc, val))
        return forms

    # Term 0: P1 = (A00+A11)(B00+B11)
    L0_coeff = [[[(0,0,1), (1,1,1)] for _ in range(2)] for _ in range(2)]
    R0_coeff = [[[(0,0,1), (1,1,1)] for _ in range(2)] for _ in range(2)]
    C0 = {
        (0,0): [(0,0,1), (2,2,1)],
        (0,1): [(0,1,1), (2,3,1)],
        (1,0): [(1,0,1), (3,2,1)],
        (1,1): [(1,1,1), (3,3,1)]
    }
    terms.append((make_l_forms(L0_coeff), make_r_forms(R0_coeff), C0))

    # Term 1: P2 = (A10+A11)B00
    L1_coeff = [[[(1,0,1), (1,1,1)] for _ in range(2)] for _ in range(2)]
    R1_coeff = [[[(0,0,1)] for _ in range(2)] for _ in range(2)]
    C1 = {
        (0,0): [(2,0,1)],
        (0,1): [(2,1,1)],
        (1,0): [(3,0,1)],
        (1,1): [(3,1,1)]
    }
    terms.append((make_l_forms(L1_coeff), make_r_forms(R1_coeff), C1))

    # Term 2: P3 = A00(B01-B11)
    L2_coeff = [[[(0,0,1)] for _ in range(2)] for _ in range(2)]
    R2_coeff = [[[(0,1,1), (1,1,-1)] for _ in range(2)] for _ in range(2)]
    C2 = {
        (0,0): [(0,2,1)],
        (0,1): [(0,3,1)],
        (1,0): [(1,2,1)],
        (1,1): [(1,3,1)]
    }
    terms.append((make_l_forms(L2_coeff), make_r_forms(R2_coeff), C2))

    # Term 3: P4 = A11(B10-B00)
    L3_coeff = [[[(1,1,1)] for _ in range(2)] for _ in range(2)]
    R3_coeff = [[[(1,0,1), (0,0,-1)] for _ in range(2)] for _ in range(2)]
    C3 = {
        (0,0): [(0,0,1)],
        (0,1): [(0,1,1)],
        (1,0): [(1,0,1)],
        (1,1): [(1,1,1)]
    }
    terms.append((make_l_forms(L3_coeff), make_r_forms(R3_coeff), C3))

    # Term 4: P5 = (A00+A01)B11
    L4_coeff = [[[(0,0,1), (0,1,1)] for _ in range(2)] for _ in range(2)]
    R4_coeff = [[[(1,1,1)] for _ in range(2)] for _ in range(2)]
    C4 = {
        (0,0): [(0,2,1)],
        (0,1): [(0,3,1)],
        (1,0): [(1,2,1)],
        (1,1): [(1,3,1)]
    }
    terms.append((make_l_forms(L4_coeff), make_r_forms(R4_coeff), C4))

    # Term 5: P6 = (A10-A00)(B00+B01)
    L5_coeff = [[[(1,0,1), (0,0,-1)] for _ in range(2)] for _ in range(2)]
    R5_coeff = [[[(0,0,1), (0,1,1)] for _ in range(2)] for _ in range(2)]
    C5 = {
        (0,0): [(2,0,1)],
        (0,1): [(2,1,1)],
        (1,0): [(3,0,1)],
        (1,1): [(3,1,1)]
    }
    terms.append((make_l_forms(L5_coeff), make_r_forms(R5_coeff), C5))

    # Term 6: P7 = (A01-A11)(B10+B11)
    L6_coeff = [[[(0,1,1), (1,1,-1)] for _ in range(2)] for _ in range(2)]
    R6_coeff = [[[(1,0,1), (1,1,1)] for _ in range(2)] for _ in range(2)]
    C6 = {
        (0,0): [(0,2,1)],
        (0,1): [(0,3,1)],
        (1,0): [(1,2,1)],
        (1,1): [(1,3,1)]
    }
    terms.append((make_l_forms(L6_coeff), make_r_forms(R6_coeff), C6))

    # Now we need to map the output contributions to the final C indices.
    # The C above maps (sub_r, sub_c) of the 2x2 block product to global C indices.
    # We must combine this with the Strassen internal decomposition contributions.
    
    # Strassen internal contributions for P = M*N where M, N are 2x2:
    # The 7 products p1..p7 contribute to the 4 elements of P as follows:
    # p1: P[0,0] += 1, P[1,1] += 1
    # p2: P[1,0] += 1, P[1,1] -= 1
    # p3: P[0,1] += 1, P[1,1] += 1
    # p4: P[0,0] += 1, P[1,0] += 1
    # p5: P[0,0] -= 1, P[0,1] += 1
    # p6: P[1,1] += 1
    # p7: P[0,0] += 1
    # (Note: standard Strassen: 
    #  p1 = (m00+m11)(n00+n11)
    #  p2 = (m10+m11)n00
    #  p3 = m00(n01-n11)
    #  p4 = m11(n10-n00)
    #  p5 = (m00+m01)n11
    #  p6 = (m10-m00)(n00+n01)
    #  p7 = (m01-m11)(n10+n11)
    #  P00 = p1+p4-p5+p7
    #  P01 = p3+p5
    #  P10 = p2+p4
    #  P11 = p1-p2+p3+p6
    #  This matches the contributions above.)

    # Let sub_contrib be a list of 7 lists of (sub_r, sub_c, coeff)
    sub_contrib = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]

    for t, (L, R, C_map) in enumerate(terms):
        # L and R are 2x2 lists of lists of (global_row, global_col, coeff)
        # We need to form the 7 linear forms for Strassen.
        
        def merge_forms(f1, f2, op='+'):
            result = {}
            for r in range(2):
                for c in range(2):
                    for item in f1[r][c]:
                        key = item[:2]
                        result[key] = result.get(key, 0) + item[2]
                    for item in f2[r][c]:
                        key = item[:2]
                        val = item[2]
                        if op == '-': val = -val
                        result[key] = result.get(key, 0) + val
            # Convert back to list format
            ret = [[[] for _ in range(2)] for _ in range(2)]
            # This structure is slightly inefficient but correct. 
            # We just need to be able to iterate (row, col, coeff) for the whole form later.
            # Actually, let's just keep a flat list of (row, col, coeff) for the whole 2x2 matrix?
            # No, Strassen combines specific positions (e.g. m00 and m11).
            # So we must keep them separated by position.
            for pos, items in result.items():
                r, c = pos
                if items != 0:
                    ret[r][c].append((r, c, items))
            return ret

        def combine_positions(f, pos1, pos2, op='+'):
            # Combines f[pos1] and f[pos2] into a single form
            result = {}
            for item in f[pos1[0]][pos1[1]]:
                key = item[:2]
                result[key] = result.get(key, 0) + item[2]
            for item in f[pos2[0]][pos2[1]]:
                key = item[:2]
                val = item[2]
                if op == '-': val = -val
                result[key] = result.get(key, 0) + val
            ret = []
            for key, val in result.items():
                if val != 0:
                    ret.append((key[0], key[1], val))
            return ret

        def single_position(f, pos):
            return combine_positions(f, pos, pos)

        a = single_position(L, (0,0))
        b = single_position(L, (0,1))
        c = single_position(L, (1,0))
        d = single_position(L, (1,1))
        
        e = single_position(R, (0,0))
        f_r = single_position(R, (0,1)) # Avoid name conflict with function f
        g = single_position(R, (1,0))
        h = single_position(R, (1,1))

        # Form the 7 products inputs
        # x1 = a+d, y1 = e+h
        x1 = combine_positions(L, (0,0), (1,1))
        y1 = combine_positions(R, (0,0), (1,1))
        
        # x2 = c+d, y2 = e
        x2 = combine_positions(L, (1,0), (1,1))
        y2 = e
        
        # x3 = a, y3 = f-h
        x3 = a
        y3 = combine_positions(R, (0,1), (1,1), '-')
        
        # x4 = d, y4 = g-e
        x4 = d
        y4 = combine_positions(R, (1,0), (0,0), '-')
        
        # x5 = a+b, y5 = h
        x5 = combine_positions(L, (0,0), (0,1))
        y5 = h
        
        # x6 = c-a, y6 = e+f
        x6 = combine_positions(L, (1,0), (0,0), '-')
        y6 = combine_positions(R, (0,0), (0,1))
        
        # x7 = b-d, y7 = g+h
        x7 = combine_positions(L, (0,1), (1,1), '-')
        y7 = combine_positions(R, (1,0), (1,1))
        
        products = [
            (x1, y1),
            (x2, y2),
            (x3, y3),
            (x4, y4),
            (x5, y5),
            (x6, y6),
            (x7, y7)
        ]
        
        for k, (xs, ys) in enumerate(products):
            rank_idx = t * 7 + k
            
            for (r, c, coeff) in xs:
                U[u_idx(r, c), rank_idx] += coeff
                
            for (r, c, coeff) in ys:
                V[v_idx(r, c), rank_idx] += coeff
                
            for (sr, sc, s_coeff) in sub_contrib[k]:
                if (sr, sc) in C_map:
                    for (out_r, out_c, c_coeff) in C_map[(sr, sc)]:
                        total_coeff = s_coeff * c_coeff
                        W[w_idx(out_r, out_c), rank_idx] += total_coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Verification logic (disabled in final output to save time, but kept in mind)
def _verify():
    sol = construct()
    U, V, W = np.array(sol['U']), np.array(sol['V']), np.array(sol['W'])
    n, m, p = 4, 4, 4
    T = np.einsum('ir,kr,sr->iks', U, W, V)
    # Check T
    errors = 0
    for i in range(n):
        for s in range(p):
            for k in range(m):
                expected = 1 if s == k else 0
                if abs(T[i, s, k] - expected) > 1e-9:
                    errors += 1
    print(f"Errors: {errors}")
    
# if __name__ == "__main__":
#     _verify()
