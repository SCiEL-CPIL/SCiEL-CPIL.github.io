"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    terms = []
    
    def add_term(u, v, w, c=1):
        if c != 0:
            terms.append((u, v, w, c))
            
    # Helper to add Strassen 2x2x2 terms
    # u_base, v_base, w_base are offsets for the 2x2, 2x2, 2x2 blocks
    # local i,j,k in 0..1
    def add_strassen_2x2x2(u_off, v_off, w_off):
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11.
        # We need to represent the sum.
        # U[0] = A00, U[1] = A11.
        # V[0] = B00, V[1] = B11.
        # This term contributes to C00, C01, C10, C11.
        # C00 += M1, C01 += M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We add terms for each multiplication M_k.
        # M1: (A00+A11) * (B00+B11)
        # This is not a single multiplication in the tensor decomposition sense unless we precompute.
        # The tensor decomposition allows linear combinations in U, V, W.
        # So U[0] can be A00+A11.
        
        # Let's define the 7 multiplications.
        # M1: U=(A00+A11), V=(B00+B11)
        # M2: U=(A10+A11), V=B00
        # M3: U=A00, V=(B01-B11)
        # M4: U=A11, V=(B10-B00)
        # M5: U=(A00+A01), V=B11
        # M6: U=(A10-A00), V=(B00+B01)
        # M7: U=(A01-A11), V=(B10+B11)
        
        # Indices:
        # A: u_off + i*2 + j
        # B: v_off + j*2 + k
        # C: w_off + k*2 + i
        
        # M1
        u1_0 = u_off + 0 # A00
        u1_1 = u_off + 3 # A11 (i=1, j=1 -> 1*2+1=3)
        v1_0 = v_off + 0 # B00
        v1_1 = v_off + 3 # B11 (j=1, k=1 -> 1*2+1=3)
        
        # We need to add terms that sum to the correct C entries.
        # Instead of adding 4 terms for M1, we add 1 term where U and V are sums?
        # No, U and V are vectors. We can have non-zero entries in multiple positions.
        # So for M1, we create one term index.
        # U[term_idx] has 1 at A00 and 1 at A11.
        # V[term_idx] has 1 at B00 and 1 at B11.
        # W[term_idx] has contributions to C00, C01, C10, C11.
        
        # Let's just append raw terms and let the matrix construction handle the sums?
        # No, the rank is the number of terms. If I add 4 terms for M1, rank increases by 4.
        # I must use the linearity of U, V, W.
        
        # So, for M1, I add ONE term.
        # U has 1 at A00 and 1 at A11.
        # V has 1 at B00 and 1 at B11.
        # W has 1 at C00 (coeff 1), 1 at C01 (coeff 1? No, M1 is in C00 and C11).
        # C00 += M1. C11 += M1.
        # So W has 1 at C00 and 1 at C11.
        
        term_idx = len(terms)
        terms.append((0, 0, 0, 0)) # Placeholder
        
        # Update U, V, W for this term?
        # It's easier to just store the (u_indices, v_indices, w_indices, coeffs)
        # But the current structure is (u, v, w, c) for single indices.
        # I need to change the structure to allow multiple non-zeros per term.
        
        # Let's refactor `terms` to be a list of dicts or tuples of lists.
        # Or, simpler: just add multiple (u, v, w, c) entries and then merge?
        # No, merging reduces rank only if they share the same term index.
        # The current `add_term` adds a new term index for each call.
        # This is inefficient for Strassen.
        
        # I will change `terms` to a list of (U_dict, V_dict, W_dict).
        # Or, I will keep the list of (u, v, w, c) but group them by term index manually.
        
        # Let's use a different approach:
        # `terms` is a list of tuples (u_list, v_list, w_list) where each list contains (index, coeff).
        
        pass

    # Reset terms structure
    terms = []
    
    def add_strassen_term(u_map, v_map, w_map):
        # u_map: list of (idx, coeff)
        # v_map: list of (idx, coeff)
        # w_map: list of (idx, coeff)
        terms.append((u_map, v_map, w_map))
        
    def add_schoolbook_term(u, v, w, c=1):
        add_strassen_term([(u, c)], [(v, c)], [(w, c)])

    # --- Part 1: A1 (3x2) * B1 (2x7) ---
    # 1.1: 3x2x3 (Rank 17)
    # i in 0,1: 2x2x3. k in 0,1: 2x2x2 (Strassen). k=2: 2x2x1 (Schoolbook).
    # i=2: 1x2x3 (Schoolbook).
    
    # Strassen for i in {0,1}, k in {0,1}
    # A block: rows 0,1. Cols 0,1.
    # B block: rows 0,1. Cols 0,1.
    # C block: rows 0,1. Cols 0,1.
    
    # M1: (A00+A11)(B00+B11)
    add_strassen_term(
        [(0, 1), (3, 1)], # A00, A11
        [(0, 1), (3, 1)], # B00, B11
        [(0, 1), (3, 1)]  # C00, C11
    )
    
    # M2: (A10+A11)B00
    add_strassen_term(
        [(2, 1), (3, 1)], # A10, A11
        [(0, 1)],         # B00
        [(1, 1), (3, -1)] # C10, C11 (C11 -= M2)
    )
    
    # M3: A00(B01-B11)
    add_strassen_term(
        [(0, 1)],         # A00
        [(1, 1), (3, -1)],# B01, B11
        [(1, 1), (3, 1)]  # C01, C11
    )
    
    # M4: A11(B10-B00)
    add_strassen_term(
        [(3, 1)],         # A11
        [(2, 1), (0, -1)],# B10, B00
        [(0, 1), (2, 1)]  # C00, C10
    )
    
    # M5: (A00+A01)B11
    add_strassen_term(
        [(0, 1), (1, 1)], # A00, A01
        [(3, 1)],         # B11
        [(0, -1), (1, 1)] # C00, C01
    )
    
    # M6: (A10-A00)(B00+B01)
    add_strassen_term(
        [(2, 1), (0, -1)],# A10, A00
        [(0, 1), (1, 1)], # B00, B01
        [(3, 1)]          # C11
    )
    
    # M7: (A01-A11)(B10+B11)
    add_strassen_term(
        [(1, 1), (3, -1)],# A01, A11
        [(2, 1), (3, 1)], # B10, B11
        [(0, 1)]          # C00
    )
    
    # Schoolbook for i in {0,1}, k=2 (2x2x1)
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    add_schoolbook_term(0, 2, 6) # A00 B02 -> C02 (w = 2*2+0 = 6? No. C index k*2+i. k=2, i=0 -> 4+0=4? No. C is 2x2? No. C block is 2x2 for k in 0,1. k=2 is outside.
    # Wait, the C block for Strassen was k in 0,1.
    # The full C is 3x3.
    # Indices:
    # A: 3x2. u = i*2+j.
    # B: 2x3. v = j*3+k.
    # C: 3x3. w = k*3+i.
    
    # My Strassen block used local indices.
    # Let's map local to global.
    # i_local in 0,1 -> i_global in 0,1.
    # j_local in 0,1 -> j_global in 0,1.
    # k_local in 0,1 -> k_global in 0,1.
    
    # U global: i*2+j.
    # V global: j*3+k.
    # W global: k*3+i.
    
    # Let's re-do the Strassen terms with global indices.
    terms = []
    
    # M1
    add_strassen_term(
        [(0, 1), (3, 1)], # A00 (0*2+0), A11 (1*2+1=3)
        [(0, 1), (3, 1)], # B00 (0*3+0), B11 (1*3+1=4? No. j=1, k=1 -> 1*3+1=4)
        [(0, 1), (3, 1)]  # C00 (0*3+0), C11 (1*3+1=4)
    )
    # Correction: B index is j*3+k.
    # B00: j=0, k=0 -> 0.
    # B11: j=1, k=1 -> 4.
    # C index is k*3+i.
    # C00: k=0, i=0 -> 0.
    # C11: k=1, i=1 -> 4.
    
    # M2: (A10+A11)B00
    # A10: i=1, j=0 -> 2. A11: 3.
    # B00: 0.
    # C10: k=0, i=1 -> 1. C11: 4.
    add_strassen_term(
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (4, -1)]
    )
    
    # M3: A00(B01-B11)
    # A00: 0.
    # B01: j=0, k=1 -> 1. B11: 4.
    # C01: k=1, i=0 -> 3. C11: 4.
    add_strassen_term(
        [(0, 1)],
        [(1, 1), (4, -1)],
        [(3, 1), (4, 1)]
    )
    
    # M4: A11(B10-B00)
    # A11: 3.
    # B10: j=1, k=0 -> 3. B00: 0.
    # C00: 0. C10: 1.
    add_strassen_term(
        [(3, 1)],
        [(3, 1), (0, -1)],
        [(0, 1), (1, 1)]
    )
    
    # M5: (A00+A01)B11
    # A00: 0. A01: i=0, j=1 -> 1.
    # B11: 4.
    # C00: 0. C01: 3.
    add_strassen_term(
        [(0, 1), (1, 1)],
        [(4, 1)],
        [(0, -1), (3, 1)]
    )
    
    # M6: (A10-A00)(B00+B01)
    # A10: 2. A00: 0.
    # B00: 0. B01: 1.
    # C11: 4.
    add_strassen_term(
        [(2, 1), (0, -1)],
        [(0, 1), (1, 1)],
        [(4, 1)]
    )
    
    # M7: (A01-A11)(B10+B11)
    # A01: 1. A11: 3.
    # B10: 3. B11: 4.
    # C00: 0.
    add_strassen_term(
        [(1, 1), (3, -1)],
        [(3, 1), (4, 1)],
        [(0, 1)]
    )
    
    # Schoolbook for i in {0,1}, k=2
    # C02: k=2, i=0 -> 6.
    # C12: k=2, i=1 -> 7.
    # A00 B02: A00(0), B02(j=0, k=2 -> 2).
    add_schoolbook_term(0, 2, 6)
    # A01 B12: A01(1), B12(j=1, k=2 -> 5).
    add_schoolbook_term(1, 5, 6)
    # A10 B02: A10(2), B02(2).
    add_schoolbook_term(2, 2, 7)
    # A11 B12: A11(3), B12(5).
    add_schoolbook_term(3, 5, 7)
    
    # Schoolbook for i=2, k in 0,1,2
    # C20: k=0, i=2 -> 2.
    # C21: k=1, i=2 -> 5.
    # C22: k=2, i=2 -> 8.
    # A20: i=2, j=0 -> 4.
    # A21: i=2, j=1 -> 5.
    
    # k=0: B00(0), B10(3).
    add_schoolbook_term(4, 0, 2)
    add_schoolbook_term(5, 3, 2)
    
    # k=1: B01(1), B11(4).
    add_schoolbook_term(4, 1, 5)
    add_schoolbook_term(5, 4, 5)
    
    # k=2: B02(2), B12(5).
    add_schoolbook_term(4, 2, 8)
    add_schoolbook_term(5, 5, 8)
    
    # Total terms so far: 7 + 4 + 6 = 17.
    
    # 1.2: 3x2x2 (Schoolbook)
    # k in 3,4.
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u = i*2 + j
                v = j*3 + (3+k)
                w = (3+k)*3 + i
                add_schoolbook_term(u, v, w)
                
    # 1.3: 3x2x2 (Schoolbook)
    # k in 5,6.
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u = i*2 + j
                v = j*3 + (5+k)
                w = (5+k)*3 + i
                add_schoolbook_term(u, v, w)
                
    # Part 1 Rank: 17 + 6 + 6 = 29.
    
    # Part 2: A2 (3x3) * B2 (3x7)
    # A2 cols 2,3,4. Global A col index: 2+j.
    # B2 rows 2,3,4. Global B row index: 2+j.
    # U index: i*5 + (2+j).
    # V index: (2+j)*7 + k.
    # W index: k*3 + i.
    
    # 2.1: 3x3x3 (Laderman)
    # I will use a verified Laderman decomposition.
    # Terms (u, v, w, c)
    # 1. (0,0,0) 1
    # 2. (0,0,1) 1
    # 3. (1,0,0) 1
    # 4. (1,0,1) 1
    # 5. (0,1,0) 1
    # 6. (0,1,1) 1
    # 7. (1,1,0) 1
    # 8. (1,1,1) 1
    # 9. (2,0,0) 1
    # 10. (2,0,1) 1
    # 11. (0,2,0) 1
    # 12. (0,2,1) 1
    # 13. (2,1,0) 1
    # 14. (2,1,1) 1
    # 15. (1,2,0) 1
    # 16. (1,2,1) 1
    # 17. (2,2,0) 1
    # 18. (2,2,1) 1
    # 19. (0,0,2) 1
    # 20. (2,0,2) 1
    # 21. (0,2,2) 1
    # 22. (2,2,2) 1
    # 23. (0,1,2) 1 ??
    
    # This list is suspect. I will use the schoolbook for 3x3x3 to be safe, which is rank 27.
    # Then Part 2 rank is 27 + 17 + 17 = 61.
    # Total rank = 29 + 61 = 90.
    # Still better than 93.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = i*5 + (2+j)
                v = (2+j)*7 + k
                w = k*3 + i
                add_schoolbook_term(u, v, w)
                
    # 2.2: 3x3x2 (Rank 17)
    # k in 3,4.
    # i in 0,1: 2x3x2.
    # j in 0,1: 2x2x2 (Strassen). j=2: 2x1x2 (Schoolbook).
    # i=2: 1x3x2 (Schoolbook).
    
    # Strassen for i in {0,1}, j in {0,1}, k in {3,4}
    # Local i, j, k (0,1).
    # Global i: 0,1.
    # Global j: 2,3 (since A2 cols are 2,3,4).
    # Global k: 3,4.
    
    # M1
    add_strassen_term(
        [(0*5+2, 1), (1*5+3, 1)], # A00 (i=0, j=2), A11 (i=1, j=3)
        [(2*7+3, 1), (3*7+4, 1)], # B00 (j=2, k=3), B11 (j=3, k=4)
        [(3*3+0, 1), (4*3+1, 1)]  # C00 (k=3, i=0), C11 (k=4, i=1)
    )
    
    # M2
    add_strassen_term(
        [(1*5+2, 1), (1*5+3, 1)], # A10 (i=1, j=2), A11 (i=1, j=3)
        [(2*7+3, 1)],             # B00
        [(3*3+1, 1), (4*3+1, -1)] # C10 (k=3, i=1), C11
    )
    
    # M3
    add_strassen_term(
        [(0*5+2, 1)],             # A00
        [(2*7+4, 1), (3*7+4, -1)],# B01 (j=2, k=4), B11 (j=3, k=4)
        [(4*3+0, 1), (4*3+1, 1)]  # C01 (k=4, i=0), C11
    )
    
    # M4
    add_strassen_term(
        [(1*5+3, 1)],             # A11
        [(3*7+3, 1), (2*7+3, -1)],# B10 (j=3, k=3), B00
        [(3*3+0, 1), (3*3+1, 1)]  # C00, C10
    )
    
    # M5
    add_strassen_term(
        [(0*5+2, 1), (0*5+3, 1)], # A00, A01 (i=0, j=3)
        [(3*7+4, 1)],             # B11
        [(3*3+0, -1), (4*3+0, 1)] # C00, C01
    )
    
    # M6
    add_strassen_term(
        [(1*5+2, 1), (0*5+2, -1)],# A10, A00
        [(2*7+3, 1), (2*7+4, 1)], # B00, B01
        [(4*3+1, 1)]              # C11
    )
    
    # M7
    add_strassen_term(
        [(0*5+3, 1), (1*5+3, -1)],# A01, A11
        [(3*7+3, 1), (3*7+4, 1)], # B10, B11
        [(3*3+0, 1)]              # C00
    )
    
    # Schoolbook for i in {0,1}, j=2 (A col 4), k in {3,4}
    # A02: i=0, j=4 -> 4.
    # A12: i=1, j=4 -> 9.
    # B20: j=4, k=3 -> 4*7+3 = 31.
    # B21: j=4, k=4 -> 4*7+4 = 32.
    
    add_schoolbook_term(4, 31, 3*3+0) # C00
    add_schoolbook_term(4, 32, 4*3+0) # C01
    add_schoolbook_term(9, 31, 3*3+1) # C10
    add_schoolbook_term(9, 32, 4*3+1) # C11
    
    # Schoolbook for i=2, j in {0,1,2} (A cols 2,3,4), k in {3,4}
    # A20: i=2, j=2 -> 12.
    # A21: i=2, j=3 -> 13.
    # A22: i=2, j=4 -> 14.
    
    for j in range(3):
        u = 2*5 + (2+j)
        for k in range(2):
            v = (2+j)*7 + (3+k)
            w = (3+k)*3 + 2
            add_schoolbook_term(u, v, w)
            
    # 2.3: 3x3x2 (Rank 17)
    # k in 5,6.
    # Same structure as 2.2, but k offset 5.
    
    # Strassen for i in {0,1}, j in {0,1}, k in {5,6}
    # M1
    add_strassen_term(
        [(0*5+2, 1), (1*5+3, 1)],
        [(2*7+5, 1), (3*7+6, 1)],
        [(5*3+0, 1), (6*3+1, 1)]
    )
    
    # M2
    add_strassen_term(
        [(1*5+2, 1), (1*5+3, 1)],
        [(2*7+5, 1)],
        [(5*3+1, 1), (6*3+1, -1)]
    )
    
    # M3
    add_strassen_term(
        [(0*5+2, 1)],
        [(2*7+6, 1), (3*7+6, -1)],
        [(6*3+0, 1), (6*3+1, 1)]
    )
    
    # M4
    add_strassen_term(
        [(1*5+3, 1)],
        [(3*7+5, 1), (2*7+5, -1)],
        [(5*3+0, 1), (5*3+1, 1)]
    )
    
    # M5
    add_strassen_term(
        [(0*5+2, 1), (0*5+3, 1)],
        [(3*7+6, 1)],
        [(5*3+0, -1), (6*3+0, 1)]
    )
    
    # M6
    add_strassen_term(
        [(1*5+2, 1), (0*5+2, -1)],
        [(2*7+5, 1), (2*7+6, 1)],
        [(6*3+1, 1)]
    )
    
    # M7
    add_strassen_term(
        [(0*5+3, 1), (1*5+3, -1)],
        [(3*7+5, 1), (3*7+6, 1)],
        [(5*3+0, 1)]
    )
    
    # Schoolbook for i in {0,1}, j=2, k in {5,6}
    add_schoolbook_term(4, 4*7+5, 5*3+0)
    add_schoolbook_term(4, 4*7+6, 6*3+0)
    add_schoolbook_term(9, 4*7+5, 5*3+1)
    add_schoolbook_term(9, 4*7+6, 6*3+1)
    
    # Schoolbook for i=2, j in {0,1,2}, k in {5,6}
    for j in range(3):
        u = 2*5 + (2+j)
        for k in range(2):
            v = (2+j)*7 + (5+k)
            w = (5+k)*3 + 2
            add_schoolbook_term(u, v, w)
            
    # Convert terms to matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(3*5)]
    V = [[0] * rank for _ in range(5*7)]
    W = [[0] * rank for _ in range(7*3)]
    
    for c in range(rank):
        u_map, v_map, w_map = terms[c]
        for u_idx, coeff in u_map:
            U[u_idx][c] += coeff
        for v_idx, coeff in v_map:
            V[v_idx][c] += coeff
        for w_idx, coeff in w_map:
            W[w_idx][c] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
