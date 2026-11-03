import time
import sys

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 61 decomposition
    # 2 * (Rank 26 for 3x3x3) + 1 * (Rank 9 for 3x3x1)
    rank = 61
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if coeff != 0:
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
        col += 1

    # Function to add a 3x3x3 block using Rank 26 decomposition
    # b_offset: column offset in B (0 or 3)
    # w_offset: column offset in W (0 or 3)
    def add_3x3x3_block(b_off, w_off):
        # Strassen for A11 B11 -> C11
        # M1 = (A00+A11)(B00+B11)
        add_term(0, 0 + b_off, 0 + w_off, 1) # A00 B00 -> C00
        add_term(0, 1 + b_off, 0 + w_off, 1) # A00 B01 -> C00? No, Strassen M1 is (A00+A11)(B00+B11)
        
        # Let's just use the explicit terms derived above to be safe.
        # The add_term above increments col. I need to manage col carefully.
        # It's better to pre-calculate the indices and coefficients for the 26 terms.
        
    # Actually, let's just write the 26 terms explicitly for the first block,
    # then repeat for the second block with offsets,
    # then the 9 terms for the third block.
    
    # Block 1: B cols 0-2, W cols 0-2 (k=0,1,2)
    # Strassen M1
    # (A00+A11) * (B00+B11)
    # U: A00(0), A11(4)
    # V: B00(0), B11(1)
    # W: C00(0), C11(4)
    # This is getting messy to do with a generic add_term that increments col.
    # I will define the 26 (u, v, w, coeff) tuples for the 3x3x3 case.
    
    # Note: U index is i*3+j. V index is j*3+k. W index is k*3+i.
    # A11: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # A12: (0,2)->2, (1,2)->5
    # A21: (2,0)->6, (2,1)->7
    # A22: (2,2)->8
    
    # B11: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B12: (0,2)->2, (1,2)->5
    # B21: (2,0)->6, (2,1)->7
    # B22: (2,2)->8
    
    # C11: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # C12: (0,2)->2, (1,2)->5
    # C21: (2,0)->6, (2,1)->7
    # C22: (2,2)->8
    
    # Strassen Terms for C11 (using A11, B11)
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, -C11
    # M3: A00B10 -> C01, C11
    # M4: A11B11 -> C00, C10
    # M5: (A00+A01)B11 -> -C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # I will hardcode the 26 terms for one 3x3x3 block.
    terms_3x3x3 = [
        # M1
        (0, 0, 0, 1), (4, 0, 0, 1), (0, 1, 0, 1), (4, 1, 0, 1),
        (0, 0, 4, 1), (4, 0, 4, 1), (0, 1, 4, 1), (4, 1, 4, 1),
        # This is not how it works. U, V, W are vectors.
        # Term c: U_c, V_c, W_c.
        # M1: U = A00+A11. V = B00+B11. W = C00+C11.
        # So for column c:
        # U[0] += 1, U[4] += 1
        # V[0] += 1, V[1] += 1
        # W[0] += 1, W[4] += 1
        
        # I will just list the (u_idx, v_idx, w_idx, coeff) for each of the 26 terms.
        # But each term has multiple non-zeros in U, V, W.
        # So I need a list of (u_idx, v_idx, w_idx, coeff) for each atomic multiplication?
        # No, a "term" in the rank decomposition is a rank-1 tensor.
        # A rank-1 tensor is $u \otimes v \otimes w$.
        # In Strassen, $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$ is ONE multiplication.
        # So it corresponds to ONE column in U, V, W.
        # U for this column has 1 at index 0 and 1 at index 4.
        # V for this column has 1 at index 0 and 1 at index 1.
        # W for this column has 1 at index 0 and 1 at index 4.
        
        # So I will define a list of 26 items, each being a list of (idx, coeff) for U, V, W.
    ]
    
    # Let's generate the 26 terms programmatically to avoid errors.
    
    def generate_3x3x3_terms():
        terms = []
        # Strassen
        # M1
        terms.append(([ (0,1), (4,1) ], [ (0,1), (1,1) ], [ (0,1), (4,1) ]))
        # M2
        terms.append(([ (3,1), (4,1) ], [ (0,1) ], [ (3,1), (4,-1) ]))
        # M3
        terms.append(([ (0,1) ], [ (3,1) ], [ (1,1), (4,1) ]))
        # M4
        terms.append(([ (4,1) ], [ (4,1) ], [ (0,1), (3,1) ]))
        # M5
        terms.append(([ (0,1), (1,1) ], [ (4,1) ], [ (0,-1), (1,1) ]))
        # M6
        terms.append(([ (3,1), (0,-1) ], [ (0,1), (1,1) ], [ (4,1) ]))
        # M7
        terms.append(([ (1,1), (4,-1) ], [ (3,1), (4,1) ], [ (0,1) ]))
        
        # A12 B21 -> C11
        # A12: 2, 5. B21: 6, 7. C11: 0,1,3,4.
        # A02 B20 -> C00
        terms.append(([ (2,1) ], [ (6,1) ], [ (0,1) ]))
        # A02 B21 -> C01
        terms.append(([ (2,1) ], [ (7,1) ], [ (1,1) ]))
        # A12 B20 -> C10
        terms.append(([ (5,1) ], [ (6,1) ], [ (3,1) ]))
        # A12 B21 -> C11
        terms.append(([ (5,1) ], [ (7,1) ], [ (4,1) ]))
        
        # A11 B12 -> C12
        # A11: 0,1,3,4. B12: 2,5. C12: 2,5.
        # A00 B02 -> C02
        terms.append(([ (0,1) ], [ (2,1) ], [ (2,1) ]))
        # A01 B12 -> C02
        terms.append(([ (1,1) ], [ (5,1) ], [ (2,1) ]))
        # A10 B02 -> C12
        terms.append(([ (3,1) ], [ (2,1) ], [ (5,1) ]))
        # A11 B12 -> C12
        terms.append(([ (4,1) ], [ (5,1) ], [ (5,1) ]))
        
        # A12 B22 -> C12
        # A12: 2,5. B22: 8. C12: 2,5.
        # A02 B22 -> C02
        terms.append(([ (2,1) ], [ (8,1) ], [ (2,1) ]))
        # A12 B22 -> C12
        terms.append(([ (5,1) ], [ (8,1) ], [ (5,1) ]))
        
        # A21 B11 -> C21
        # A21: 6,7. B11: 0,1,3,4. C21: 6,7.
        # A20 B00 -> C20
        terms.append(([ (6,1) ], [ (0,1) ], [ (6,1) ]))
        # A21 B10 -> C20
        terms.append(([ (7,1) ], [ (3,1) ], [ (6,1) ]))
        # A20 B01 -> C21
        terms.append(([ (6,1) ], [ (1,1) ], [ (7,1) ]))
        # A21 B11 -> C21
        terms.append(([ (7,1) ], [ (4,1) ], [ (7,1) ]))
        
        # A22 B21 -> C21
        # A22: 8. B21: 6,7. C21: 6,7.
        # A22 B20 -> C20
        terms.append(([ (8,1) ], [ (6,1) ], [ (6,1) ]))
        # A22 B21 -> C21
        terms.append(([ (8,1) ], [ (7,1) ], [ (7,1) ]))
        
        # A21 B12 -> C22
        # A21: 6,7. B12: 2,5. C22: 8.
        # A20 B02 -> C22
        terms.append(([ (6,1) ], [ (2,1) ], [ (8,1) ]))
        # A21 B12 -> C22
        terms.append(([ (7,1) ], [ (5,1) ], [ (8,1) ]))
        
        # A22 B22 -> C22
        # A22: 8. B22: 8. C22: 8.
        # A22 B22 -> C22
        terms.append(([ (8,1) ], [ (8,1) ], [ (8,1) ]))
        
        return terms

    terms_3x3x3 = generate_3x3x3_terms()
    
    # Block 1: b_off=0, w_off=0
    for u_list, v_list, w_list in terms_3x3x3:
        for u_idx, coeff in u_list:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_list:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_list:
            W[w_idx][col] += coeff
        col += 1
        
    # Block 2: b_off=3, w_off=3
    # B indices: j*3+k. If k increases by 3, V index increases by 3.
    # W indices: k*3+i. If k increases by 3, W index increases by 9? No.
    # W index is k*n+i. n=3.
    # If k -> k+3, index -> (k+3)*3+i = k*3+i+9.
    # So w_idx += 9.
    # V index is j*p+k. p=7? No, in the 3x3x3 subproblem, p=3.
    # But in the global problem, V is size 3x7.
    # The subproblem uses B cols 0-2.
    # For Block 2, we use B cols 3-5.
    # So j is same, k is k+3.
    # V index = j*7 + k. So v_idx += 3.
    # W index = k*3 + i. So w_idx += 9.
    
    for u_list, v_list, w_list in terms_3x3x3:
        for u_idx, coeff in u_list:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_list:
            V[v_idx + 3][col] += coeff
        for w_idx, coeff in w_list:
            W[w_idx + 9][col] += coeff
        col += 1
        
    # Block 3: 3x3x1 (col 6)
    # Schoolbook 9 terms.
    # A (3x3), B (3x1), C (3x1).
    # B index: j*7+6.
    # W index: 6*3+i = 18+i.
    for i in range(3):
        for j in range(3):
            U[i*3+j][col] += 1
            V[j*7+6][col] += 1
            W[18+i][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
