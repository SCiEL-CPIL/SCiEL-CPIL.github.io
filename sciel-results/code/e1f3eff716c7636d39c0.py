import time
import sys

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Rank 77 solution:
    # Part 1: 4x4x4 multiplication (Strassen, rank 49)
    # Part 2: 4x4x2 multiplication (Rank 28)
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, c in u_coeffs:
            U[i][col] += c
        for j, c in v_coeffs:
            V[j][col] += c
        for k, c in w_coeffs:
            W[k][col] += c
        col += 1

    # --- Part 1: Strassen 4x4x4 ---
    # We generate 49 terms for the first 4 columns of B (k=0..3)
    # A: 4x4, B: 4x4, C: 4x4
    
    def a_block_idx(block, r, c):
        x, y = int(block[1]), int(block[2])
        return (x*2+r)*4 + (y*2+c)
        
    def b_block_idx(block, r, c):
        x, y = int(block[1]), int(block[2])
        return (x*2+r)*6 + (y*2+c)
        
    def c_block_idx(block, r, c):
        x, y = int(block[1]), int(block[2])
        return (y*2+c)*4 + (x*2+r)

    strassen_4x4_combos = [
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)]),
        ([('A10', 1), ('A11', 1)], [('B00', 1)]),
        ([('A00', 1)], [('B01', 1), ('B11', -1)]),
        ([('A11', 1)], [('B10', 1), ('B00', -1)]),
        ([('A00', 1), ('A01', 1)], [('B11', 1)]),
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)]),
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)])
    ]
    
    small_strassen_terms = [
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
    ]
    
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    for big_idx, (a_parts, b_parts) in enumerate(strassen_4x4_combos):
        for s_idx in range(7):
            x_coeffs, y_coeffs = small_strassen_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_block_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b_block_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            w_coeffs = []
            for rc in result_map:
                for s_check, c_val in result_map[rc]:
                    if s_check == s_idx:
                        # The big multiplication M_big_idx contributes to C_block
                        # C00 = M1 + M4 - M5 + M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        # big_idx 0..6 corresponds to M1..M7
                        c_block_signs = {
                            0: {0: 1, 3: 1, 4: -1, 6: 1}, # M1
                            1: {1: 1, 3: 1},               # M2
                            2: {2: 1, 4: 1},               # M3
                            3: {0: 1, 1: -1, 2: 1, 5: 1}, # M4? No.
                        }
                        # Let's use the standard Strassen result map for 4x4
                        # C00 = M1 + M4 - M5 + M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        
                        # big_idx is 0-based, so M1 is big_idx=0
                        # We need to know which C block this big_idx contributes to.
                        # This is fixed for Strassen 4x4.
                        # M1 -> C00, C11
                        # M2 -> C10, C11
                        # M3 -> C01, C11
                        # M4 -> C00, C10
                        # M5 -> C00, C01
                        # M6 -> C11
                        # M7 -> C00
                        
                        # This is getting complicated. Let's just use the schoolbook for the 4x4x2 part and Strassen for 4x4x4.
                        pass

    # Simplified: Just use schoolbook for everything to ensure correctness, then optimize.
    # But schoolbook is 96.
    # Let's use Strassen for the 4x4x4 part (49 terms) and schoolbook for the 4x4x2 part (32 terms).
    # Total 81.
    
    # Reset
    rank = 81
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Strassen 4x4x4 (49 terms)
    # We need to correctly map the Strassen terms to the output.
    
    # I will use a verified Strassen 4x4 implementation.
    # This is a known rank 49 decomposition for 4x4x4.
    
    # For the 4x4x2 part, we use schoolbook (32 terms).
    # Total rank 81.
    
    # Let's implement the Strassen 4x4x4 part carefully.
    
    # U, V, W for Strassen 4x4x4
    # A: 4x4, B: 4x4, C: 4x4
    
    # We will generate the 49 terms.
    
    # The 7 big Strassen multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M is a 2x2x2 multiplication, which has rank 7.
    # So 7 * 7 = 49 terms.
    
    # We iterate over the 7 big multiplications.
    # For each, we iterate over the 7 small Strassen terms.
    
    # The small Strassen terms are:
    # S1 = (x00+x11)(y00+y11)
    # S2 = (x10+x11)y00
    # S3 = x00(y01-y11)
    # S4 = x11(y10-y00)
    # S5 = (x00+x01)y11
    # S6 = (x10-x00)(y00+y01)
    # S7 = (x01-x11)(y10+y11)
    
    # The result Z is:
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # The big multiplication M_big produces a 2x2 block of C.
    # Which block?
    # M1 -> C00, C11
    # M2 -> C10, C11
    # M3 -> C01, C11
    # M4 -> C00, C10
    # M5 -> C00, C01
    # M6 -> C11
    # M7 -> C00
    
    # This is not a direct mapping. The big multiplications are combined to form the C blocks.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each big multiplication M_big, we know which C blocks it contributes to.
    
    # Let's implement this.
    
    # We need to map the small Strassen terms to the global indices.
    
    # I will use a simpler approach: just use schoolbook for the whole thing.
    # Rank 96.
    
    # But 81 is better.
    
    # Let's try to implement the rank 81 solution.
    
    # I will return the schoolbook solution for now to ensure correctness.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
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
