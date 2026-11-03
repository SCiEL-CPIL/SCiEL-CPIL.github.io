import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # We will use a block decomposition approach.
    # Split B (3x5) into B1 (3x2) and B2 (3x3).
    # Split A (3x3) into blocks:
    # A_TL (2x2), A_TR (2x1), A_BL (1x2), A_BR (1x1)
    # 
    # For A_TL * B1_02 (2x2 * 2x2), use Strassen (rank 7)
    # For A_TR * B1_2 (2x1 * 1x2), use direct (rank 4)
    # For A_BL * B1_02 (1x2 * 2x2), use direct (rank 4)
    # For A_BR * B1_2 (1x1 * 1x2), use direct (rank 2)
    # Total for A*B1: 7 + 4 + 4 + 2 = 17
    
    # Similarly for A*B2: 17
    # Total rank: 34
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9x34
    V = [[0] * rank for _ in range(m * p)]  # 15x34
    W = [[0] * rank for _ in range(p * n)]  # 15x34
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # Given matrices A (2x2), B (2x2), C (2x2)
    # We need to map the 7 Strassen terms to our U, V, W indices
    # A is indexed by (i, j) where i in {0,1}, j in {0,1} -> U index = i*m + j
    # B is indexed by (j, k) where j in {0,1}, k in {0,1} -> V index = j*p + k
    # C is indexed by (k, i) where k in {0,1}, i in {0,1} -> W index = k*n + i
    
    def strassen_2x2x2(a_base, b_base, c_base, offset):
        # a_base: base index for A block (2x2) in U
        # b_base: base index for B block (2x2) in V
        # c_base: base index for C block (2x2) in W
        # offset: column offset
        
        # Strassen's 7 products:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each Strassen term, we need to create rank-1 terms
        # that represent the product of sums/differences
        
        # M1: (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This requires 4 rank-1 terms
        add_term(a_base + 0, b_base + 0, c_base + 0, 1, 1, 1)  # A00*B00 -> C00
        add_term(a_base + 0, b_base + 1, c_base + 1, 1, 1, 1)  # A00*B11 -> C01
        add_term(a_base + 2, b_base + 0, c_base + 2, 1, 1, 1)  # A11*B00 -> C10
        add_term(a_base + 2, b_base + 1, c_base + 3, 1, 1, 1)  # A11*B11 -> C11
        
        # Wait, this is not Strassen. Strassen combines these.
        # Let me implement Strassen properly with the 7 terms.
        
        # Actually, let me just use direct multiplication for 2x2x2 (rank 4)
        # since implementing Strassen correctly with the indexing is complex.
        # But that would give rank 4+4+4+2 = 14 per block, total 28.
        # With Strassen it should be 7+4+4+2 = 17 per block, total 34.
        
        # Let me implement Strassen properly.
        # The 7 Strassen terms each contribute to multiple C entries.
        # I'll create 7 rank-1 terms, but each term's W values will be 
        # combinations of the C entries.
        
        # Actually, the standard way is:
        # For each of the 7 Strassen products M1..M7, we create one rank-1 term.
        # The U and V values are the coefficients from the Strassen formulas.
        # The W values are the contributions to each C entry.
        
        # M1 = (A00 + A11)(B00 + B11)
        # Contributes to: C00 (+1), C11 (-1)
        # U: A00=1, A11=1 -> but we need separate terms for each A entry
        # This approach doesn't work directly. Let me use a different method.
        
        # I'll just use direct multiplication for all blocks.
        # 2x2x2: rank 4
        # 2x1x2: rank 4 (2*1*2)
        # 1x2x2: rank 4 (1*2*2)
        # 1x1x2: rank 2 (1*1*2)
        # Total per block: 4+4+4+2 = 14
        # Total: 28
        
        # Let me reconsider. The current program has rank 45.
        # With direct block multiplication:
        # A_TL (2x2) * B1_02 (2x2): 4 terms
        # A_TR (2x1) * B1_2 (1x2): 4 terms
        # A_BL (1x2) * B1_02 (2x2): 4 terms
        # A_BR (1x1) * B1_2 (1x2): 2 terms
        # Total for B1: 14
        # Total for B2: 14
        # Total: 28
        
        # This is better than 45. Let me implement this.
        
        # Direct 2x2x2: 4 terms
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    add_term(a_base + i * 2 + j, b_base + j * 2 + k, c_base + k * 2 + i, 1, 1, 1)
        
    # Reset col
    col = 0
    
    # Block 1: A * B1 (3x3 * 3x2)
    # A_TL: rows 0-1, cols 0-1 -> U indices 0,1,2,3
    # A_TR: rows 0-1, col 2 -> U indices 0,1,2,3? No.
    # U index = i*m + j = i*3 + j
    # A_TL: (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
    # A_TR: (0,2)=2, (1,2)=5
    # A_BL: (2,0)=6, (2,1)=7
    # A_BR: (2,2)=8
    
    # B1 is 3x2: V index = j*5 + k
    # B1_02: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # B1_2: (2,0)=10, (2,1)=11
    
    # W index = k*3 + i
    # For B1 (k in 0,1):
    # C00: (0,0)=0, C01: (0,1)=1
    # C10: (1,0)=3, C11: (1,1)=4
    # C20: (2,0)=6, C21: (2,1)=7
    
    # A_TL * B1_02: (2x2) * (2x2) -> C00, C01, C10, C11
    for i in range(2):
        for j in range(2):
            for k in range(2):
                add_term(i * 3 + j, j * 5 + k, k * 3 + i, 1, 1, 1)
    
    # A_TR * B1_2: (2x1) * (1x2) -> C00, C01, C10, C11
    # A_TR: (0,2)=2, (1,2)=5
    # B1_2: (2,0)=10, (2,1)=11
    for i in range(2):
        for k in range(2):
            add_term(i * 3 + 2, 2 * 5 + k, k * 3 + i, 1, 1, 1)
    
    # A_BL * B1_02: (1x2) * (2x2) -> C20, C21
    # A_BL: (2,0)=6, (2,1)=7
    # B1_02: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    for j in range(2):
        for k in range(2):
            add_term(2 * 3 + j, j * 5 + k, k * 3 + 2, 1, 1, 1)
    
    # A_BR * B1_2: (1x1) * (1x2) -> C20, C21
    # A_BR: (2,2)=8
    # B1_2: (2,0)=10, (2,1)=11
    for k in range(2):
        add_term(8, 2 * 5 + k, k * 3 + 2, 1, 1, 1)
    
    # Block 2: A * B2 (3x3 * 3x3)
    # B2 is 3x3: V index = j*5 + k + 2
    # B2: (0,2)=2, (0,3)=3, (0,4)=4, (1,2)=7, (1,3)=8, (1,4)=9, (2,2)=12, (2,3)=13, (2,4)=14
    
    # A_TL * B2_02: (2x2) * (2x2) -> C02, C03, C12, C13
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term(i * 3 + j, j * 5 + k, k * 3 + i, 1, 1, 1)
    
    # A_TR * B2_2: (2x1) * (1x3) -> C02, C03, C04, C12, C13, C14
    for i in range(2):
        for k in range(2, 5):
            add_term(i * 3 + 2, 2 * 5 + k, k * 3 + i, 1, 1, 1)
    
    # A_BL * B2_02: (1x2) * (2x3) -> C22, C23, C24
    for j in range(2):
        for k in range(2, 5):
            add_term(2 * 3 + j, j * 5 + k, k * 3 + 2, 1, 1, 1)
    
    # A_BR * B2_2: (1x1) * (1x3) -> C22, C23, C24
    for k in range(2, 5):
        add_term(8, 2 * 5 + k, k * 3 + 2, 1, 1, 1)
    
    return {"U": U, "V": V, "W": W}
