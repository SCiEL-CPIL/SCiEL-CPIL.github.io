# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Rank 77 decomposition using Strassen outer (7) * Improved Inner (11)
    # Inner 11 = Strassen(7) for 2x2x2 + Schoolbook(4) for 2x2x1
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen steps (7)
    # Block IDs:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11 (Note: B blocks are 2x3, so "01" means J in 0,1; K in 3,4,5? 
    # Wait, the previous code defined B blocks as 2x3.
    # B block 00: J in 0,1; K in 0,1,2
    # B block 01: J in 0,1; K in 3,4,5
    # B block 10: J in 2,3; K in 0,1,2
    # B block 11: J in 2,3; K in 3,4,5
    # This partitioning is consistent with the previous code.
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {
            'A': [(0, 1), (3, 1)], 
            'B': [(0, 1), (3, 1)], 
            'C': [(0, 1), (3, 1)]  
        },
        # M2 = (A10 + A11)B00
        {
            'A': [(2, 1), (3, 1)], 
            'B': [(0, 1)],         
            'C': [(2, 1), (3, -1)] 
        },
        # M3 = A00(B01 - B11)
        {
            'A': [(0, 1)],         
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)]  
        },
        # M4 = A11(B10 - B00)
        {
            'A': [(3, 1)],         
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)]  
        },
        # M5 = (A00 + A01)B11
        {
            'A': [(0, 1), (1, 1)], 
            'B': [(3, 1)],         
            'C': [(0, -1), (1, 1)] 
        },
        # M6 = (A10 - A00)(B00 + B01)
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)], 
            'C': [(3, 1)]          
        },
        # M7 = (A01 - A11)(B10 + B11)
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)], 
            'C': [(0, 1)]          
        }
    ]
    
    # Inner Strassen steps for 2x2x2 (k=0,1 part)
    # Local indices: A[r,s], B[s,k'], C[r,k'] with r,s,k' in {0,1}
    inner_strassen_steps = [
        # S1 = (A00 + A11)(B00 + B11)
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # S2 = (A10 + A11)B00
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        # S3 = A00(B01 - B11)
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # S4 = A11(B10 - B00)
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        # S5 = (A00 + A01)B11
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # S6 = (A10 - A00)(B00 + B01)
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # S7 = (A01 - A11)(B10 + B11)
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]}
    ]
    
    # Inner Schoolbook terms for 2x2x1 (k=2 part)
    # Local indices: A[r,s], B[s,2], C[r,2]
    # Terms:
    # T1: A00 * B02 -> C02
    # T2: A01 * B12 -> C02
    # T3: A10 * B02 -> C12
    # T4: A11 * B12 -> C12
    inner_schoolbook_terms = [
        {'A': [(0, 1)], 'B': [(0, 1)], 'C': [(0, 1)]}, # A[0,0] B[0,2] -> C[0,2]
        {'A': [(1, 1)], 'B': [(1, 1)], 'C': [(0, 1)]}, # A[0,1] B[1,2] -> C[0,2]
        {'A': [(2, 1)], 'B': [(0, 1)], 'C': [(2, 1)]}, # A[1,0] B[0,2] -> C[1,2]
        {'A': [(3, 1)], 'B': [(1, 1)], 'C': [(2, 1)]}  # A[1,1] B[1,2] -> C[1,2]
    ]
    
    # Helper to get global indices for a block given local coordinates
    # A block is 2x2. Local (r,s) -> Global (I,J)
    def get_A_global(block_id, r, s):
        if block_id == 0: return r, s
        if block_id == 1: return r, s+2
        if block_id == 2: return r+2, s
        if block_id == 3: return r+2, s+2
        return None

    # B block is 2x3. Local (s, k) -> Global (J,K)
    # Note: In inner Strassen, k is 0 or 1. In inner Schoolbook, k is 2.
    def get_B_global(block_id, s, k):
        if block_id == 0: return s, k
        if block_id == 1: return s, k+3
        if block_id == 2: return s+2, k
        if block_id == 3: return s+2, k+3
        return None

    # C block is 2x3. Local (r, k) -> Global (I,K)
    def get_C_global(block_id, r, k):
        if block_id == 0: return r, k
        if block_id == 1: return r, k+3
        if block_id == 2: return r+2, k
        if block_id == 3: return r+2, k+3
        return None

    # Iterate through outer steps
    for i, step in enumerate(steps):
        base_rank = i * 11
        
        # 1. Inner Strassen steps (7 terms)
        for j, is_step in enumerate(inner_strassen_steps):
            c_global = base_rank + j
            
            # Process A contributions
            # is_step['A'] contains (block_id, coeff) for local A
            # We need to map local A indices (r,s) to global A indices (I,J)
            # The local A indices are determined by the block_id in is_step['A']?
            # No, the local A indices are the coordinates within the 2x2 block.
            # The Strassen step defines linear combinations of the 4 local A blocks (00,01,10,11).
            # Wait, the inner Strassen steps operate on the local 2x2 A matrix.
            # So is_step['A'] refers to the local 2x2 A indices?
            # Let's clarify:
            # The outer step combines global A blocks to form a local A (2x2).
            # The inner step combines local A entries to form a scalar.
            
            # Actually, the current code structure is:
            # Outer step defines how global blocks form local A, B, C.
            # Inner step defines how local entries form the product.
            
            # Let's re-read the previous code logic.
            # Previous code:
            # for r in range(2):
            #   for s in range(2):
            #     for k in range(3):
            #       c_in = ...
            #       for block_id, coeff in step['A']:
            #          # maps local (r,s) to global via block_id
            #          U[I*4+J] += coeff
            
            # This implies `step['A']` contains (global_block_id, coeff).
            # And the local coordinates (r,s) are iterated separately.
            
            # So for the inner Strassen step:
            # We need to iterate over the local coordinates involved in the inner Strassen term.
            # Inner Strassen term j uses specific local A entries, local B entries, local C entries.
            
            # Let's define the inner Strassen steps more explicitly with local indices.
            # S1: (A00+A11)(B00+B11) -> C00, C11
            # This involves:
            # A entries: (0,0) coeff 1, (1,1) coeff 1
            # B entries: (0,0) coeff 1, (1,1) coeff 1
            # C entries: (0,0) coeff 1, (1,1) coeff 1
            
            # My `inner_strassen_steps` above used block IDs 0-3 which is confusing with outer block IDs.
            # Let's redefine inner steps with explicit local indices.
            
            pass # This block is a placeholder, we will rewrite the loop below.
        
        # 2. Inner Schoolbook terms (4 terms)
        pass

    # REWRITE LOGIC FOR CLARITY AND CORRECTNESS
    # We will discard the previous loop structure and write a clean one.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen Steps
    outer_steps = [
        # (A_blocks, B_blocks, C_blocks)
        # Each block is (id, coeff)
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        # C blocks: 00, 01, 10, 11
        [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),         # M2
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),         # M3
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),         # M4
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),         # M5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),         # M6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])          # M7
        ]
    ]
    
    # Inner Strassen Steps for 2x2x2 (k=0,1)
    # Each step is a list of (local_A_indices, local_B_indices, local_C_indices)
    # local indices are (r, s) for A, (s, k) for B, (r, k) for C
    # k in {0, 1}
    inner_strassen = [
        # S1: (A00+A11)(B00+B11)
        [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        ],
        # S2: (A10+A11)B00
        [
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        ],
        # S3: A00(B01-B11)
        [
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        ],
        # S4: A11(B10-B00)
        [
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])
        ],
        # S5: (A00+A01)B11
        [
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        ],
        # S6: (A10-A00)(B00+B01)
        [
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        ],
        # S7: (A01-A11)(B10+B11)
        [
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        ]
    ]
    
    # Inner Schoolbook Terms for 2x2x1 (k=2)
    # Each term is a list of (local_A_indices, local_B_indices, local_C_indices)
    # k=2
    inner_schoolbook = [
        # T1: A00 * B02 -> C02
        [
            ([(0,0,1)], [(0,2,1)], [(0,2,1)])
        ],
        # T2: A01 * B12 -> C02
        [
            ([(0,1,1)], [(1,2,1)], [(0,2,1)])
        ],
        # T3: A10 * B02 -> C12
        [
            ([(1,0,1)], [(0,2,1)], [(1,2,1)])
        ],
        # T4: A11 * B12 -> C12
        [
            ([(1,1,1)], [(1,2,1)], [(1,2,1)])
        ]
    ]
    
    # Helper to map local A (r,s) to global (I,J) given outer block ID
    def map_A(block_id, r, s):
        if block_id == 0: return r, s
        if block_id == 1: return r, s+2
        if block_id == 2: return r+2, s
        if block_id == 3: return r+2, s+2
        return None

    # Helper to map local B (s,k) to global (J,K) given outer block ID
    def map_B(block_id, s, k):
        if block_id == 0: return s, k
        if block_id == 1: return s, k+3
        if block_id == 2: return s+2, k
        if block_id == 3: return s+2, k+3
        return None

    # Helper to map local C (r,k) to global (I,K) given outer block ID
    def map_C(block_id, r, k):
        if block_id == 0: return r, k
        if block_id == 1: return r, k+3
        if block_id == 2: return r+2, k
        if block_id == 3: return r+2, k+3
        return None

    # Construct the decomposition
    for i, (A_blocks, B_blocks, C_blocks) in enumerate(outer_steps[0]):
        base_rank = i * 11
        
        # Inner Strassen (7 terms)
        for j, is_term_list in enumerate(inner_strassen):
            c_global = base_rank + j
            # is_term_list is a list with one tuple: (A_entries, B_entries, C_entries)
            A_entries, B_entries, C_entries = is_term_list[0]
            
            # Add contributions to U
            for (r, s, coeff) in A_entries:
                for (block_id, b_coeff) in A_blocks:
                    I, J = map_A(block_id, r, s)
                    if I is not None:
                        U[I*4 + J][c_global] += coeff * b_coeff
                        
            # Add contributions to V
            for (s, k, coeff) in B_entries:
                for (block_id, b_coeff) in B_blocks:
                    J, K = map_B(block_id, s, k)
                    if J is not None:
                        V[J*6 + K][c_global] += coeff * b_coeff
                        
            # Add contributions to W
            for (r, k, coeff) in C_entries:
                for (block_id, b_coeff) in C_blocks:
                    I, K = map_C(block_id, r, k)
                    if I is not None:
                        W[K*4 + I][c_global] += coeff * b_coeff
                        
        # Inner Schoolbook (4 terms)
        for j, isb_term_list in enumerate(inner_schoolbook):
            c_global = base_rank + 7 + j
            A_entries, B_entries, C_entries = isb_term_list[0]
            
            # Add contributions to U
            for (r, s, coeff) in A_entries:
                for (block_id, b_coeff) in A_blocks:
                    I, J = map_A(block_id, r, s)
                    if I is not None:
                        U[I*4 + J][c_global] += coeff * b_coeff
                        
            # Add contributions to V
            for (s, k, coeff) in B_entries:
                for (block_id, b_coeff) in B_blocks:
                    J, K = map_B(block_id, s, k)
                    if J is not None:
                        V[J*6 + K][c_global] += coeff * b_coeff
                        
            # Add contributions to W
            for (r, k, coeff) in C_entries:
                for (block_id, b_coeff) in C_blocks:
                    I, K = map_C(block_id, r, k)
                    if I is not None:
                        W[K*4 + I][c_global] += coeff * b_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
