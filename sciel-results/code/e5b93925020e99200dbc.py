import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Base case: 2x2 Strassen decomposition (rank 7)
    # Indices i, j, k in 0..1
    # U[i, j], V[j, k], W[k, i]
    
    # Define the 7 terms for 2x2
    # Each term is a dict or list of (idx, val) for U, V, W
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    base_terms = []
    
    # Term 0: (A00 + A11)(B00 + B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    base_terms.append({
        'U': [(0*2+0, 1), (1*2+1, 1)],
        'V': [(0*2+0, 1), (1*2+1, 1)],
        'W': [(0*2+0, 1), (1*2+1, 1)]
    })
    
    # Term 1: (A10 + A11)B00
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (1,0)=1, (1,1)=-1
    base_terms.append({
        'U': [(1*2+0, 1), (1*2+1, 1)],
        'V': [(0*2+0, 1)],
        'W': [(1*2+0, 1), (1*2+1, -1)]
    })
    
    # Term 2: A00(B01 - B11)
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # W: (1,0)=1, (1,1)=1
    base_terms.append({
        'U': [(0*2+0, 1)],
        'V': [(0*2+1, 1), (1*2+1, -1)],
        'W': [(1*2+0, 1), (1*2+1, 1)]
    })
    
    # Term 3: A11(B10 - B00)
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # W: (0,0)=1, (1,0)=1
    base_terms.append({
        'U': [(1*2+1, 1)],
        'V': [(1*2+0, 1), (0*2+0, -1)],
        'W': [(0*2+0, 1), (1*2+0, 1)]
    })
    
    # Term 4: (A00 + A01)B11
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (0,0)=-1, (1,0)=1
    base_terms.append({
        'U': [(0*2+0, 1), (0*2+1, 1)],
        'V': [(1*2+1, 1)],
        'W': [(0*2+0, -1), (1*2+0, 1)]
    })
    
    # Term 5: (A10 - A00)(B00 + B01)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (1,1)=1
    base_terms.append({
        'U': [(1*2+0, 1), (0*2+0, -1)],
        'V': [(0*2+0, 1), (0*2+1, 1)],
        'W': [(1*2+1, 1)]
    })
    
    # Term 6: (A01 - A11)(B10 + B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1
    base_terms.append({
        'U': [(0*2+1, 1), (1*2+1, -1)],
        'V': [(1*2+0, 1), (1*2+1, 1)],
        'W': [(0*2+0, 1)]
    })
    
    # Now combine for 4x4
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    
    # Strassen steps for 4x4
    # Each step defines:
    # A blocks to sum, B blocks to sum
    # C blocks to add to, with coefficients
    
    # Step 0: M1 = (A00 + A11)(B00 + B11)
    # C00 += M1, C11 += M1
    # A blocks: 00, 11. B blocks: 00, 11.
    # C targets: 00 (coeff 1), 11 (coeff 1)
    
    # We need a helper to map 2x2 indices to 4x4 indices for a given block
    def map_U(idx, block_row, block_col):
        i, j = divmod(idx, 2)
        return (i + 2*block_row) * 4 + (j + 2*block_col)
        
    def map_V(idx, block_row, block_col):
        j, k = divmod(idx, 2)
        return (j + 2*block_row) * 4 + (k + 2*block_col)
        
    def map_W(idx, block_row, block_col):
        k, i = divmod(idx, 2)
        return (k + 2*block_row) * 4 + (i + 2*block_col)
        
    # Define the 7 Strassen steps
    # Each step is a list of (A_blocks, B_blocks, C_targets)
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_targets: list of (block_row, block_col, coeff)
    
    steps = [
        # M1
        [
            [(0,0,1), (1,1,1)], # A00 + A11
            [(0,0,1), (1,1,1)], # B00 + B11
            [(0,0,1), (1,1,1)]  # C00 + M1, C11 + M1
        ],
        # M2
        [
            [(1,0,1), (1,1,1)], # A10 + A11
            [(0,0,1)],           # B00
            [(1,0,1), (1,1,-1)]  # C10 + M2, C11 - M2
        ],
        # M3
        [
            [(0,0,1)],           # A00
            [(0,1,1), (1,1,-1)], # B01 - B11
            [(0,1,1), (1,1,1)]   # C01 + M3, C11 + M3
        ],
        # M4
        [
            [(1,1,1)],           # A11
            [(1,0,1), (0,0,-1)], # B10 - B00
            [(0,0,1), (1,0,1)]   # C00 + M4, C10 + M4
        ],
        # M5
        [
            [(0,0,1), (0,1,1)], # A00 + A01
            [(1,1,1)],           # B11
            [(0,0,-1), (0,1,1)]  # C00 - M5, C01 + M5
        ],
        # M6
        [
            [(1,0,1), (0,0,-1)], # A10 - A00
            [(0,0,1), (0,1,1)],  # B00 + B01
            [(1,1,1)]            # C11 + M6
        ],
        # M7
        [
            [(0,1,1), (1,1,-1)], # A01 - A11
            [(1,0,1), (1,1,1)],  # B10 + B11
            [(0,0,1)]            # C00 + M7
        ]
    ]
    
    for step in steps:
        A_defs, B_defs, C_defs = step
        for base_term in base_terms:
            # For each base term, we add a rank-1 component to U, V, W
            # U part
            for (idx, val) in base_term['U']:
                for (br, bc, coeff) in A_defs:
                    u_idx = map_U(idx, br, bc)
                    U[u_idx][col] += val * coeff
            
            # V part
            for (idx, val) in base_term['V']:
                for (br, bc, coeff) in B_defs:
                    v_idx = map_V(idx, br, bc)
                    V[v_idx][col] += val * coeff
            
            # W part
            for (idx, val) in base_term['W']:
                for (br, bc, coeff) in C_defs:
                    w_idx = map_W(idx, br, bc)
                    W[w_idx][col] += val * coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
