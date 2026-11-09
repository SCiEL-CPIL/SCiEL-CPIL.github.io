import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Generate Strassen 4x4x4 decomposition (rank 49)
    # We represent vectors as lists of floats/ints
    
    # Base case: Strassen 2x2x2 terms
    # Returns list of 7 tuples (U_local, V_local, W_local)
    # U, V, W are length 4
    def get_strassen_2x2():
        terms = []
        # M1: (A00+A11)(B00+B11)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]))
        # M2: (A10+A11)B00
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
        # M3: A00(B10-B11)
        terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]))
        # M4: A11(B01-B00)
        terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 0, 0]))
        # M5: (A00+A10)B11
        terms.append(([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0]))
        # M6: (A10-A00)(B00+B01)
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7: (A01-A11)(B10+B11)
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return terms

    base_terms = get_strassen_2x2()
    
    # Definitions of the 7 multiplications for 4x4 Strassen
    # Each entry: (A_blocks, B_blocks, C_blocks, coeff)
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    # block_idx: 0=00, 1=01, 2=10, 3=11
    
    strassen_4x4_mults = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]), # M2 contributes to C10 (block 2)
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]), # M3 contributes to C01 (block 1)
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1)]), # M4 contributes to C00 (block 0)
        # M5
        ([(0, 1), (2, 1)], [(3, 1)], [(0, -1)]), # M5 contributes to C00 (block 0) with -1
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6 contributes to C11 (block 3)
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]), # M7 contributes to C00 (block 0)
    ]
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    def expand_2x2_to_4x4(vec, blocks):
        # vec: length 4
        # blocks: list of (block_idx, coeff)
        # returns: length 16
        res = [0.0] * 16
        for b_idx, coeff in blocks:
            # b_idx 0 -> rows 0-1, cols 0-1 of 2x2 blocks?
            # 4x4 matrix index: i (0..3), j (0..3)
            # Block 0 (00): i in 0..1, j in 0..1
            # Block 1 (01): i in 0..1, j in 2..3
            # Block 2 (10): i in 2..3, j in 0..1
            # Block 3 (11): i in 2..3, j in 2..3
            
            # vec index: k (0..1), i (0..1) -> k*2 + i
            # 4x4 index: I (0..3), J (0..3) -> I*4 + J
            # For block b_idx:
            # row_offset = (b_idx // 2) * 2
            # col_offset = (b_idx % 2) * 2
            
            row_off = (b_idx // 2) * 2
            col_off = (b_idx % 2) * 2
            
            for k in range(2):
                for i in range(2):
                    val = vec[k * 2 + i]
                    if val != 0:
                        I = row_off + k
                        J = col_off + i
                        idx = I * 4 + J
                        res[idx] += coeff * val
        return res

    for A_blocks, B_blocks, C_blocks in strassen_4x4_mults:
        for U_loc, V_loc, W_loc in base_terms:
            U_full = expand_2x2_to_4x4(U_loc, A_blocks)
            V_full = expand_2x2_to_4x4(V_loc, B_blocks)
            W_full = expand_2x2_to_4x4(W_loc, C_blocks)
            
            # Check if term is non-zero
            if any(x != 0 for x in U_full) and any(x != 0 for x in V_full) and any(x != 0 for x in W_full):
                U_4x4.append(U_full)
                V_4x4.append(V_full)
                W_4x4.append(W_full)
                
    # 2. Generate Schoolbook terms for the rest (indices 0..4, excluding 0..3 for i,j,k)
    U_5x5 = []
    V_5x5 = []
    W_5x5 = []
    
    # Append the 4x4 Strassen terms to the 5x5 lists
    # We need to embed the 4x4 vectors into 5x5 vectors
    def embed_4x4_to_5x5(vec4):
        # vec4 is length 16 (indices 0..15 corresponding to 4x4)
        # 5x5 vector is length 25
        res = [0.0] * 25
        for idx in range(16):
            # idx in 4x4: I*4 + J where I,J in 0..3
            # Same indices in 5x5: I*5 + J
            I = idx // 4
            J = idx % 4
            res[I * 5 + J] = vec4[idx]
        return res

    for u, v, w in zip(U_4x4, V_4x4, W_4x4):
        U_5x5.append(embed_4x4_to_5x5(u))
        V_5x5.append(embed_4x4_to_5x5(v))
        W_5x5.append(embed_4x4_to_5x5(w))
        
    # Add schoolbook terms for the remaining indices
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue
                u = [0.0] * 25
                v = [0.0] * 25
                w = [0.0] * 25
                u[i * 5 + j] = 1.0
                v[j * 5 + k] = 1.0
                w[k * 5 + i] = 1.0
                U_5x5.append(u)
                V_5x5.append(v)
                W_5x5.append(w)
                
    # 3. Format output
    # U: (n*m, rank) -> (25, rank)
    # V: (m*p, rank) -> (25, rank)
    # W: (p*n, rank) -> (25, rank)
    
    rank = len(U_5x5)
    
    # Transpose to (rows, rank)
    U_out = [[U_5x5[r][c] for r in range(25)] for c in range(rank)]
    V_out = [[V_5x5[r][c] for r in range(25)] for c in range(rank)]
    W_out = [[W_5x5[r][c] for r in range(25)] for c in range(rank)]
    
    # Convert to ints if possible
    U_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_out]
    V_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_out]
    W_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_out]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
