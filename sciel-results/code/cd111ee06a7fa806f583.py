import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition for 2x4 by 4x8
    # Structure:
    # A (2x4) split into A_L (2x2), A_R (2x2)
    # B (4x8) split into B_T (2x8), B_B (2x8)
    # B_T split into 4 blocks of 2x2: b1, b2, b3, b4
    # B_B split into 4 blocks of 2x2: b5, b6, b7, b8
    # C = A_L B_T + A_R B_B
    # Each term A_L b_i is M_{2,2,2}
    # Total 8 M_{2,2,2} tensors.
    # Rank of each M_{2,2,2} is 7.
    # Total rank 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen decomposition for 2x2x2
    # Inputs: A (2x2), B (2x2)
    # Indices: i in 0..1, j in 0..1, k in 0..1
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    def add_strassen_tensor(offset, a_base, b_base, c_base, sign=1):
        # a_base: offset in A's flattened index (i*2+j)
        # b_base: offset in B's flattened index (j*2+k)
        # c_base: offset in C's flattened index (k*2+i)
        # This function adds the 7 rank-1 terms for one 2x2x2 multiplication
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        
        # Strassen's 7 products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10+b11)
        # M4 = a11(b01-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # We need to map these to the global U, V, W matrices.
        # The "scalars" a00, etc. are linear combinations of the input matrix entries.
        # The "scalars" b00, etc. are linear combinations of the input matrix entries.
        
        # Global A index: i*m + j. Here m=4.
        # A_L corresponds to j=0,1. A_R to j=2,3.
        # If a_base is the offset for A_L, then a00 is at a_base+0, a01 at a_base+1, a10 at a_base+2, a11 at a_base+3.
        # Wait, A is 2x4. Flattened:
        # Row 0: j=0,1,2,3 -> indices 0,1,2,3
        # Row 1: j=0,1,2,3 -> indices 4,5,6,7
        # A_L (cols 0,1):
        # a00 -> idx 0
        # a01 -> idx 1
        # a10 -> idx 4
        # a11 -> idx 5
        
        # B is 4x8. Flattened:
        # Row 0: k=0..7 -> indices 0..7
        # Row 1: k=0..7 -> indices 8..15
        # Row 2: k=0..7 -> indices 16..23
        # Row 3: k=0..7 -> indices 24..31
        # B_T (rows 0,1).
        # b1 (cols 0,1):
        # b00 -> idx 0
        # b01 -> idx 1
        # b10 -> idx 8
        # b11 -> idx 9
        
        # C is 2x8. Flattened (k*n+i):
        # k=0: i=0,1 -> 0,1
        # k=1: i=0,1 -> 2,3
        # ...
        # c_base is offset for the 2x2 block of C.
        # c00 (k=0, i=0) -> c_base + 0
        # c01 (k=0, i=1) -> c_base + 1
        # c10 (k=1, i=0) -> c_base + 2
        # c11 (k=1, i=1) -> c_base + 3
        
        # We define the 7 terms.
        # Each term is (u_vec, v_vec, w_vec)
        # u_vec has non-zeros at A indices.
        # v_vec has non-zeros at B indices.
        # w_vec has non-zeros at C indices.
        
        terms = []
        
        # M1: (a00+a11)(b00+b11) -> c00 += M1, c01 += M1, c10 += M1, c11 += M1?
        # No, Strassen:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's list the contributions to each C entry.
        # C00: M1, -M4, M5, -M7
        # C01: M3, M7
        # C10: M2, M4
        # C11: M1, -M2, M3, M6
        
        # We will add 7 columns to U, V, W.
        # For each of the 7 products, we define the A part, B part, and C part.
        
        # Product 1: (a00+a11) * (b00+b11)
        # A part: a00 (1), a11 (1)
        # B part: b00 (1), b11 (1)
        # C part: c00 (1), c11 (1)
        # Wait, M1 contributes to C00 and C11.
        # But Strassen's M1 is just the product. The assignment to C is done by summing the terms.
        # In tensor decomposition, each rank-1 term is $u v^T w^T$.
        # We need to decompose the tensor.
        # The tensor for M_{2,2,2} has rank 7.
        # The 7 vectors (u,v,w) are:
        
        # 1. u=(1,0,0,1) [a00, a01, a10, a11], v=(1,0,0,1) [b00, b01, b10, b11], w=(1,0,0,1) [c00, c01, c10, c11]?
        # No, w is (1, 0, 0, 1) corresponds to c00 and c11?
        # C indices: 0:c00, 1:c01, 2:c10, 3:c11.
        # w=(1, 0, 0, 1) means w[0]=1, w[3]=1.
        
        # Let's use the standard Strassen vectors for the tensor.
        # T = sum_{r=1}^7 u_r v_r^T w_r^T
        # u_r, v_r, w_r are vectors of length 4 (for 2x2 matrices).
        
        u1 = [1, 0, 0, 1] # a00 + a11
        v1 = [1, 0, 0, 1] # b00 + b11
        w1 = [1, 0, 0, 1] # c00 + c11 ? No.
        
        # Correct Strassen decomposition:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to express this as sum of 7 rank-1 tensors.
        # Term 1: M1. u=(1,0,0,1), v=(1,0,0,1). w must be such that it contributes to C00 and C11.
        # w1 = [1, 0, 0, 1].
        
        # Term 2: M2. u=(0,0,1,1), v=(1,0,0,0). w2 = [0, 0, 1, -1]. (C10 += M2, C11 -= M2)
        
        # Term 3: M3. u=(1,0,0,0), v=(0,0,1,1). w3 = [0, 1, 0, 1]. (C01 += M3, C11 += M3)
        
        # Term 4: M4. u=(0,0,0,1), v=(0,1,-1,0). w4 = [-1, 0, 1, 0]. (C00 -= M4, C10 += M4)
        
        # Term 5: M5. u=(1,1,0,0), v=(0,0,0,1). w5 = [1, 0, 0, 0]. (C00 += M5)
        
        # Term 6: M6. u=(0,0,1,-1), v=(1,1,0,0). w6 = [0, 0, 0, 1]. (C11 += M6)
        
        # Term 7: M7. u=(0,1,-1,0), v=(0,0,1,1). w7 = [0, 1, 0, -1]. (C01 += M7, C11 -= M7)
        
        # Let's verify C00: w1[0] + w4[0] + w5[0] + w7[0] = 1 - 1 + 1 + 0 = 1. Correct.
        # C01: w1[1] + w4[1] + w5[1] + w7[1] = 0 + 0 + 0 + 1 = 1. Correct.
        # C10: w1[2] + w4[2] + w5[2] + w7[2] = 0 + 1 + 0 + 0 = 1. Correct.
        # C11: w1[3] + w2[3] + w3[3] + w6[3] + w7[3] = 1 - 1 + 1 + 1 - 1 = 1. Correct.
        
        vecs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 1, -1, 0], [-1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, -1, 0], [0, 0, 1, 1], [0, 1, 0, -1])
        ]
        
        for r in range(7):
            col = offset + r
            u_vec, v_vec, w_vec = vecs[r]
            
            # Map local indices to global indices
            # Local A idx: 0:a00, 1:a01, 2:a10, 3:a11
            # Global A idx: 
            # a00: a_base + 0
            # a01: a_base + 1
            # a10: a_base + 2
            # a11: a_base + 3
            
            # Local B idx: 0:b00, 1:b01, 2:b10, 3:b11
            # Global B idx:
            # b00: b_base + 0
            # b01: b_base + 1
            # b10: b_base + 2
            # b11: b_base + 3
            
            # Local C idx: 0:c00, 1:c01, 2:c10, 3:c11
            # Global C idx:
            # c00: c_base + 0
            # c01: c_base + 1
            # c10: c_base + 2
            # c11: c_base + 3
            
            for l in range(4):
                if u_vec[l] != 0:
                    U[a_base + l][col] += sign * u_vec[l]
                if v_vec[l] != 0:
                    V[b_base + l][col] += sign * v_vec[l]
                if w_vec[l] != 0:
                    W[c_base + l][col] += sign * w_vec[l]
            
    # Define the 8 blocks
    # A_L: a_base = 0 (indices 0,1,4,5? No, 0,1,2,3 are row 0? No.)
    # A is 2x4.
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # A_L (cols 0,1):
    # a00 (0,0) -> 0
    # a01 (0,1) -> 1
    # a10 (1,0) -> 4
    # a11 (1,1) -> 5
    # So a_base_L is not a simple offset.
    # We need to map local 0,1,2,3 to global 0,1,4,5.
    
    # Let's modify add_strassen_tensor to take explicit index maps.
    
    def add_strassen_tensor_mapped(offset, a_map, b_map, c_map, sign=1):
        vecs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [0, 1, -1, 0], [-1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, -1, 0], [0, 0, 1, 1], [0, 1, 0, -1])
        ]
        for r in range(7):
            col = offset + r
            u_vec, v_vec, w_vec = vecs[r]
            for l in range(4):
                if u_vec[l] != 0:
                    U[a_map[l]][col] += sign * u_vec[l]
                if v_vec[l] != 0:
                    V[b_map[l]][col] += sign * v_vec[l]
                if w_vec[l] != 0:
                    W[c_map[l]][col] += sign * w_vec[l]

    # Maps for A_L (cols 0,1)
    a_L = [0, 1, 4, 5]
    # Maps for A_R (cols 2,3)
    a_R = [2, 3, 6, 7]
    
    # Maps for B blocks
    # B is 4x8.
    # Row 0: 0-7
    # Row 1: 8-15
    # Row 2: 16-23
    # Row 3: 24-31
    
    # B_T (rows 0,1)
    # b1 (cols 0,1): 0,1,8,9
    # b2 (cols 2,3): 2,3,10,11
    # b3 (cols 4,5): 4,5,12,13
    # b4 (cols 6,7): 6,7,14,15
    
    # B_B (rows 2,3)
    # b5 (cols 0,1): 16,17,24,25
    # b6 (cols 2,3): 18,19,26,27
    # b7 (cols 4,5): 20,21,28,29
    # b8 (cols 6,7): 22,23,30,31
    
    # Maps for C blocks
    # C is 2x8. k*n+i.
    # k=0: 0,1
    # k=1: 2,3
    # k=2: 4,5
    # k=3: 6,7
    # k=4: 8,9
    # k=5: 10,11
    # k=6: 12,13
    # k=7: 14,15
    
    # c1 (k=0,1): 0,1,2,3
    # c2 (k=2,3): 4,5,6,7
    # c3 (k=4,5): 8,9,10,11
    # c4 (k=6,7): 12,13,14,15
    
    # Term 1: A_L b1
    add_strassen_tensor_mapped(0, a_L, [0,1,8,9], [0,1,2,3], 1)
    # Term 2: A_R b5
    add_strassen_tensor_mapped(7, a_R, [16,17,24,25], [0,1,2,3], 1)
    
    # Term 3: A_L b2
    add_strassen_tensor_mapped(14, a_L, [2,3,10,11], [4,5,6,7], 1)
    # Term 4: A_R b6
    add_strassen_tensor_mapped(21, a_R, [18,19,26,27], [4,5,6,7], 1)
    
    # Term 5: A_L b3
    add_strassen_tensor_mapped(28, a_L, [4,5,12,13], [8,9,10,11], 1)
    # Term 6: A_R b7
    add_strassen_tensor_mapped(35, a_R, [20,21,28,29], [8,9,10,11], 1)
    
    # Term 7: A_L b4
    add_strassen_tensor_mapped(42, a_L, [6,7,14,15], [12,13,14,15], 1)
    # Term 8: A_R b8
    add_strassen_tensor_mapped(49, a_R, [22,23,30,31], [12,13,14,15], 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
