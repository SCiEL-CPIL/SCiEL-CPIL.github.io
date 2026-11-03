import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# Precomputed Strassen 2x2x2 decomposition
# Indices: 0:11, 1:12, 2:21, 3:22
# C = A * B
# A is 2x2, B is 2x2, C is 2x2
# Decomposition into 7 terms: U[:, t] * V[:, t] * W[:, t]
# where U, V, W are 4x7 matrices (flattened 2x2)

# Let's verify the standard Strassen coefficients for C = AB
# A = [[a, b], [c, d]], B = [[e, f], [g, h]]
# C = [[ae+bg, af+bh], [ce+dg, cf+dh]]

# m1 = (a+d)(e+h)
# m2 = (c+d)e
# m3 = a(f-h)
# m4 = d(g-e)
# m5 = (a+b)h
# m6 = (c-a)(e+f)
# m7 = (b-d)(g+h)

# C11 = m1 + m4 - m5 + m7
# C12 = m3 + m5
# C21 = m2 + m4
# C22 = m1 - m2 + m3 + m6

# Let's check C11:
# m1 = ae + ah + de + dh
# m4 = dg - de
# -m5 = -ah - bh
# m7 = bg + bh - dg - dh
# Sum = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
# = ae + bg. Correct.

# C12:
# m3 = af - ah
# m5 = ah + bh
# Sum = af + bh. Correct.

# C21:
# m2 = ce + de
# m4 = dg - de
# Sum = ce + dg. Correct.

# C22:
# m1 = ae + ah + de + dh
# -m2 = -ce - de
# m3 = af - ah
# m6 = ce + cf - ae - af
# Sum = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
# = dh + cf. Correct.

# So the vectors are:
# U (A side):
# t1: [1, 0, 0, 1]  (a, d)
# t2: [0, 0, 1, 1]  (c, d)
# t3: [1, 0, 0, 0]  (a)
# t4: [0, 0, 0, 1]  (d)
# t5: [1, 1, 0, 0]  (a, b)
# t6: [-1, 0, 1, 0] (-a, c) -> (c-a)
# t7: [0, 1, 0, -1] (b, -d) -> (b-d)

# V (B side):
# t1: [1, 0, 0, 1]  (e, h)
# t2: [1, 0, 0, 0]  (e)
# t3: [0, 1, 0, -1] (f, -h)
# t4: [0, -1, 1, 0] (-e, g)
# t5: [0, 0, 0, 1]  (h)
# t6: [1, 1, 0, 0]  (e, f)
# t7: [0, 0, 1, 1]  (g, h)

# W (C side):
# t1: [1, 0, 0, 1]  (C11, C22)
# t2: [0, 0, 1, 0]  (C21)
# t3: [0, 1, 0, 0]  (C12)
# t4: [1, 0, 1, 0]  (C11, C21)
# t5: [-1, 1, 0, 0] (-C11, C12)
# t6: [0, 0, 0, 1]  (C22)
# t7: [1, 0, 0, 0]  (C11)

STRASSEN_U = np.array([
    [1, 0, 1, 0, 1, -1, 0],
    [0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, -1]
], dtype=int)

STRASSEN_V = np.array([
    [1, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, -1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 1],
    [1, 0, -1, 0, 1, 0, 1]
], dtype=int)

STRASSEN_W = np.array([
    [1, 0, 0, 1, -1, 0, 1],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0]
], dtype=int)

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # Strategy:
    # Split B (4x9) into B1 (4x4) and B2 (4x5).
    # C = A * B = A * B1 + A * B2
    # C1 = A * B1 (4x4)
    # C2 = A * B2 (4x5)
    
    # For C1 (4x4x4):
    # Use Strassen on 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B1 = [[B00, B01], [B10, B11]]
    # C1 = [[C00, C01], [C10, C11]]
    
    # The 7 Strassen multiplications for 4x4x4 are:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # Each Mi is a product of 2x2 matrices.
    # We can decompose each Mi using the 2x2x2 Strassen decomposition (rank 7).
    # Total rank for C1 = 7 * 7 = 49.
    
    # For C2 (4x4x5):
    # Split B2 (4x5) into B2a (4x2) and B2b (4x3).
    # C2a = A * B2a (4x2)
    # C2b = A * B2b (4x3)
    
    # C2a (4x4x2):
    # Split A into 2x2 blocks. B2a is 4x2.
    # Let B2a = [[B2a00, B2a01], [B2a10, B2a11]] where each is 2x1.
    # This is not a standard square block.
    # We can treat it as two 4x4x1 multiplications?
    # Or 2x2x2 blocks?
    # Let's use schoolbook for the non-square parts to be safe, or find a better split.
    
    # Actually, let's just use Strassen for the 4x4x4 part (rank 49)
    # and Schoolbook for the 4x4x5 part (rank 80).
    # Total = 129.
    # Schoolbook total = 144.
    # Improvement: 15.
    
    # Can we do better for 4x4x5?
    # 4x4x5 = 4x4x2 + 4x4x3.
    # 4x4x2: Rank?
    # 4x4x3: Rank?
    
    # Let's stick to the 49 + 80 = 129 strategy first. It's safer and a clear improvement.
    
    rank_total = 49 + 80
    U = np.zeros((n*m, rank_total), dtype=int)
    V = np.zeros((m*p, rank_total), dtype=int)
    W = np.zeros((p*n, rank_total), dtype=int)
    
    col_offset = 0
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    # Indices:
    # A: 0..15 (4x4)
    # B1: 0..15 (4x4) -> V indices 0..15
    # C1: 0..15 (4x4) -> W indices 0..15
    
    # We need to map the 7 Strassen terms of the 4x4x4 to the global U, V, W.
    # The 7 terms are products of 2x2 matrices.
    # Let's define the 2x2 block indices.
    # A blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B1 blocks:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C1 blocks:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # The 7 Strassen multiplications M1..M7 involve sums/differences of these blocks.
    # Each Mi is a 2x2 matrix product.
    # We decompose each Mi using the 2x2x2 Strassen vectors.
    
    # Let's define a helper to add a 2x2x2 block decomposition at specific offsets.
    # But the inputs are sums of blocks.
    # This means the U vector for a term in M1 will have 1s in A00 and A11 positions.
    # This is exactly what the Strassen vectors do, but scaled to the 4x4 grid.
    
    # Let's build the 4x4x4 decomposition explicitly using the 2x2x2 vectors.
    
    # M1 = (A00+A11) * (B00+B11)
    # A_part for M1: A00 + A11
    # B_part for M1: B00 + B11
    # C_part for M1: C00 + C11 (with signs from 2x2x2 decomposition)
    
    # This is getting complex to map manually.
    # Let's construct the U, V, W matrices for the 4x4x4 case using the 2x2x2 vectors.
    
    # For each of the 7 outer Strassen terms, we have a 2x2x2 inner Strassen decomposition.
    # Outer term t (1..7) has:
    # U_outer[t]: linear combo of A blocks
    # V_outer[t]: linear combo of B blocks
    # W_outer[t]: linear combo of C blocks
    
    # Inner term s (1..7) has:
    # U_inner[s]: vector for 2x2 A
    # V_inner[s]: vector for 2x2 B
    # W_inner[s]: vector for 2x2 C
    
    # The combined term is U_outer[t] * U_inner[s] * V_outer[t] * V_inner[s] * W_outer[t] * W_inner[s].
    # This is not a simple outer product because U_outer is a sum of blocks.
    # Wait, U_outer[t] is a 4x4 matrix (sum of blocks).
    # U_inner[s] is a 2x2 matrix.
    # The product U_outer[t] * U_inner[s] is not defined in this context.
    
    # Correct approach:
    # The tensor for 4x4x4 is a sum of 49 rank-1 tensors.
    # Each rank-1 tensor is u * v * w.
    # u is a vector of length 16.
    # v is a vector of length 16.
    # w is a vector of length 16.
    
    # We can generate these 49 vectors by expanding the Strassen formula.
    
    # Let's write a function to generate the 49 vectors for 4x4x4.
    
    def generate_444_vectors():
        # Outer Strassen terms
        # A blocks: 00, 01, 10, 11
        # B blocks: 00, 01, 10, 11
        # C blocks: 00, 01, 10, 11
        
        # Outer U vectors (for A)
        # t1: A00 + A11
        # t2: A10 + A11
        # t3: A00
        # t4: A11
        # t5: A00 + A01
        # t6: A10 - A00
        # t7: A01 - A11
        
        # Outer V vectors (for B)
        # t1: B00 + B11
        # t2: B00
        # t3: B01 - B11
        # t4: B10 - B00
        # t5: B11
        # t6: B00 + B01
        # t7: B10 + B11
        
        # Outer W vectors (for C)
        # t1: C00 + C11
        # t2: C10
        # t3: C01
        # t4: C00 + C10
        # t5: -C00 + C01
        # t6: C11
        # t7: C00
        
        # We represent these as 4x4 matrices (or 16-vectors).
        
        A_blocks = {
            '00': [0,1,4,5],
            '01': [2,3,6,7],
            '10': [8,9,12,13],
            '11': [10,11,14,15]
        }
        
        def make_block_vector(blocks, signs):
            # blocks: list of block keys
            # signs: list of signs
            v = np.zeros(16, dtype=int)
            for key, sign in zip(blocks, signs):
                for idx in A_blocks[key]:
                    v[idx] += sign
            return v
        
        outer_U = [
            make_block_vector(['00','11'], [1,1]),
            make_block_vector(['10','11'], [1,1]),
            make_block_vector(['00'], [1]),
            make_block_vector(['11'], [1]),
            make_block_vector(['00','01'], [1,1]),
            make_block_vector(['10','00'], [1,-1]),
            make_block_vector(['01','11'], [1,-1])
        ]
        
        outer_V = [
            make_block_vector(['00','11'], [1,1]),
            make_block_vector(['00'], [1]),
            make_block_vector(['01','11'], [1,-1]),
            make_block_vector(['10','00'], [1,-1]),
            make_block_vector(['11'], [1]),
            make_block_vector(['00','01'], [1,1]),
            make_block_vector(['10','11'], [1,1])
        ]
        
        outer_W = [
            make_block_vector(['00','11'], [1,1]),
            make_block_vector(['10'], [1]),
            make_block_vector(['01'], [1]),
            make_block_vector(['00','10'], [1,1]),
            make_block_vector(['00','01'], [-1,1]),
            make_block_vector(['11'], [1]),
            make_block_vector(['00'], [1])
        ]
        
        # Inner Strassen vectors (2x2x2)
        # We have STRASSEN_U, STRASSEN_V, STRASSEN_W (4x7 each)
        
        vectors_U = []
        vectors_V = []
        vectors_W = []
        
        for t in range(7):
            u_out = outer_U[t]
            v_out = outer_V[t]
            w_out = outer_W[t]
            
            for s in range(7):
                u_in = STRASSEN_U[:, s]
                v_in = STRASSEN_V[:, s]
                w_in = STRASSEN_W[:, s]
                
                # Combine
                # u_comb is a 16-vector.
                # u_out has non-zeros in specific blocks.
                # u_in has non-zeros in specific 2x2 positions.
                
                # We need to expand u_out * u_in.
                # u_out is a sum of block vectors.
                # Each block vector is a 16-vector with 1s in a 2x2 block.
                # u_in is a 4-vector.
                
                # Let's expand u_out into its constituent blocks.
                # This is inefficient but clear.
                
                u_comb = np.zeros(16, dtype=int)
                v_comb = np.zeros(16, dtype=int)
                w_comb = np.zeros(16, dtype=int)
                
                # We need to know which blocks are in u_out and with what sign.
                # Instead of re-deriving, let's just use the fact that
                # u_out is a linear combination of basis blocks.
                
                # Let's define the basis blocks as 16-vectors.
                basis_00 = np.zeros(16, dtype=int); basis_00[[0,1,4,5]] = 1
                basis_01 = np.zeros(16, dtype=int); basis_01[[2,3,6,7]] = 1
                basis_10 = np.zeros(16, dtype=int); basis_10[[8,9,12,13]] = 1
                basis_11 = np.zeros(16, dtype=int); basis_11[[10,11,14,15]] = 1
                
                # Re-extract signs from outer_U[t]
                # This is tricky. Let's just hardcode the expansion.
                
                # Alternative: Just use the precomputed 4x4x4 Strassen vectors.
                # There are known tables.
                # Or, we can just use Schoolbook for 4x4x4 (rank 64) and Strassen for 2x2x2 parts?
                # No, Strassen 4x4x4 is rank 49.
                
                # Let's try a different approach:
                # Build the 4x4x4 decomposition by applying Strassen 2x2x2 to the 7 outer terms.
                # We need to map the 2x2 inner vectors to the 4x4 outer vectors.
                
                # For a given outer term t, the A part is a sum of blocks.
                # Let's say A_part = sum_k c_k * Block_k.
                # Then u_comb = sum_k c_k * (Block_k * u_in).
                # Block_k * u_in means placing the 4-vector u_in into the 16-vector at the positions of Block_k.
                
                # Let's define a function to expand a block.
                def expand_block(block_key, vec4):
                    v16 = np.zeros(16, dtype=int)
                    if block_key == '00':
                        v16[0] = vec4[0]
                        v16[1] = vec4[1]
                        v16[4] = vec4[2]
                        v16[5] = vec4[3]
                    elif block_key == '01':
                        v16[2] = vec4[0]
                        v16[3] = vec4[1]
                        v16[6] = vec4[2]
                        v16[7] = vec4[3]
                    elif block_key == '10':
                        v16[8] = vec4[0]
                        v16[9] = vec4[1]
                        v16[12] = vec4[2]
                        v16[13] = vec4[3]
                    elif block_key == '11':
                        v16[10] = vec4[0]
                        v16[11] = vec4[1]
                        v16[14] = vec4[2]
                        v16[15] = vec4[3]
                    return v16
                
                # We need the coefficients c_k for each outer term t.
                # Let's define them explicitly.
                
                outer_coeffs_U = [
                    {'00': 1, '11': 1},
                    {'10': 1, '11': 1},
                    {'00': 1},
                    {'11': 1},
                    {'00': 1, '01': 1},
                    {'10': 1, '00': -1},
                    {'01': 1, '11': -1}
                ]
                
                outer_coeffs_V = [
                    {'00': 1, '11': 1},
                    {'00': 1},
                    {'01': 1, '11': -1},
                    {'10': 1, '00': -1},
                    {'11': 1},
                    {'00': 1, '01': 1},
                    {'10': 1, '11': 1}
                ]
                
                outer_coeffs_W = [
                    {'00': 1, '11': 1},
                    {'10': 1},
                    {'01': 1},
                    {'00': 1, '10': 1},
                    {'00': -1, '01': 1},
                    {'11': 1},
                    {'00': 1}
                ]
                
                for block_key, coeff in outer_coeffs_U[t].items():
                    u_comb += coeff * expand_block(block_key, u_in)
                for block_key, coeff in outer_coeffs_V[t].items():
                    v_comb += coeff * expand_block(block_key, v_in)
                for block_key, coeff in outer_coeffs_W[t].items():
                    w_comb += coeff * expand_block(block_key, w_in)
                
                vectors_U.append(u_comb)
                vectors_V.append(v_comb)
                vectors_W.append(w_comb)
                
        return vectors_U, vectors_V, vectors_W
    
    u_444, v_444, w_444 = generate_444_vectors()
    
    for i in range(49):
        U[:, col_offset] = u_444[i]
        V[:16, col_offset] = v_444[i]  # B1 is first 4 cols of B
        W[:16, col_offset] = w_444[i]  # C1 is first 4 cols of C
        col_offset += 1
    
    # --- Part 2: 4x4x5 (Rank 80) ---
    # Schoolbook
    # A (4x4), B2 (4x5), C2 (4x5)
    # B2 indices in V: 16..35 (4*5=20)
    # C2 indices in W: 16..35 (4*5=20)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # B2 column k is actually column k+4 of B
                # C2 column k is actually column k+4 of C
                U[i*m+j, col_offset] = 1
                V[j*p + (k+4), col_offset] = 1
                W[(k+4)*n + i, col_offset] = 1
                col_offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
