import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin before 150s limit

    n, m, p = DIMENSIONS  # all 6

    # Build a rank-189 decomposition using 3x3 blocks of 2x2 Strassen
    # Each 2x2 block uses rank 7 (Strassen)
    # 3x3 = 9 blocks, total rank = 9 * 7 = 63... wait that's too few.
    
    # Actually for 6x6 matrix multiplication, we need to think about this differently.
    # The standard block decomposition: split 6x6 into 2x2 blocks of size 3x3 each.
    # Then multiply using Strassen on 2x2 blocks of 3x3 matrices.
    # Rank = 7 * (3*3)^2 = 7 * 81 = 567. That's worse than 216.
    
    # Alternative: split into 3x3 blocks of size 2x2.
    # 9 blocks of 2x2, Strassen each: rank = 9 * 7 = 63. No wait, that doesn't work.
    
    # Let me reconsider. The naive 6x6x6 has rank 6*6*6 = 216.
    # We need to find a decomposition with rank < 216.
    
    # One approach: use the fact that we can group operations.
    # Actually, let me try a different structural approach:
    # Use 2x2 Strassen embedded in a tiling scheme.
    
    # Tile 6x6 into a 3x3 grid of 2x2 blocks.
    # For each pair of 2x2 blocks (A-block, B-block), compute their product using Strassen (rank 7).
    # But wait, that gives us products of individual block pairs, not the full matrix product.
    
    # The correct approach: 
    # A is 6x6, B is 6x6. Split A into 3x3 grid of 2x2 blocks: A[i,j] for i,j in {0,1,2}.
    # Similarly B[i,j]. The product C[i,j] = sum_k A[i,k] B[k,j].
    # 
    # For each output block position (i,j), we need to compute sum_k A[i,k]*B[k,j].
    # Each term A[i,k]*B[k,j] is a product of two 2x2 matrices, which has rank 7 via Strassen.
    # For fixed (i,j), we sum over k in {0,1,2}, so 3 such products, each rank 7: total 21 per output block.
    # 9 output blocks * 21 = 189. 
    # 
    # But wait, the U, V, W factors are shared across all terms. Let me think more carefully.
    #
    # The tensor T has indices (i, j, k) in 6^3 = 216 dimensions for the naive case.
    # With Strassen tiling: 
    #   U has dimensions 36 x rank (n*m = 6*6 = 36 rows)
    #   V has dimensions 36 x rank (m*p = 6*6 = 36 rows)  
    #   W has dimensions 36 x rank (p*n = 6*6 = 36 rows)
    #
    # For each (i,j,k) block triple (where i,j,k range over 0,1,2), we use Strassen's 7 terms.
    # For a given block (i,k) of A and (k,j) of B, the Strassen decomposition gives 7 linear combinations.
    #
    # Let me define the mapping:
    # Global A index: (i*2 + r, k*2 + s) where i,r,s in {0,1,2}
    # Global B index: (k*2 + s, j*2 + t) where j,t in {0,1,2}
    # Global C index: (i*2 + r, j*2 + t)
    #
    # For each (i,k,j) triple, Strassen uses 7 intermediate values:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B11 + B01)
    #
    # And C is computed as:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now for the tensor decomposition, each "term" s in 1..7 for block (i,k,j) contributes:
    # U factor: linear combination of A[i*2+r, k*2+s] entries
    # V factor: linear combination of B[k*2+s, j*2+t] entries
    # W factor: linear combination of C[i*2+r, j*2+t] entries
    #
    # Let me code this up.

    # Strassen 2x2 decomposition coefficients
    # For 2x2 matrices A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b11+b01)
    # C00 = M1-M4+M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # Coefficients for each M term: (coeff for each A element, coeff for each B element)
    # A elements indexed as [a00, a01, a10, a11]
    # B elements indexed as [b00, b01, b10, b11]
    
    strassen_A = [
        [1, 0, 0, 1],   # M1: a00+a11
        [0, 0, 1, 1],   # M2: a10+a11
        [1, 0, 0, 0],   # M3: a00
        [0, 0, 0, 1],   # M4: a11
        [1, 1, 0, 0],   # M5: a00+a01
        [-1, 0, 1, 0],  # M6: a10-a00
        [0, 1, 0, -1],  # M7: a01-a11
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # M1: b00+b11
        [1, 0, 0, 0],   # M2: b00
        [0, -1, 1, -1], # M3: b10-b11
        [-1, 1, 0, 0],  # M4: b01-b00
        [0, 0, 0, 1],   # M5: b11
        [1, 0, 1, 0],   # M6: b00+b10
        [0, 1, 0, 1],   # M7: b11+b01
    ]
    
    # C coefficients: for each M term, how it contributes to each C element [C00,C01,C10,C11]
    strassen_C = [
        [1, 0, 0, 0],    # M1 -> C00
        [0, 0, 0, -1],   # M2 -> C11 (coefficient -1)
        [0, 1, 0, 1],    # M3 -> C01, C11
        [-1, 0, 1, 0],   # M4 -> C00 (-1), C10 (+1)
        [1, 1, 0, 0],    # M5 -> C00, C01
        [0, 0, 0, 1],    # M6 -> C11
        [1, 0, 0, 0],    # M7 -> C00
    ]
    
    # Total rank = 3 (i) * 3 (k) * 3 (j) * 7 (strassen) = 189
    rank = 189
    
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for iblock in range(3):      # block row of A / output
        for kblock in range(3):  # block col of A / block row of B
            for jblock in range(3):  # block col of B / output
                for s in range(7):   # Strassen term
                    # A elements: global indices (iblock*2+r, kblock*2+ss) for r,ss in {0,1}
                    # a00 -> (iblock*2+0, kblock*2+0) -> index iblock*2*6 + kblock*2+0 = 12*iblock + 2*kblock
                    # a01 -> (iblock*2+0, kblock*2+1) -> index 12*iblock + 2*kblock + 1
                    # a10 -> (iblock*2+1, kblock*2+0) -> index 12*iblock + 2*kblock + 6
                    # a11 -> (iblock*2+1, kblock*2+1) -> index 12*iblock + 2*kblock + 7
                    
                    a_base = iblock * 12 + kblock * 2
                    a_indices = [a_base, a_base + 1, a_base + 6, a_base + 7]
                    
                    for ei in range(4):
                        if strassen_A[s][ei] != 0:
                            U[a_indices[ei]][col] = strassen_A[s][ei]
                    
                    # B elements: global indices (kblock*2+ss, jblock*2+t) for ss,t in {0,1}
                    # b00 -> (kblock*2+0, jblock*2+0) -> index kblock*2*6 + jblock*2+0 = 12*kblock + 2*jblock
                    # b01 -> (kblock*2+0, jblock*2+1) -> index 12*kblock + 2*jblock + 1
                    # b10 -> (kblock*2+1, jblock*2+0) -> index 12*kblock + 2*jblock + 6
                    # b11 -> (kblock*2+1, jblock*2+1) -> index 12*kblock + 2*jblock + 7
                    
                    b_base = kblock * 12 + jblock * 2
                    b_indices = [b_base, b_base + 1, b_base + 6, b_base + 7]
                    
                    for ei in range(4):
                        if strassen_B[s][ei] != 0:
                            V[b_indices[ei]][col] = strassen_B[s][ei]
                    
                    # C elements: global indices (iblock*2+r, jblock*2+t) for r,t in {0,1}
                    # C00 -> (iblock*2+0, jblock*2+0) -> index iblock*2*6 + jblock*2+0 = 12*iblock + 2*jblock
                    # C01 -> (iblock*2+0, jblock*2+1) -> index 12*iblock + 2*jblock + 1
                    # C10 -> (iblock*2+1, jblock*2+0) -> index 12*iblock + 2*jblock + 6
                    # C11 -> (iblock*2+1, jblock*2+1) -> index 12*iblock + 2*jblock + 7
                    
                    c_base = iblock * 12 + jblock * 2
                    c_indices = [c_base, c_base + 1, c_base + 6, c_base + 7]
                    
                    for ei in range(4):
                        if strassen_C[s][ei] != 0:
                            W[c_indices[ei]][col] = strassen_C[s][ei]
                    
                    col += 1
    
    # Verify the decomposition is correct
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: sum_s U[:,s] * V[:,s] * W[:,s]^T should equal the matrix multiplication tensor
    # Tensor T[i,j,k] = delta for matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
    # In our indexing: C has index (i*6+j), A has index (i*6+k), B has index (k*6+j)
    # The tensor for matrix multiplication: T[c_idx, a_idx, b_idx] where
    # c_idx = i*6+j, a_idx = i*6+k, b_idx = k*6+j
    # T[c_idx, a_idx, b_idx] = 1 if c_idx corresponds to (i,j), a_idx to (i,k), b_idx to (k,j) for consistent i,j,k
    
    # Actually let's verify by computing: for each rank term s,
    # the contribution to C is (U[:,s] @ diag of something) ... 
    # Better: the decomposition says T = sum_s u_s v_s^T w_s^{[2]}
    # where T[a_idx, b_idx, c_idx] = sum_s U[a_idx,s] * V[b_idx,s] * W[c_idx,s]
    # We need T[a,b,c] = 1 iff a=i*6+k, b=k*6+j, c=i*6+j (the matrix multiplication tensor)
    
    # Let's verify on a random example
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:  # Only verify if we have time
        np.random.seed(42)
        A_test = np.random.randn(6, 6)
        B_test = np.random.randn(6, 6)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition:
        # C = sum_s (column of U corresponding to A entries) * (column of V for B entries) -> no
        # Actually: C[c_idx] = sum_s sum_a sum_b A[a_idx]*B[b_idx]*U[a_idx,s]*V[b_idx,s]*W[c_idx,s]
        # But that's not how tensor decomposition works for matrix multiplication.
        #
        # The correct interpretation: 
        # T[a,b,c] = sum_s U[a,s] V[b,s] W[c,s]
        # where a indexes A (36 entries), b indexes B (36 entries), c indexes C (36 entries)
        # And T should equal the matrix multiplication tensor.
        
        C_computed = np.zeros((6, 6))
        for s in range(rank):
            # u_s is the vector of coefficients for A
            # v_s is the vector of coefficients for B
            # w_s is the vector of coefficients for C
            pass
        
        # Actually the way to evaluate: 
        # Given A (flattened to 36) and B (flattened to 36),
        # the result C (flattened to 36) satisfies:
        # C = sum_s (W[:,s]) * dot(U[:,s], A_flat) * dot(V[:,s], B_flat)
        
        A_flat = A_test.flatten()  # row-major: A[i,j] at index i*6+j
        B_flat = B_test.flatten()
        
        C_flat = np.zeros(36)
        for s in range(rank):
            a_coeff = np.dot(U_arr[:, s], A_flat)
            b_coeff = np.dot(V_arr[:, s], B_flat)
            C_flat += W_arr[:, s] * a_coeff * b_coeff
        
        C_test = C_flat.reshape((6, 6))
        error = np.max(np.abs(C_test - C_expected))
        
        if error > 1e-10:
            # Something is wrong, fall back to schoolbook
            U_school = [[0] * 216 for _ in range(36)]
            V_school = [[0] * 216 for _ in range(36)]
            W_school = [[0] * 216 for _ in range(36)]
            col_s = 0
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        U_school[i * 6 + j][col_s] = 1
                        V_school[j * 6 + k][col_s] = 1
                        W_school[k * 6 + i][col_s] = 1
                        col_s += 1
            return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
