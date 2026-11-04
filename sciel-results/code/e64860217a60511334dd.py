import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Strategy: Hybrid Strassen/Schoolbook
    # Decompose the 4x4x7 multiplication into:
    # 1. A 4x4x4 multiplication (using Strassen's algorithm, rank 49)
    # 2. A 4x4x3 multiplication (using Schoolbook, rank 48)
    # Total rank = 49 + 48 = 97
    
    # We need to construct U, V, W for the full problem.
    # U shape: (n*m, rank) = (16, 97)
    # V shape: (m*p, rank) = (28, 97)
    # W shape: (p*n, rank) = (28, 97)
    
    # The tensor T is defined by indices (a_idx, b_idx, c_idx)
    # a_idx = i*m + j (i in 0..3, j in 0..3)
    # b_idx = j*p + k (j in 0..3, k in 0..6)
    # c_idx = k*n + i (k in 0..6, i in 0..3)
    
    # We split the k-index (0..6) into two parts:
    # Part 1: k in 0..3 (4 columns)
    # Part 2: k in 4..6 (3 columns)
    
    # Part 1: 4x4x4 multiplication
    # This is a standard 4x4 matrix multiplication.
    # We use Strassen's algorithm for 4x4.
    # Strassen for 2x2 has rank 7.
    # Strassen for 4x4 is the tensor square of Strassen for 2x2, so rank 7*7 = 49.
    
    # Part 2: 4x4x3 multiplication
    # We use schoolbook for this part.
    # Rank = 4 * 4 * 3 = 48.
    
    total_rank = 49 + 48
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # --- Part 1: Strassen 4x4x4 ---
    # We need to construct the rank-49 decomposition for 4x4x4.
    # This is equivalent to taking the Kronecker product of the rank-7 decomposition for 2x2x2.
    
    # First, define the rank-7 decomposition for 2x2x2 (Strassen).
    # Indices for 2x2x2:
    # a_idx2 = i*m2 + j (i,j in 0..1), m2=2
    # b_idx2 = j*p2 + k (j,k in 0..1), p2=2
    # c_idx2 = k*n2 + i (k,i in 0..1), n2=2
    
    # Strassen's 7 multiplications for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # We represent each rank-1 tensor by its (u, v, w) vectors.
    # For 2x2x2, u, v, w are of length 4.
    # We will store these as a list of 7 tuples (u, v, w).
    
    strassen_2x2 = []
    
    # Helper to create a vector for 2x2 indices
    def vec2(i, j, val=1):
        v = np.zeros(4, dtype=int)
        v[i*2 + j] = val
        return v
    
    # M1: (A11 + A22) * (B11 + B22) -> C11 += M1, C12 += M1, C21 += M1, C22 += M1? No.
    # C11 = M1 - M2 + M3 + M4
    # C12 = M5 + M6
    # C21 = M2 + M7
    # C22 = M1 - M3 + M5 - M7
    
    # Let's define the contributions to C.
    # M1 contributes to C11, C12, C21, C22? No, M1 is a scalar.
    # The formula is:
    # C11 = M1 - M2 + M3 + M4
    # C12 = M5 + M6
    # C21 = M2 + M7
    # C22 = M1 - M3 + M5 - M7
    
    # We need to express T = sum_r u_r v_r w_r.
    # For each M_r, we have a specific u_r, v_r, and it contributes to specific c_indices with specific signs.
    # Actually, it's easier to define the rank-1 tensors directly.
    # A rank-1 tensor is u \otimes v \otimes w.
    # For Strassen, we have 7 "intermediate" multiplications.
    # Each intermediate multiplication M_r = u_r^T A v_r (where A is 2x2).
    # Then C is a linear combination of M_r.
    # This means the tensor T can be written as sum_r (u_r \otimes v_r \otimes w_r) where w_r is the vector of coefficients for C.
    
    # Let's define the 7 terms:
    # Term 1: u1 = [1, 0, 0, 1] (A11 + A22), v1 = [1, 0, 0, 1] (B11 + B22)
    # w1: C11 += 1, C22 += 1. (And others? No, M1 is used in C11 and C22)
    # Wait, the formula is C11 = M1 - M2 + M3 + M4.
    # So M1 contributes +1 to C11.
    # M1 also contributes +1 to C22.
    # So w1 has 1 at C11 and 1 at C22.
    
    # Let's map indices for 2x2:
    # A: 00, 01, 10, 11 -> idx 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> idx 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> idx 0, 1, 2, 3 (using k*n+i: 0*2+0=0, 0*2+1=1? No. c_idx = k*n+i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 2
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 3
    # So C indices: 00->0, 10->1, 01->2, 11->3.
    
    # Let's re-verify c_idx = k*n + i.
    # n=2.
    # C[0,0] (i=0, k=0) -> 0*2+0 = 0
    # C[0,1] (i=0, k=1) -> 1*2+0 = 2
    # C[1,0] (i=1, k=0) -> 0*2+1 = 1
    # C[1,1] (i=1, k=1) -> 1*2+1 = 3
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M6
    # C10 = M2 + M7
    # C11 = M1 - M3 + M5 - M7
    
    # We will create 7 rank-1 tensors for 2x2x2.
    # Each is (u, v, w).
    
    # 1. M1
    u1 = np.array([1, 0, 0, 1]) # A00 + A11
    v1 = np.array([1, 0, 0, 1]) # B00 + B11
    w1 = np.array([1, 0, 0, 1]) # C00 + C11 (idx 0 and 3)
    strassen_2x2.append((u1, v1, w1))
    
    # 2. M2
    u2 = np.array([0, 0, 1, 1]) # A10 + A11
    v2 = np.array([1, 0, 0, 0]) # B00
    w2 = np.array([-1, 1, 0, 0]) # -C00 + C10 (idx 0 and 1)
    strassen_2x2.append((u2, v2, w2))
    
    # 3. M3
    u3 = np.array([1, 0, 0, 0]) # A00
    v3 = np.array([0, 1, 0, -1]) # B01 - B11
    w3 = np.array([1, 0, 0, -1]) # C00 - C11 (idx 0 and 3)
    strassen_2x2.append((u3, v3, w3))
    
    # 4. M4
    u4 = np.array([0, 0, 0, 1]) # A11
    v4 = np.array([-1, 0, 1, 0]) # B10 - B00
    w4 = np.array([1, 0, 0, 0]) # C00 (idx 0)
    strassen_2x2.append((u4, v4, w4))
    
    # 5. M5
    u5 = np.array([1, 1, 0, 0]) # A00 + A01
    v5 = np.array([0, 0, 0, 1]) # B11
    w5 = np.array([0, 0, 1, 1]) # C01 + C11 (idx 2 and 3)
    strassen_2x2.append((u5, v5, w5))
    
    # 6. M6
    u6 = np.array([-1, 0, 1, 0]) # A10 - A00
    v6 = np.array([1, 1, 0, 0]) # B00 + B01
    w6 = np.array([0, 0, 1, 0]) # C01 (idx 2)
    strassen_2x2.append((u6, v6, w6))
    
    # 7. M7
    u7 = np.array([0, 1, 0, -1]) # A01 - A11
    v7 = np.array([0, 0, 1, 1]) # B10 + B11
    w7 = np.array([0, 1, 0, -1]) # C10 - C11 (idx 1 and 3)
    strassen_2x2.append((u7, v7, w7))
    
    # Now, construct the 4x4x4 decomposition using the Kronecker product of these 7 terms.
    # The 4x4x4 tensor is the tensor square of the 2x2x2 tensor.
    # If T_2 = sum_{r=1}^7 u_r \otimes v_r \otimes w_r, then T_4 = sum_{r1, r2} (u_{r1} \otimes u_{r2}) \otimes (v_{r1} \otimes v_{r2}) \otimes (w_{r1} \otimes w_{r2}).
    # The rank is 7*7 = 49.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # A 4x4 matrix is composed of four 2x2 blocks:
    # A[0:2, 0:2] (A00), A[0:2, 2:4] (A01), A[2:4, 0:2] (A10), A[2:4, 2:4] (A11)
    # The index for A in 4x4 is i*4 + j.
    # The index for A in 2x2 is i2*2 + j2.
    # The mapping is:
    # i = i_block * 2 + i2
    # j = j_block * 2 + j2
    
    # We will fill the first 49 columns of U, V, W.
    col = 0
    for r1 in range(7):
        u1_2, v1_2, w1_2 = strassen_2x2[r1]
        for r2 in range(7):
            u2_2, v2_2, w2_2 = strassen_2x2[r2]
            
            # Kronecker product for u
            u_4 = np.kron(u1_2, u2_2)
            # Kronecker product for v
            v_4 = np.kron(v1_2, v2_2)
            # Kronecker product for w
            w_4 = np.kron(w1_2, w2_2)
            
            # We need to be careful with the index mapping.
            # The Kronecker product u1 \otimes u2 corresponds to the vectorization of the block matrix.
            # If u1 is for the block index (i_block, j_block) and u2 is for the local index (i2, j2),
            # then the combined index is (i_block*2 + i2, j_block*2 + j2).
            # The vectorization order is row-major: idx = i*4 + j.
            # The Kronecker product of two vectors of length 4 (which are 2x2 matrices flattened)
            # gives a vector of length 16.
            # Does np.kron(u1, u2) match the row-major flattening of the block matrix?
            # Let u1 = [a, b, c, d] (A00, A01, A10, A11 of the 2x2 block matrix? No, u1 is the coefficients for the 2x2 A).
            # Actually, u1 is a vector of length 4 representing the 2x2 matrix A in the 2x2 problem.
            # The 4x4 matrix A is:
            # [ A00 A01 ]
            # [ A10 A11 ]
            # where Aij are 2x2 blocks.
            # The vectorization of A is [A00_vec, A01_vec, A10_vec, A11_vec]?
            # No, standard row-major flattening of a 4x4 matrix is:
            # [A00, A01, A02, A03, A10, ...]
            # The Kronecker product u1 \otimes u2 where u1 is the block coefficient and u2 is the local coefficient
            # results in a vector that groups by the first index.
            # Let's verify:
            # u1 = [1, 0, 0, 0] (selects A00 block)
            # u2 = [1, 0, 0, 0] (selects A00 local 0,0)
            # u1 \otimes u2 = [1, 0, 0, 0, 0, 0, 0, 0, ...] -> Index 0. Correct.
            # u1 = [0, 1, 0, 0] (selects A01 block)
            # u2 = [1, 0, 0, 0]
            # u1 \otimes u2 = [0, 0, 0, 0, 1, 0, 0, 0, ...] -> Index 4.
            # Index 4 in 4x4 row-major is i=1, j=0.
            # But A01 block is at i=0, j=2. Index 2.
            # So the Kronecker product order is different.
            # np.kron(u1, u2) where u1 is length 4 and u2 is length 4.
            # The result is length 16.
            # The index is i1*4 + i2? No.
            # If u1 is [a, b, c, d] and u2 is [e, f, g, h],
            # u1 \otimes u2 = [a*e, a*f, a*g, a*h, b*e, b*f, b*g, b*h, ...]
            # This corresponds to the index i1*4 + i2.
            # In our case, i1 is the block index (0..3) and i2 is the local index (0..3).
            # But the block index for A is (i_block, j_block) which maps to i1 = i_block*2 + j_block?
            # No, the 2x2 problem has indices 0,1,2,3 for A00, A01, A10, A11.
            # Let's call the 2x2 indices I2.
            # The 4x4 index I4 is related to I2 and I2' (local) by I4 = I2*4 + I2'?
            # If I2 = 0 (A00 block), and I2' = 0 (local 0,0), I4 = 0. Correct.
            # If I2 = 1 (A01 block), and I2' = 0, I4 = 4.
            # But A01 block is at rows 0-1, cols 2-3.
            # The first element is A[0,2] which is index 2.
            # So the mapping is not simply I2*4 + I2'.
            # We need to permute the indices.
            
            # Let's construct the 4x4 vectors explicitly by iterating over the 16 indices.
            # For a given rank-1 term (u1, v1, w1) and (u2, v2, w2):
            # The coefficient for A[i,j] is u1[i_block, j_block] * u2[i_local, j_local]
            # where i = i_block*2 + i_local, j = j_block*2 + j_local.
            
            u_4 = np.zeros(16, dtype=int)
            v_4 = np.zeros(16, dtype=int)
            w_4 = np.zeros(16, dtype=int)
            
            for i_block in range(2):
                for j_block in range(2):
                    # Index in 2x2 vector
                    idx_2_u = i_block * 2 + j_block
                    # Note: u1_2 is a flat vector of length 4.
                    # u1_2[idx_2_u] is the coefficient for A[i_block, j_block] in the 2x2 problem.
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            idx_2_v = i_local * 2 + j_local
                            idx_4_u = i_block * 4 + j_block * 2 + i_local * 2 + j_local
                            # Wait, the 4x4 index is i*4 + j.
                            # i = i_block*2 + i_local
                            # j = j_block*2 + j_local
                            idx_4_u = (i_block*2 + i_local)*4 + (j_block*2 + j_local)
                            
                            u_4[idx_4_u] = u1_2[idx_2_u] * u2_2[idx_2_v]
                            v_4[idx_4_u] = v1_2[idx_2_u] * v2_2[idx_2_v]
                            
                            # For W, the index is k*n + i.
                            # k = k_block*2 + k_local
                            # i = i_block*2 + i_local
                            # The 2x2 W index is k2*n2 + i2 = k2*2 + i2.
                            # The 4x4 W index is k4*n4 + i4 = k4*4 + i4.
                            # We need to map the 2x2 W indices to 4x4.
                            # w1_2 and w2_2 are vectors of length 4.
                            # w1_2[idx_2_w] where idx_2_w = k_block*2 + i_block.
                            # w2_2[idx_2_w] where idx_2_w = k_local*2 + i_local.
                            # The coefficient for C[k4, i4] is w1_2[k_block, i_block] * w2_2[k_local, i_local].
                            
                            # We need to fill w_4 for all k4, i4.
                            # We can do this in a separate loop or combine.
                            pass
            
            # Let's redo the W construction properly.
            w_4 = np.zeros(16, dtype=int)
            for k_block in range(2):
                for i_block in range(2):
                    idx_2_w1 = k_block * 2 + i_block
                    for k_local in range(2):
                        for i_local in range(2):
                            idx_2_w2 = k_local * 2 + i_local
                            k4 = k_block * 2 + k_local
                            i4 = i_block * 2 + i_local
                            idx_4_w = k4 * 4 + i4
                            w_4[idx_4_w] = w1_2[idx_2_w1] * w2_2[idx_2_w2]
            
            # Fill U, V, W
            U[:, col] = u_4
            V[:, col] = v_4
            W[:, col] = w_4
            col += 1
    
    # --- Part 2: Schoolbook 4x4x3 ---
    # We need to compute C[:, 4:7] = A * B[:, 4:7].
    # This is a 4x4 matrix times a 4x3 matrix.
    # Schoolbook rank is 4 * 4 * 3 = 48.
    # We add these 48 terms to U, V, W.
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # k_global = k + 4
                k_global = k + 4
                a_idx = i * 4 + j
                b_idx = j * 7 + k_global
                c_idx = k_global * 4 + i
                
                U[a_idx, col] = 1
                V[b_idx, col] = 1
                W[c_idx, col] = 1
                col += 1
    
    # Verify the decomposition (optional, but good for debugging)
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # T_true = np.zeros((16, 28, 28), dtype=int)
    # for i in range(4):
    #     for j in range(4):
    #         for k in range(7):
    #             a_idx = i*4+j
    #             b_idx = j*7+k
    #             c_idx = k*4+i
    #             T_true[a_idx, b_idx, c_idx] = 1
    # if not np.allclose(T, T_true):
    #     print("Decomposition failed")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
