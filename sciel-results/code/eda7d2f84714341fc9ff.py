"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition: split 6x6 into 2x2 blocks of 3x3
    # 7 Strassen steps, each with 27 schoolbook terms for 3x3 multiplication
    rank_initial = 7 * 27
    U = [[0.0] * rank_initial for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank_initial for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank_initial for _ in range(p * n)]  # 36 x 189
    
    # Block indices: A00=0, A01=1, A10=2, A11=3
    # B blocks: B00=0, B01=1, B10=2, B11=3
    # C blocks: C00=0, C01=1, C10=2, C11=3
    
    # Strassen steps: (A_coeffs, B_coeffs)
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),  # M1
        ({2: 1, 3: 1}, {0: 1}),          # M2
        ({0: 1}, {1: 1, 3: -1}),         # M3
        ({3: 1}, {2: 1, 0: -1}),         # M4
        ({0: 1, 1: 1}, {3: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),   # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),   # M7
    ]
    
    c_contrib = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: -1, 2: 1},
        {1: 1},
        {0: 1},
        {0: -1, 3: 1},
    ]
    
    for s in range(7):
        a_coeffs = steps[s][0]
        b_coeffs = steps[s][1]
        c_coeffs = c_contrib[s]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = s * 27 + i * 9 + j * 3 + k
                    
                    for a_block, a_sign in a_coeffs.items():
                        a_row = (a_block // 2) * 3 + i
                        a_col = (a_block % 2) * 3 + j
                        U[a_row * m + a_col][col] += float(a_sign)
                    
                    for b_block, b_sign in b_coeffs.items():
                        b_row = (b_block // 2) * 3 + j
                        b_col = (b_block % 2) * 3 + k
                        V[b_row * p + b_col][col] += float(b_sign)
                    
                    for c_block, c_sign in c_coeffs.items():
                        c_row = (c_block // 2) * 3 + i
                        c_col = (c_block % 2) * 3 + k
                        W[c_col * n + c_row][col] += float(c_sign)
    
    # Convert to numpy arrays
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check deadline
    if time.time() > DEADLINE - 10:
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linearly dependent terms
    # The matrix multiplication tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Each term r defines a rank-1 tensor u_r ⊗ v_r ⊗ w_r
    # We can check if any subset of terms is linearly dependent by forming
    # a matrix where each column is the vectorized rank-1 tensor
    
    # Vectorize each rank-1 term into a 216-dim vector
    # Term r: vec(u_r ⊗ v_r ⊗ w_r)
    # This is the Kronecker product u_r ⊗ v_r ⊗ w_r flattened
    
    # For efficiency, form a 216 x 189 matrix M where M[:, r] = vec(u_r ⊗ v_r ⊗ w_r)
    # Then compute the rank of M. If rank < 189, we can reduce.
    
    # But this is expensive. Instead, let's use a simpler approach:
    # Check if the columns of U, V, W are linearly independent in a way that
    # allows rank reduction.
    
    # Actually, a better approach: use the fact that the tensor is defined by
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We can try to find a lower-rank decomposition by solving:
    # Given T, find U', V', W' with rank r' < 189 such that T = sum_{r'<r} U'[:,r'] ⊗ V'[:,r'] ⊗ W'[:,r']
    
    # This is NP-hard in general, but for small cases we can try.
    
    # Alternative: use the fact that 189 = 7 * 27, and check if the 7 Strassen
    # steps can be merged. For example, if two steps produce the same combination
    # of A and B blocks, they can be merged.
    
    # Let's try a different structural approach: use a 3x3 block decomposition
    # with 2x2 Strassen inside each block. This gives 9 * 7 * 4 = 252, which is
    # worse, but let's see if we can do better by sharing computations.
    
    # Actually, let's just return the Strassen decomposition. The rank is 189,
    # which is better than the schoolbook 216.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
