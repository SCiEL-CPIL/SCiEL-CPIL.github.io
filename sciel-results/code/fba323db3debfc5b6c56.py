import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by summing contributions from different block interactions.
    # The total rank will be the sum of ranks of the parts.
    
    # Part 1: 4x4x4 Strassen
    # Part 2: 4x1x4 Outer Product
    # Part 3: 4x4x1 Mat-Vec
    # Part 4: 1x4x4 Vec-Mat
    # Part 5: 4x1x1 Vec-Scalar
    # Part 6: 1x4x1 Scalar-Vec
    # Part 7: 1x1x1 Scalar
    
    # We will accumulate U, V, W columns.
    # Start with empty lists, then concatenate.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a decomposition term
    # term_u: index in U (i*m+j)
    # term_v: index in V (j*p+k)
    # term_w: index in W (k*n+i)
    # val: coefficient (1 or -1 or 0.5 etc)
    
    def add_term(u_idx, v_idx, w_idx, val):
        # We will store these as lists of tuples for now, then convert to matrices
        pass

    # Actually, it's easier to just build the matrices directly if we know the rank.
    # But the rank is 110.
    # Let's define the rank and fill zeros.
    
    # Rank calculation:
    # R_444 = 49
    # R_414 = 16
    # R_441 = 16
    # R_144 = 16
    # R_411 = 4
    # R_141 = 4
    # R_111 = 1
    total_rank = 49 + 16 + 16 + 16 + 4 + 4 + 1
    total_rank = 106
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # --- 1. 4x4x4 Strassen ---
    # Indices 0..3 for i, j, k
    # We need the Strassen decomposition for 4x4.
    # This is composed of 7 block multiplications of 2x2.
    # Each 2x2 block multiplication uses Strassen (rank 7).
    
    # Let's define the 2x2 Strassen decomposition.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to global indices.
    # For a 2x2 block at position (bi, bj) in the 4x4 matrix:
    # Global i = bi*2 + li, Global j = bj*2 + lj
    # Global k = bk*2 + lk
    
    # The 4x4 matrix is partitioned into 4 2x2 blocks:
    # (0,0), (0,1), (1,0), (1,1)
    
    # We apply Strassen to the block matrix.
    # This means we compute 7 "super-multiplications" of 2x2 matrices.
    # Each super-multiplication is a linear combination of the block elements.
    
    # Let's iterate over the 7 Strassen steps for the 4x4 level.
    # For each step, we have a coefficient for each 2x2 block of A and B.
    # Then we multiply those linear combinations (which are 2x2 matrices) using 2x2 Strassen.
    
    # This is getting complex to nest.
    # Alternative: Just use the known rank 49 decomposition for 4x4 directly?
    # Or, simpler: Use schoolbook for 4x4? Rank 64.
    # 64 + 16 + 16 + 16 + 4 + 4 + 1 = 121.
    # 121 is worse than 125? No, 121 < 125.
    # But 106 is better.
    
    # Let's stick to 106.
    # We need to implement the nested Strassen.
    
    # 2x2 Strassen Terms (local indices 0,1 for row/col in the 2x2 block)
    # U_local: (2*2, 7)
    # V_local: (2*2, 7)
    # W_local: (2*2, 7)
    
    # Let's hardcode the 2x2 Strassen decomposition.
    # U[li*2 + lj, t]
    # V[lj*2 + lk, t]
    # W[lk*2 + li, t]
    
    def get_2x2_strassen():
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # a=U[0,0], d=U[1,1] -> U[0,0]=1, U[1,1]=1
        # e=V[0,0], h=V[1,1] -> V[0,0]=1, V[1,1]=1
        # W[0,0] += 1, W[1,1] += 1, W[0,1] -= 1, W[1,0] -= 1
        u[0,0]=1; u[1,1]=1
        v[0,0]=1; v[1,1]=1
        w[0,0]=1; w[1,1]=1; w[0,1]=-1; w[1,0]=-1
        
        # M2 = (c+d)e
        # c=U[1,0], d=U[1,1]
        # e=V[0,0]
        u[1,0]=1; u[1,1]=1
        v[0,0]=1
        w[1,0]=1
        
        # M3 = a(f-h)
        # a=U[0,0]
        # f=V[0,1], h=V[1,1]
        u[0,0]=1
        v[0,1]=1; v[1,1]=-1
        w[0,1]=1
        
        # M4 = d(g-e)
        # d=U[1,1]
        # g=V[1,0], e=V[0,0]
        u[1,1]=1
        v[1,0]=1; v[0,0]=-1
        w[1,1]=1; w[0,0]=-1
        
        # M5 = (a+b)h
        # a=U[0,0], b=U[0,1]
        # h=V[1,1]
        u[0,0]=1; u[0,1]=1
        v[1,1]=1
        w[0,0]=-1; w[1,0]=1
        
        # M6 = (c-a)(e+f)
        # c=U[1,0], a=U[0,0]
        # e=V[0,0], f=V[0,1]
        u[1,0]=1; u[0,0]=-1
        v[0,0]=1; v[0,1]=1
        w[1,1]=1
        
        # M7 = (b-d)(g+h)
        # b=U[0,1], d=U[1,1]
        # g=V[1,0], h=V[1,1]
        u[0,1]=1; u[1,1]=-1
        v[1,0]=1; v[1,1]=1
        w[0,0]=1; w[1,1]=-1
        
        return u, v, w

    u2, v2, w2 = get_2x2_strassen()
    
    # Now build 4x4 Strassen using 2x2 Strassen.
    # The 4x4 Strassen has 7 steps.
    # Each step involves linear combinations of the four 2x2 blocks of A and B.
    # Let A_blocks be A00, A01, A10, A11.
    # Let B_blocks be B00, B01, B10, B11.
    
    # Strassen formulas for block matrices:
    # S1 = (A00 + A11) * (B00 + B11)
    # S2 = (A10 + A11) * B00
    # S3 = A00 * (B01 - B11)
    # S4 = A11 * (B10 - B00)
    # S5 = (A00 + A01) * B11
    # S6 = (A10 - A00) * (B00 + B01)
    # S7 = (A01 - A11) * (B10 + B11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each Si is a product of two 2x2 matrices (which are linear combinations of the original 2x2 blocks).
    # We can compute Si using the 2x2 Strassen decomposition.
    # However, the inputs to the 2x2 Strassen are not the original elements, but the linear combinations.
    # This means we need to "expand" the linear combinations into the global indices.
    
    # For each Strassen step s (1..7):
    # We have a linear combination of A blocks: A_s = sum_{bi,bj} coeff_A[bi,bj] * A_{bi,bj}
    # And B_s = sum_{bk,bj} coeff_B[bk,bj] * B_{bk,bj}
    # Then C_s = A_s * B_s is computed by 2x2 Strassen.
    # The 2x2 Strassen introduces 7 intermediate multiplications.
    # Each intermediate multiplication involves products of elements from A_s and B_s.
    # So we need to map the 2x2 Strassen indices back to the global 4x4 indices.
    
    # This is doable but tedious.
    # Given the time, maybe I should just use the schoolbook 4x4 (rank 64) to be safe and simple?
    # 64 + 16 + 16 + 16 + 4 + 4 + 1 = 121.
    # 121 is an improvement over 125.
    # Implementing nested Strassen correctly is error-prone.
    
    # Let's use Schoolbook for the 4x4 part.
    # Rank 64.
    
    # 1. 4x4x4 Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*5+i, col] = 1
                col += 1
    
    # 2. 4x1x4 Outer Product
    # A[i,4] * B[4,k] -> C[i,k]
    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1
            V[4*5+k, col] = 1
            W[k*5+i, col] = 1
            col += 1
            
    # 3. 4x4x1 Mat-Vec
    # A[i,j] * B[j,4] -> C[i,4]
    for i in range(4):
        for j in range(4):
            U[i*5+j, col] = 1
            V[j*5+4, col] = 1
            W[4*5+i, col] = 1
            col += 1
            
    # 4. 1x4x4 Vec-Mat
    # A[4,j] * B[j,k] -> C[4,k]
    for j in range(4):
        for k in range(4):
            U[4*5+j, col] = 1
            V[j*5+k, col] = 1
            W[k*5+4, col] = 1
            col += 1
            
    # 5. 4x1x1 Vec-Scalar
    # A[i,4] * B[4,4] -> C[i,4]
    for i in range(4):
        U[i*5+4, col] = 1
        V[4*5+4, col] = 1
        W[4*5+i, col] = 1
        col += 1
        
    # 6. 1x4x1 Scalar-Vec
    # A[4,4] * B[4,k] -> C[4,k]
    for k in range(4):
        U[4*5+4, col] = 1
        V[4*5+k, col] = 1
        W[k*5+4, col] = 1
        col += 1
        
    # 7. 1x1x1 Scalar
    U[4*5+4, col] = 1
    V[4*5+4, col] = 1
    W[4*5+4, col] = 1
    col += 1
    
    # Check if col == total_rank
    # 64 + 16 + 16 + 16 + 4 + 4 + 1 = 121
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
