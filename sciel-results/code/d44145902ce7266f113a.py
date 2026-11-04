import numpy as np

def construct():
    n, m, p = 3, 4, 8
    nm = n * m
    mp = m * p
    pn = p * n
    
    # We will build the decomposition for M_2,2,2 (Strassen)
    # and combine it.
    
    # Strassen for 2x2x2
    # Indices: i in 0,1; j in 0,1; k in 0,1
    # T[i, j, k]
    
    # The 7 rank-1 terms for Strassen
    # We need to express C = A * B as sum of 7 terms M_r * U_r * V_r * W_r
    # where M_r are the intermediate products.
    
    # Actually, we just need U, V, W matrices such that
    # sum_r U[:,r] V[:,r] W[:,r]^T = T_true
    
    # Let's construct the full U, V, W for 3x4x8 directly using the logic.
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, coeff):
        # i: 0..n-1, j: 0..m-1, k: 0..p-1
        # This adds a term coeff * e_i (in A) * e_j (in B) * e_k (in C)
        # Wait, the tensor T[i, j, k] corresponds to A[i,j] * B[j,k] -> C[i,k]
        # The rank-1 tensor is u * v * w
        # u has 1 at index i*m + j
        # v has 1 at index j*p + k
        # w has 1 at index k*n + i
        
        u = [0.0] * nm
        v = [0.0] * mp
        w = [0.0] * pn
        
        u[i*m + j] = coeff
        v[j*p + k] = coeff
        w[k*n + i] = 1.0 / (coeff*coeff) if coeff != 0 else 0
        
        # This is not quite right. The product is u_r v_r w_r.
        # If we set u_r = 1, v_r = 1, w_r = 1, the term is 1.
        # If we need a coefficient, we can put it in any of them.
        # Let's put the coefficient in U.
        u[i*m + j] = coeff
        v[j*p + k] = 1.0
        w[k*n + i] = 1.0
        
        U.append(u)
        V.append(v)
        W.append(w)

    # This approach is too slow if we just iterate.
    # We need to use the Strassen structure to reduce the number of terms.
    
    # Let's implement the recursive construction.
    
    # 1. M_2,2,2 Strassen
    # We define the 7 terms.
    # The inputs are A_00, A_01, A_10, A_11
    # B_00, B_01, B_10, B_11
    # Outputs C_00, C_01, C_10, C_11
    
    # The terms are linear combinations of A and B.
    # M1 = (A00+A11)(B00+B11)
    # This involves 4 terms: A00B00, A00B11, A11B00, A11B11
    # In the tensor decomposition, we don't form M1 explicitly.
    # We form the sum of rank-1 tensors.
    # The rank-1 tensor for A00B00 is e_00 (in A) * e_00 (in B) * e_00 (in C)
    
    # Strassen's algorithm works by reducing the number of multiplications.
    # The tensor rank is the number of multiplications.
    # So we need to define the 7 "multiplications" as rank-1 tensors?
    # No. A multiplication $M = A_{sub} B_{sub}$ is not a rank-1 tensor in the global space.
    # It is a sub-tensor.
    
    # The decomposition is:
    # T = sum_{r=1}^7 U_r V_r W_r
    # where U_r, V_r, W_r are vectors.
    # This implies that the "multiplication" $M_r$ is represented by the outer product of the coefficients.
    
    # For Strassen:
    # M1 = (A00+A11)(B00+B11)
    # This term contributes to C.
    # C00 += M1
    # C01 += -M1? No.
    
    # Let's look at the standard tensor decomposition for Strassen.
    # The 7 rank-1 tensors are:
    # 1. (A00+A11) (B00+B11) -> contributes to C00, C11, etc.
    # Actually, the rank-1 tensors are defined on the spaces of A, B, C.
    
    # U1: (1,0,1,0) for A? No, U1 is a vector in R^4 (for 2x2 A).
    # U1 = [1, 0, 1, 0] (coefficients for A00, A01, A10, A11)
    # V1 = [1, 1, 0, 0] (coefficients for B00, B01, B10, B11)
    # W1 = [1, 0, 0, 1] (coefficients for C00, C01, C10, C11)
    
    # Let's verify:
    # Term 1: U1 * V1 * W1
    # Sum over indices:
    # U1_a V1_b W1_c
    # a=0 (A00), b=0 (B00), c=0 (C00): 1*1*1 = 1
    # a=0, b=1, c=0: 1*1*1 = 1
    # a=2 (A11), b=0, c=0: 1*1*1 = 1
    # a=2, b=1, c=0: 1*1*1 = 1
    # a=0, b=0, c=3 (C11): 1*1*1 = 1
    # ...
    
    # This seems correct.
    
    # We will build the U, V, W matrices for the 2x2x2 case first.
    
    def strassen_222():
        # Returns U, V, W lists of length 7
        # Each is a list of 4 floats
        U_s = []
        V_s = []
        W_s = []
        
        # 1. (A00+A11)(B00+B11)
        # U: [1, 0, 0, 1]
        # V: [1, 1, 0, 0]
        # W: [1, 0, 0, 1]
        U_s.append([1, 0, 0, 1])
        V_s.append([1, 1, 0, 0])
        W_s.append([1, 0, 0, 1])
        
        # 2. (A10+A11)B00
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: [0, 0, 1, 0]
        U_s.append([0, 0, 1, 1])
        V_s.append([1, 0, 0, 0])
        W_s.append([0, 0, 1, 0])
        
        # 3. A00(B01-B11)
        # U: [1, 0, 0, 0]
        # V: [0, 1, 0, -1]
        # W: [0, 1, 0, 0]
        U_s.append([1, 0, 0, 0])
        V_s.append([0, 1, 0, -1])
        W_s.append([0, 1, 0, 0])
        
        # 4. A11(B10-B00)
        # U: [0, 0, 0, 1]
        # V: [0, 0, 1, -1]
        # W: [0, 0, 1, 0]
        U_s.append([0, 0, 0, 1])
        V_s.append([0, 0, 1, -1])
        W_s.append([0, 0, 1, 0])
        
        # 5. (A00+A01)B11
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: [0, 1, 0, 0]
        U_s.append([1, 1, 0, 0])
        V_s.append([0, 0, 0, 1])
        W_s.append([0, 1, 0, 0])
        
        # 6. (A10-A00)(B00+B01)
        # U: [-1, 0, 1, 0]
        # V: [1, 1, 0, 0]
        # W: [0, 0, 1, 0]
        U_s.append([-1, 0, 1, 0])
        V_s.append([1, 1, 0, 0])
        W_s.append([0, 0, 1, 0])
        
        # 7. (A01-A11)(B10+B11)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # W: [0, 0, 0, 1]
        U_s.append([0, 1, 0, -1])
        V_s.append([0, 0, 1, 1])
        W_s.append([0, 0, 0, 1])
        
        return U_s, V_s, W_s

    # Now we need to expand this to 3x4x8.
    # We use the splitting logic.
    
    # We will maintain U, V, W as lists of lists (columns).
    # Start with empty.
    
    final_U = []
    final_V = []
    final_W = []
    
    # Function to add a scaled and shifted version of a base decomposition
    # base_U, base_V, base_W are for a sub-problem of size n_s, m_s, p_s
    # offset_i, offset_j, offset_k are the starting indices in the global space
    
    def add_decomp(base_U, base_V, base_W, n_s, m_s, p_s, off_i, off_j, off_k, scale=1.0):
        for r in range(len(base_U)):
            u = [0.0] * (n_s * m_s)
            v = [0.0] * (m_s * p_s)
            w = [0.0] * (p_s * n_s)
            
            for a in range(n_s):
                for b in range(m_s):
                    u[a*m_s + b] = base_U[r][a*m_s + b] * scale
            for b in range(m_s):
                for c in range(p_s):
                    v[b*p_s + c] = base_V[r][b*p_s + c] * scale
            for c in range(p_s):
                for a in range(n_s):
                    w[c*n_s + a] = base_W[r][c*n_s + a] * scale
            
            # Embed into global space
            U_g = [0.0] * (3 * 4)
            V_g = [0.0] * (4 * 8)
            W_g = [0.0] * (8 * 3)
            
            for a in range(n_s):
                for b in range(m_s):
                    U_g[(off_i + a)*4 + (off_j + b)] = u[a*m_s + b]
            for b in range(m_s):
                for c in range(p_s):
                    V_g[(off_j + b)*8 + (off_k + c)] = v[b*p_s + c]
            for c in range(p_s):
                for a in range(n_s):
                    W_g[(off_k + c)*3 + (off_i + a)] = w[c*n_s + a]
                    
            final_U.append(U_g)
            final_V.append(V_g)
            final_W.append(W_g)

    # 1. Handle i=2 (M_1,4,8)
    # Rank 32.
    # This is just schoolbook for 1x4x8.
    # We can generate it directly.
    for j in range(4):
        for k in range(8):
            U_g = [0.0] * 12
            V_g = [0.0] * 32
            W_g = [0.0] * 24
            U_g[2*4 + j] = 1.0
            V_g[j*8 + k] = 1.0
            W_g[k*3 + 2] = 1.0
            final_U.append(U_g)
            final_V.append(V_g)
            final_W.append(W_g)
            
    # 2. Handle i in {0,1} (M_2,4,8)
    # Rank 56.
    # Decompose M_2,4,8 into 2 * M_2,4,4 (split p)
    # Then M_2,4,4 into 2 * M_2,2,4 (split m)
    # Then M_2,2,4 into 2 * M_2,2,2 (split p)
    # Then M_2,2,2 is Strassen.
    
    # Base case: M_2,2,2 Strassen
    U_s, V_s, W_s = strassen_222()
    
    # Level 1: M_2,2,4
    # Split p=4 into p1=2, p2=2
    # Two copies of M_2,2,2
    U_224 = []
    V_224 = []
    W_224 = []
    # Copy for k_offset 0
    for r in range(7):
        U_g = [0.0] * 4
        V_g = [0.0] * 8
        W_g = [0.0] * 8
        for a in range(2):
            for b in range(2):
                U_g[a*2 + b] = U_s[r][a*2 + b]
        for b in range(2):
            for c in range(2):
                V_g[b*4 + c] = V_s[r][b*2 + c]
        for c in range(2):
            for a in range(2):
                W_g[c*2 + a] = W_s[r][c*2 + a]
        U_224.append(U_g)
        V_224.append(V_g)
        W_224.append(W_g)
        
    # Copy for k_offset 2
    for r in range(7):
        U_g = [0.0] * 4
        V_g = [0.0] * 8
        W_g = [0.0] * 8
        for a in range(2):
            for b in range(2):
                U_g[a*2 + b] = U_s[r][a*2 + b]
        for b in range(2):
            for c in range(2):
                V_g[b*4 + (2+c)] = V_s[r][b*2 + c]
        for c in range(2):
            for a in range(2):
                W_g[(2+c)*2 + a] = W_s[r][c*2 + a]
        U_224.append(U_g)
        V_224.append(V_g)
        W_224.append(W_g)
        
    # Level 2: M_2,4,4
    # Split m=4 into m1=2, m2=2
    U_244 = []
    V_244 = []
    W_244 = []
    # Copy for j_offset 0
    for r in range(14):
        U_g = [0.0] * 8
        V_g = [0.0] * 16
        W_g = [0.0] * 8
        for a in range(2):
            for b in range(2):
                U_g[a*4 + b] = U_224[r][a*2 + b]
        for b in range(2):
            for c in range(4):
                V_g[b*4 + c] = V_224[r][b*2 + c]
        for c in range(4):
            for a in range(2):
                W_g[c*2 + a] = W_224[r][c*2 + a]
        U_244.append(U_g)
        V_244.append(V_g)
        W_244.append(W_g)
        
    # Copy for j_offset 2
    for r in range(14):
        U_g = [0.0] * 8
        V_g = [0.0] * 16
        W_g = [0.0] * 8
        for a in range(2):
            for b in range(2):
                U_g[a*4 + (2+b)] = U_224[r][a*2 + b]
        for b in range(2):
            for c in range(4):
                V_g[(2+b)*4 + c] = V_224[r][b*2 + c]
        for c in range(4):
            for a in range(2):
                W_g[c*2 + a] = W_224[r][c*2 + a]
        U_244.append(U_g)
        V_244.append(V_g)
        W_244.append(W_g)
        
    # Level 3: M_2,4,8
    # Split p=8 into p1=4, p2=4
    U_248 = []
    V_248 = []
    W_248 = []
    # Copy for k_offset 0
    for r in range(28):
        U_g = [0.0] * 8
        V_g = [0.0] * 32
        W_g = [0.0] * 16
        for a in range(2):
            for b in range(4):
                U_g[a*4 + b] = U_244[r][a*4 + b]
        for b in range(4):
            for c in range(4):
                V_g[b*8 + c] = V_244[r][b*4 + c]
        for c in range(4):
            for a in range(2):
                W_g[c*2 + a] = W_244[r][c*2 + a]
        U_248.append(U_g)
        V_248.append(V_g)
        W_248.append(W_g)
        
    # Copy for k_offset 4
    for r in range(28):
        U_g = [0.0] * 8
        V_g = [0.0] * 32
        W_g = [0.0] * 16
        for a in range(2):
            for b in range(4):
                U_g[a*4 + b] = U_244[r][a*4 + b]
        for b in range(4):
            for c in range(4):
                V_g[b*8 + (4+c)] = V_244[r][b*4 + c]
        for c in range(4):
            for a in range(2):
                W_g[(4+c)*2 + a] = W_244[r][c*2 + a]
        U_248.append(U_g)
        V_248.append(V_g)
        W_248.append(W_g)
        
    # Now embed M_2,4,8 into M_3,4,8 at i_offset 0
    for r in range(56):
        U_g = [0.0] * 12
        V_g = [0.0] * 32
        W_g = [0.0] * 24
        for a in range(2):
            for b in range(4):
                U_g[a*4 + b] = U_248[r][a*4 + b]
        for b in range(4):
            for c in range(8):
                V_g[b*8 + c] = V_248[r][b*8 + c]
        for c in range(8):
            for a in range(2):
                W_g[c*3 + a] = W_248[r][c*2 + a]
        final_U.append(U_g)
        final_V.append(V_g)
        final_W.append(W_g)
        
    # Convert to lists of lists
    U = [col for col in zip(*final_U)] if final_U else []
    V = [col for col in zip(*final_V)] if final_V else []
    W = [col for col in zip(*final_W)] if final_W else []
    
    # The interface expects U, V, W to be (nm, rank), (mp, rank), (pn, rank)
    # My final_U is a list of rows? No, final_U is a list of columns (vectors).
    # U_g is a vector of size 12.
    # So final_U is a list of 88 vectors of size 12.
    # We need to transpose to get (12, 88).
    
    U_mat = np.array(final_U).T
    V_mat = np.array(final_V).T
    W_mat = np.array(final_W).T
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

