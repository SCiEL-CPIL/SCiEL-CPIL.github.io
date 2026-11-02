import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # Rank 1: 4x4x4 part (Rank 49)
    # Rank 2: 4x4x2 part (Rank 28)
    # Rank 3: 4x4x1 part (Rank 16)
    # Total Rank 93
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # We will build lists of (u_coeff, v_coeff, w_coeff) for each rank-1 term
    # u_coeff is a vector of length n*m
    # v_coeff is a vector of length m*p
    # w_coeff is a vector of length p*n
    
    terms = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        u_vec = [0.0] * U_rows
        v_vec = [0.0] * V_rows
        w_vec = [0.0] * W_rows
        for (i, j), val in u_dict.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_dict.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_dict.items():
            w_vec[k * n + i] = val
        terms.append((u_vec, v_vec, w_vec))
        
    # Strassen coefficients for 2x2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Outer Strassen for 4x4x4 (blocks are 2x2)
    # Block indices for A: (0,0), (0,1), (1,0), (1,1) -> corresponds to 2x2 blocks
    # Block indices for B: (0,0), (0,1), (1,0), (1,1)
    # Block indices for C: (0,0), (0,1), (1,0), (1,1)
    
    # Map block (I, J) to local indices (i, j) in 0..1
    # A[I, J] is a 2x2 matrix with entries A[I*2+i, J*2+j]
    
    # Outer Strassen terms s=1..7
    # A_s = sum alpha_s[I,J] A[I,J]
    # B_s = sum beta_s[J,K] B[J,K]
    # C = sum_s (A_s B_s) * C_s
    
    # We need to expand A_s and B_s into scalar A[i,j] and B[j,k].
    
    # Outer Strassen definitions (using 1-based indexing for clarity, will convert to 0-based)
    # s=1: A11+A22, B11+B22
    # s=2: A21+A22, B11
    # s=3: A11, B12-B22
    # s=4: A22, B21-B11
    # s=5: A11+A12, B22
    # s=6: A21-A11, B11+B12
    # s=7: A12-A22, B21+B22
    
    outer_strassen = [
        # s, A_coeffs (I,J), B_coeffs (J,K)
        (1, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        (2, {(1,0):1, (1,1):1}, {(0,0):1}),
        (3, {(0,0):1}, {(0,1):1, (1,1):-1}),
        (4, {(1,1):1}, {(1,0):1, (0,0):-1}),
        (5, {(0,0):1, (0,1):1}, {(1,1):1}),
        (6, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        (7, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1})
    ]
    
    # Inner Strassen for 2x2x2 (scalar)
    # Used to decompose the product of two 2x2 matrices (the blocks)
    # This applies to 4x4x4 and 4x4x2 (partially)
    
    inner_strassen = [
        # t, A_coeffs (i,j), B_coeffs (j,k), C_coeffs (k,i)
        (1, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1 -> C11, C22
        (2, {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # M2 -> C21
        (3, {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}), # M3 -> C12
        (4, {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4 -> C11, C21
        (5, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5 -> C11, C12
        (6, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6 -> C22
        (7, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1})  # M7 -> C11, C22
    ]
    
    # Correct Inner Strassen C coefficients:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's redefine inner_strassen with correct C contributions
    # Each term t contributes to specific C entries.
    
    inner_strassen_terms = [
        # t, A_coeffs, B_coeffs, C_contributions (k,i) -> coeff
        (1, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        (2, {(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        (3, {(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        (4, {(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        (5, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        (6, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        (7, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    ]
    
    # 1. Construct 4x4x4 part
    # For each outer term s, we form A_s and B_s as 2x2 matrices of coefficients for scalar A[i,j] and B[j,k].
    # Then we apply inner Strassen to this 2x2x2 problem.
    
    for s, a_blocks, b_blocks in outer_strassen:
        # Expand A_s into scalar coefficients a_scalar[i,j]
        # A_s[I,J] = sum a_scalar[I*2+i, J*2+j] * A[I*2+i, J*2+j]
        # Actually, A_s is a linear combination of A blocks.
        # A_s = sum_{I,J} a_blocks[I,J] A[I,J]
        # A[I,J] is a 2x2 matrix.
        # So A_s is a 2x2 matrix where entry (i,j) is sum_{I,J} a_blocks[I,J] A[I*2+i, J*2+j].
        
        a_scalar = {}
        for I in range(2):
            for J in range(2):
                if (I,J) in a_blocks:
                    coeff = a_blocks[(I,J)]
                    for i in range(2):
                        for j in range(2):
                            key = (I*2+i, J*2+j)
                            a_scalar[key] = a_scalar.get(key, 0) + coeff
        
        b_scalar = {}
        for J in range(2):
            for K in range(2):
                if (J,K) in b_blocks:
                    coeff = b_blocks[(J,K)]
                    for j in range(2):
                        for k in range(2):
                            key = (J*2+j, K*2+k)
                            b_scalar[key] = b_scalar.get(key, 0) + coeff
        
        # Now we have a_scalar (4x4) and b_scalar (4x4) representing the "effective" A and B for this term.
        # Wait, this is not quite right.
        # The product A_s B_s is a product of two 2x2 matrices (the blocks).
        # The inner Strassen algorithm computes this product using 7 scalar multiplications.
        # The "scalars" in the inner algorithm are the entries of the block matrices.
        # So we apply the inner Strassen coefficients to the entries of A_s and B_s.
        
        for t, a_inner, b_inner, c_inner in inner_strassen_terms:
            # a_inner maps (i,j) in 0..1 to coeff.
            # These (i,j) are the local indices within the 2x2 block.
            # The global indices are (I*2+i, J*2+j) ? No.
            # The inner algorithm operates on the 2x2 matrices A_s and B_s.
            # The entries of A_s are (i,j) where i,j in 0..1.
            # But A_s itself is composed of global A entries.
            
            # Let's trace one term.
            # Term (s,t).
            # U coefficients:
            # u[i,j] = sum_{I,J} a_blocks[I,J] * a_inner[i,j] ?
            # No.
            # A_s has entries A_s[i,j] = sum_{I,J} a_blocks[I,J] * A[I*2+i, J*2+j].
            # The inner algorithm forms linear combinations of A_s entries.
            # e.g., M1 uses A_s[0,0] + A_s[1,1].
            # A_s[0,0] = sum_{I,J} a_blocks[I,J] A[I*2+0, J*2+0].
            # A_s[1,1] = sum_{I,J} a_blocks[I,J] A[I*2+1, J*2+1].
            # So the coefficient for A[I*2+i, J*2+j] in the linear combination is
            # sum_{i',j'} a_inner[i',j'] * a_blocks[I,J] * delta(i, I*2+i') delta(j, J*2+j')?
            # No, i' is the local index.
            
            # Let's compute the global coefficients for U, V, W for this term.
            
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            
            # U: A entries
            # The inner algorithm uses linear combinations of A_s entries.
            # A_s[i', j'] = sum_{I,J} a_blocks[I,J] A[I*2+i', J*2+j']
            # The coefficient of A[I*2+i', J*2+j'] in the term is a_blocks[I,J] * a_inner[i', j']
            
            for I in range(2):
                for J in range(2):
                    if (I,J) in a_blocks:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                if (i_prime, j_prime) in a_inner:
                                    global_i = I*2 + i_prime
                                    global_j = J*2 + j_prime
                                    coeff = a_blocks[(I,J)] * a_inner[(i_prime, j_prime)]
                                    key = (global_i, global_j)
                                    u_coeffs[key] = u_coeffs.get(key, 0) + coeff
            
            # V: B entries
            for J in range(2):
                for K in range(2):
                    if (J,K) in b_blocks:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                if (j_prime, k_prime) in b_inner:
                                    global_j = J*2 + j_prime
                                    global_k = K*2 + k_prime
                                    coeff = b_blocks[(J,K)] * b_inner[(j_prime, k_prime)]
                                    key = (global_j, global_k)
                                    v_coeffs[key] = v_coeffs.get(key, 0) + coeff
            
            # W: C entries
            # The inner algorithm produces C_s entries.
            # C_s[k', i'] = sum_{t} c_inner_t[k', i'] * M_t
            # Here we are in term t, so we use c_inner[k', i'].
            # C_s is a 2x2 matrix of blocks.
            # C_s[k', i'] corresponds to global C[K*2+k', I*2+i']?
            # No, C_s is the result of A_s B_s.
            # A_s is 2x2 blocks. B_s is 2x2 blocks.
            # C_s is 2x2 blocks.
            # The block index is (K, I).
            # The local index within the block is (k', i').
            # So global C index is (K*2+k', I*2+i').
            
            # Wait, the W tensor index is (k, i).
            # C[k, i].
            
            # The inner algorithm computes the product of two 2x2 matrices (blocks).
            # The result is a 2x2 matrix (blocks).
            # Let the block indices be (K, I).
            # The entry C_s[K, I] is a 2x2 matrix.
            # The local entry (k', i') of C_s[K, I] is C[K*2+k', I*2+i'].
            
            # The inner Strassen algorithm computes C_s[K, I] using the c_inner coefficients.
            # c_inner maps (k', i') to coeff.
            
            for K in range(2):
                for I in range(2):
                    # The block C_s[K, I] is formed.
                    # The inner algorithm's output is the whole 2x2 matrix C_s.
                    # The coefficients c_inner are for the local indices (k', i').
                    
                    for k_prime in range(2):
                        for i_prime in range(2):
                            if (k_prime, i_prime) in c_inner:
                                global_k = K*2 + k_prime
                                global_i = I*2 + i_prime
                                coeff = c_inner[(k_prime, i_prime)]
                                key = (global_k, global_i)
                                w_coeffs[key] = w_coeffs.get(key, 0) + coeff
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # 2. Construct 4x4x2 part
    # Outer Strassen (7 terms).
    # Inner multiplication is 2x2x1 (Rank 4).
    # B is 4x2. Blocks are 2x1.
    # B[J, K] where J in 0..1, K in 0..1.
    # B[J, K] is a 2x1 matrix.
    
    # Inner 2x2x1 decomposition (Schoolbook Rank 4)
    # A is 2x2, B is 2x1, C is 2x1.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    
    inner_221_terms = [
        # A_coeffs, B_coeffs, C_coeffs
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(0,1):1}, {(1,0):1}, {(0,0):1}),
        ({(1,0):1}, {(0,0):1}, {(1,0):1}),
        ({(1,1):1}, {(1,0):1}, {(1,0):1})
    ]
    
    for s, a_blocks, b_blocks in outer_strassen:
        # Note: b_blocks for 4x4x2 are the same structure, but B is 4x2.
        # The outer Strassen structure is the same.
        
        a_scalar = {}
        for I in range(2):
            for J in range(2):
                if (I,J) in a_blocks:
                    coeff = a_blocks[(I,J)]
                    for i in range(2):
                        for j in range(2):
                            key = (I*2+i, J*2+j)
                            a_scalar[key] = a_scalar.get(key, 0) + coeff
        
        b_scalar = {}
        for J in range(2):
            for K in range(2):
                if (J,K) in b_blocks:
                    coeff = b_blocks[(J,K)]
                    for j in range(2):
                        for k in range(1): # B is 2x1, so k is 0
                            key = (J*2+j, K*2+k)
                            b_scalar[key] = b_scalar.get(key, 0) + coeff
        
        for a_inner, b_inner, c_inner in inner_221_terms:
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            
            for I in range(2):
                for J in range(2):
                    if (I,J) in a_blocks:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                if (i_prime, j_prime) in a_inner:
                                    global_i = I*2 + i_prime
                                    global_j = J*2 + j_prime
                                    coeff = a_blocks[(I,J)] * a_inner[(i_prime, j_prime)]
                                    key = (global_i, global_j)
                                    u_coeffs[key] = u_coeffs.get(key, 0) + coeff
            
            for J in range(2):
                for K in range(2):
                    if (J,K) in b_blocks:
                        for j_prime in range(2):
                            for k_prime in range(1):
                                if (j_prime, k_prime) in b_inner:
                                    global_j = J*2 + j_prime
                                    global_k = K*2 + k_prime
                                    coeff = b_blocks[(J,K)] * b_inner[(j_prime, k_prime)]
                                    key = (global_j, global_k)
                                    v_coeffs[key] = v_coeffs.get(key, 0) + coeff
            
            for K in range(2):
                for I in range(2):
                    for k_prime in range(1):
                        for i_prime in range(2):
                            if (k_prime, i_prime) in c_inner:
                                global_k = K*2 + k_prime
                                global_i = I*2 + i_prime
                                coeff = c_inner[(k_prime, i_prime)]
                                key = (global_k, global_i)
                                w_coeffs[key] = w_coeffs.get(key, 0) + coeff
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    # 3. Construct 4x4x1 part
    # Outer structure 2x2x1 (Rank 4).
    # Inner 2x2x1 (Rank 4).
    # Total 16 terms.
    
    outer_221_terms = [
        # A_coeffs (I,J), B_coeffs (J,K), C_coeffs (K,I)
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        ({(1,0):1}, {(1,0):1}, {(1,0):1}),
        ({(0,1):1}, {(0,0):1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1}, {(1,1):1})
    ]
    
    for a_outer, b_outer, c_outer in outer_221_terms:
        a_scalar = {}
        for I in range(2):
            for J in range(2):
                if (I,J) in a_outer:
                    coeff = a_outer[(I,J)]
                    for i in range(2):
                        for j in range(2):
                            key = (I*2+i, J*2+j)
                            a_scalar[key] = a_scalar.get(key, 0) + coeff
        
        b_scalar = {}
        for J in range(2):
            for K in range(1):
                if (J,K) in b_outer:
                    coeff = b_outer[(J,K)]
                    for j in range(2):
                        for k in range(1):
                            key = (J*2+j, K*2+k)
                            b_scalar[key] = b_scalar.get(key, 0) + coeff
        
        for a_inner, b_inner, c_inner in inner_221_terms:
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            
            for I in range(2):
                for J in range(2):
                    if (I,J) in a_outer:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                if (i_prime, j_prime) in a_inner:
                                    global_i = I*2 + i_prime
                                    global_j = J*2 + j_prime
                                    coeff = a_outer[(I,J)] * a_inner[(i_prime, j_prime)]
                                    key = (global_i, global_j)
                                    u_coeffs[key] = u_coeffs.get(key, 0) + coeff
            
            for J in range(2):
                for K in range(1):
                    if (J,K) in b_outer:
                        for j_prime in range(2):
                            for k_prime in range(1):
                                if (j_prime, k_prime) in b_inner:
                                    global_j = J*2 + j_prime
                                    global_k = K*2 + k_prime
                                    coeff = b_outer[(J,K)] * b_inner[(j_prime, k_prime)]
                                    key = (global_j, global_k)
                                    v_coeffs[key] = v_coeffs.get(key, 0) + coeff
            
            for K in range(1):
                for I in range(2):
                    for k_prime in range(1):
                        for i_prime in range(2):
                            if (k_prime, i_prime) in c_inner:
                                global_k = K*2 + k_prime
                                global_i = I*2 + i_prime
                                coeff = c_outer[(K,I)] * c_inner[(k_prime, i_prime)]
                                key = (global_k, global_i)
                                w_coeffs[key] = w_coeffs.get(key, 0) + coeff
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    rank = len(terms)
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]
    
    for c, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i in range(U_rows):
            U[i][c] = u_vec[i]
        for j in range(V_rows):
            V[j][c] = v_vec[j]
        for k in range(W_rows):
            W[k][c] = w_vec[k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
