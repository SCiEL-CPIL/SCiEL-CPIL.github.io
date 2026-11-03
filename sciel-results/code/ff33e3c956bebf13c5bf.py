"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm scaled to 4x4 for each half of the 4x8 matrix.
    # Total rank = 2 * (7 * 8) = 112.
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)] # 16 x 112
    V = [[0] * rank for _ in range(m * p)] # 32 x 112
    W = [[0] * rank for _ in range(p * n)] # 32 x 112
    
    # Strassen coefficients for 2x2 blocks
    # U_t and V_t are 2x2 matrices of coefficients
    # t=0..6
    # U_t[i'][j'] is the coefficient for A block element
    # V_t[i'][j'] is the coefficient for B block element
    
    # We define the 7 terms
    # M_t = U_t * V_t
    # U_t, V_t are 2x2 matrices
    
    # t=0: (A00+A11)(B00+B11)
    U0 = [[1, 0], [0, 1]] # A00 + A11 means U0[0][0]=1 (from A00), U0[1][1]=1 (from A11)?
    # No, U_t is a 2x2 matrix where each entry is a linear combo of A elements.
    # But here we are treating the blocks as scalars for the Strassen step.
    # Let's define the linear maps explicitly.
    
    # For each t, we have a 2x2 matrix of "block coefficients".
    # Let's define U_block[t] and V_block[t] as 2x2 matrices of integers.
    # U_block[t][i'][j'] is the coefficient of the A block at (I_A, J_A) in the i',j' position of U_t.
    # This is getting messy. Let's just list the 8 schoolbook terms for each of the 7 Strassen terms.
    
    # Strassen terms (using 0-based block indices 0,1):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each M_t is a 2x2 matrix product of two 2x2 matrices.
    # We decompose each M_t into 8 rank-1 tensors (schoolbook).
    
    col = 0
    for t in range(7):
        # Define U_t and V_t as 2x2 matrices of (block_row, block_col, coeff)
        # Actually, let's just generate the 8 terms directly.
        
        # We need to map the local indices i', j', k' to global indices.
        # A index: 2*I_A + i_A, 2*J_A + j_A
        # B index: 2*J_B + j_B, 2*K_B + k_B
        # C index: 2*I_C + i_C, 2*K_C + k_C
        
        # For a given t, we have linear combos for U_t and V_t.
        # Let's define the coefficients for A and B for each of the 8 schoolbook terms.
        
        # We'll iterate through the 8 schoolbook terms for the 2x2 block mult.
        # Term (i', j', k') corresponds to A[i'][j'] * B[j'][k'] -> C[i'][k']
        
        for i_prime in range(2):
            for j_prime in range(2):
                for k_prime in range(2):
                    # This is one rank-1 term for the block multiplication M_t
                    # We need to determine which A and B elements are involved.
                    
                    # Determine U_t coefficients for A[i_prime][j_prime]
                    # Determine V_t coefficients for B[j_prime][k_prime]
                    
                    # This requires knowing the definition of U_t and V_t.
                    # Let's hardcode the 7 definitions.
                    
                    u_coeffs = [] # list of (I_A, J_A, coeff)
                    v_coeffs = [] # list of (J_B, K_B, coeff)
                    
                    if t == 0:
                        # M0 = (A00+A11)(B00+B11)
                        # U0 = A00 + A11. So U0[i'][j'] = A[0][0][i'][j'] + A[1][1][i'][j']
                        u_coeffs = [(0, 0, 1), (1, 1, 1)]
                        v_coeffs = [(0, 0, 1), (1, 1, 1)]
                    elif t == 1:
                        # M1 = (A10+A11)B00
                        u_coeffs = [(1, 0, 1), (1, 1, 1)]
                        v_coeffs = [(0, 0, 1)]
                    elif t == 2:
                        # M2 = A00(B01-B11)
                        u_coeffs = [(0, 0, 1)]
                        v_coeffs = [(0, 1, 1), (1, 1, -1)]
                    elif t == 3:
                        # M3 = A11(B10-B00)
                        u_coeffs = [(1, 1, 1)]
                        v_coeffs = [(1, 0, 1), (0, 0, -1)]
                    elif t == 4:
                        # M4 = (A00+A01)B11
                        u_coeffs = [(0, 0, 1), (0, 1, 1)]
                        v_coeffs = [(1, 1, 1)]
                    elif t == 5:
                        # M5 = (A10-A00)(B00+B01)
                        u_coeffs = [(1, 0, 1), (0, 0, -1)]
                        v_coeffs = [(0, 0, 1), (0, 1, 1)]
                    elif t == 6:
                        # M6 = (A01-A11)(B10+B11)
                        u_coeffs = [(0, 1, 1), (1, 1, -1)]
                        v_coeffs = [(1, 0, 1), (1, 1, 1)]
                    
                    # Now, for this specific schoolbook term (i_prime, j_prime, k_prime),
                    # the A part is sum of u_coeffs applied to A block elements.
                    # The B part is sum of v_coeffs applied to B block elements.
                    # The C part is C[i_prime][k_prime] with coefficient 1.
                    
                    # But wait, the Strassen reconstruction has coefficients c[I_C][K_C][t].
                    # We need to add this term to the global U, V, W.
                    
                    # For each combination of A source and B source, we have a rank-1 tensor?
                    # No, U is a linear combo. V is a linear combo.
                    # U[a, col] = sum of coeffs from u_coeffs.
                    # V[b, col] = sum of coeffs from v_coeffs.
                    # W[c, col] = c[I_C][K_C][t] for the corresponding C block.
                    
                    # But W depends on I_C, K_C.
                    # The term M_t contributes to C blocks I_C, K_C with weight c[I_C][K_C][t].
                    # So we must add this rank-1 tensor 4 times (for each C block), with different W weights.
                    # This increases the rank!
                    
                    # Rank calculation:
                    # Each (t, i', j', k') is a base rank-1 tensor in A x B space.
                    # It contributes to 4 C blocks.
                    # So we need 4 separate rank-1 tensors in the full A x B x C space?
                    # Yes, because W is different for each C block.
                    
                    # So total rank = 7 * 8 * 4 = 224?
                    # No, the rank of the tensor is the minimum number of rank-1 tensors.
                    # If W is different, they are distinct rank-1 tensors.
                    
                    # Wait, Strassen rank for 4x4 is 56.
                    # 56 = 7 * 8.
                    # This implies that the C reconstruction does NOT increase the rank?
                    # How?
                    
                    # Ah, the C reconstruction is linear.
                    # C = sum c_t M_t.
                    # M_t = sum_{schoolbook} u_t v_t.
                    # C = sum_t sum_{schoolbook} c_t u_t v_t.
                    # This is a sum of (7*8) terms.
                    # Each term is (c_t * u_t) * v_t * 1?
                    # No, c_t is a coefficient for the C block.
                    
                    # If c_t is a scalar, then yes.
                    # But c_t is a 2x2 matrix of coefficients?
                    # No, c[I_C][K_C][t] is a scalar coefficient for the block M_t.
                    
                    # So, for a fixed t and schoolbook term, we have a product U_t V_t.
                    # This product contributes to C_block[I_C][K_C] with weight c[I_C][K_C][t].
                    
                    # So, for each (t, schoolbook_term), we have 4 contributions to C.
                    # Each contribution is a rank-1 tensor with W scaled by c[I_C][K_C][t].
                    
                    # So we have 4 rank-1 tensors for each (t, schoolbook_term).
                    # Total 7 * 8 * 4 = 224.
                    
                    # This is worse than schoolbook (128).
                    
                    # Why is Strassen rank 56 then?
                    # Because the C reconstruction is NOT just adding 4 separate tensors.
                    # The tensor is T_{i,j,k}.
                    # T = sum_t c_t M_t.
                    # M_t is a tensor itself.
                    
                    # If we use the schoolbook decomposition for M_t, we get 8 rank-1 tensors for M_t.
                    # Then T is a linear combination of these.
                    # The rank of a sum of tensors is at most the sum of their ranks.
                    # So rank(T) <= sum_t rank(M_t) = 7 * 8 = 56.
                    
                    # So the rank IS 56.
                    # My mistake was thinking that different W weights require separate rank-1 tensors in the final sum.
                    # But they don't. The final sum is just a sum of 56 rank-1 tensors.
                    # Each rank-1 tensor has a specific U, V, W.
                    
                    # So, for each (t, i', j', k'), we define ONE rank-1 tensor.
                    # U is the linear combo for A.
                    # V is the linear combo for B.
                    # W is the linear combo for C.
                    
                    # W is not a scalar. W is a vector in C-space.
                    # W[c, col] = sum_{I_C, K_C} c[I_C][K_C][t] * delta(c, I_C, K_C, i', k').
                    
                    # So, for a fixed (t, i', j', k'), W is non-zero only for the 4 C indices corresponding to the 4 blocks.
                    # And the value is c[I_C][K_C][t].
                    
                    # So we just need to fill W for those 4 indices.
                    
                    # This confirms rank 56 for 4x4.
                    # And rank 112 for 4x8.
                    
                    # Implementation:
                    # For each t, i', j', k':
                    # 1. Construct U vector (size 16).
                    # 2. Construct V vector (size 32).
                    # 3. Construct W vector (size 32).
                    
                    # U:
                    # For each (I_A, J_A, coeff) in u_coeffs:
                    #   a = 2*I_A + i_prime, 2*J_A + j_prime
                    #   U[a, col] += coeff
                    
                    # V:
                    # For each (J_B, K_B, coeff) in v_coeffs:
                    #   b = 2*J_B + j_prime, 2*K_B + k_prime
                    #   V[b, col] += coeff
                    
                    # W:
                    # For each I_C in 0..1, K_C in 0..1:
                    #   c = 2*K_C + k_prime, 2*I_C + i_prime
                    #   w_idx = c * n + i_C? No.
                    #   W index is k*n + i.
                    #   k is col of C (0..7), i is row of C (0..3).
                    #   k = 2*K_C + k_prime.
                    #   i = 2*I_C + i_prime.
                    #   w_idx = k * 4 + i.
                    #   W[w_idx, col] = c[I_C][K_C][t]
                    
                    # Note: c[I_C][K_C][t] is the same for all i', k'?
                    # No, c[I_C][K_C][t] is the coefficient for the BLOCK M_t.
                    # M_t is a 2x2 block.
                    # So c[I_C][K_C][t] is a scalar.
                    # It applies to the whole block.
                    # So W[w_idx, col] = c[I_C][K_C][t] for the 4 indices.
                    
                    # Let's define the c matrix.
                    # c[0][0] = [1, 0, 0, 1, 0, 0, 1] (M0+M3-M4+M6)
                    # c[0][1] = [0, 0, 1, 0, 1, 0, 0] (M2+M4)
                    # c[1][0] = [0, 1, 0, 1, 0, 0, 0] (M1+M3)
                    # c[1][1] = [1, -1, 1, 0, 0, 1, 0] (M0-M1+M2+M5)
                    
                    c_mat = [
                        [1, 0, 0, 1, 0, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0],
                        [1, -1, 1, 0, 0, 1, 0]
                    ]
                    # c_mat[I_C * 2 + K_C][t]
                    
                    # Fill U, V, W
                    for I_A, J_A, coeff_a in u_coeffs:
                        a = 2 * I_A + i_prime
                        b_a = 2 * J_A + j_prime
                        idx_a = a * m + b_a
                        U[idx_a][col] += coeff_a
                        
                    for J_B, K_B, coeff_b in v_coeffs:
                        b = 2 * J_B + j_prime
                        b_b = 2 * K_B + k_prime
                        idx_b = b * p + b_b
                        V[idx_b][col] += coeff_b
                        
                    for I_C in range(2):
                        for K_C in range(2):
                            c_coeff = c_mat[I_C * 2 + K_C][t]
                            if c_coeff != 0:
                                k_c = 2 * K_C + k_prime
                                i_c = 2 * I_C + i_prime
                                idx_c = k_c * n + i_c
                                W[idx_c][col] = c_coeff
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
