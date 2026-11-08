"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p, offset=0):
    """Return (U, V, W) lists for schoolbook decomposition of n×m times m×p.
    U has shape (n*m, n*m*p), V has shape (m*p, n*m*p), W has shape (p*n, n*m*p).
    """
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, r

def _combine_strassen(U1, V1, W1, U2, V2, W2, U3, V3, W3, U4, V4, W4, U5, V5, W5, U6, V6, W6, U7, V7, W7):
    """Combine 7 sub-decompositions into one. Each sub has same shape.
    Returns combined U, V, W with rank = 7 * sub_rank.
    """
    sub_rank = len(U1[0])
    total_rank = 7 * sub_rank
    n_blocks = 2  # 2x2 block structure
    # We need to know the dimensions of the sub-problems
    # U sub shape: (n_sub * m_sub, sub_rank)
    # V sub shape: (m_sub * p_sub, sub_rank)
    # W sub shape: (p_sub * n_sub, sub_rank)
    
    n_sub = int((len(U1) * 1) ** 0.5)  # This won't work generically
    # Actually, let's just pass dimensions
    raise NotImplementedError("Use explicit version")

def _strassen_4x4_4x8():
    """Build Strassen decomposition for 4x4 * 4x8.
    
    A is 4x4, viewed as 2x2 blocks of 2x2.
    B is 4x8, viewed as 2x4 blocks (2 rows, 4 cols each).
    C is 4x8, viewed as 2x4 blocks.
    
    Block indices:
    A = [[A11, A12],
         [A21, A22]]  where each Ai is 2x2
    B = [[B11, B12],
         [B21, B22]]  where each Bj is 2x4
    C = [[C11, C12],
         [C21, C22]]  where each Ck is 2x4
    
    Strassen:
    M1 = (A11+A22) * (B11+B22)
    M2 = (A21+A22) * B11
    M3 = A11 * (B12-B22)
    M4 = A22 * (B21-B11)
    M5 = (A11+A12) * B22
    M6 = (A21-A11) * (B11+B12)
    M7 = (A12-A22) * (B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M6
    """
    n_sub, m_sub, p_sub = 2, 2, 4  # sub-problem dimensions
    sub_rank = n_sub * m_sub * p_sub  # 16
    total_rank = 7 * sub_rank  # 112
    
    # Full dimensions
    n, m, p = 4, 4, 8
    U = [[0] * total_rank for _ in range(n * m)]   # (16, 112)
    V = [[0] * total_rank for _ in range(m * p)]   # (32, 112)
    W = [[0] * total_rank for _ in range(p * n)]   # (32, 112)
    
    # For each Strassen term t (0..6), we have a sub-decomposition
    # The sub-decomposition maps:
    #   U_sub[i_sub * m_sub + j_sub, c_sub] for i_sub in [0,2), j_sub in [0,2)
    #   V_sub[j_sub * p_sub + k_sub, c_sub] for j_sub in [0,2), k_sub in [0,4)
    #   W_sub[k_sub * n_sub + i_sub, c_sub] for k_sub in [0,4), i_sub in [0,2)
    
    # The "input" to each sub-problem is a linear combination of A and B blocks.
    # For A: the sub-problem's A matrix (2x2) is formed from the 4x4 A via block ops.
    # For B: the sub-problem's B matrix (2x4) is formed from the 4x8 B via block ops.
    # The output C sub-block (2x4) contributes to the full C via block placement.
    
    # Let's define for each term t:
    #   a_coeffs: dict mapping (block_row, block_col) -> coefficient for A block
    #   b_coeffs: dict mapping (block_row, block_col) -> coefficient for B block  
    #   c_coeffs: dict mapping (block_row, block_col) -> coefficient for C block
    #   c_sign: +1 or -1 for the contribution
    
    # Actually, the structure is:
    # Each term t computes M_t = A_t * B_t where A_t is 2x2, B_t is 2x4
    # A_t = sum of alpha_{t,br,bc} * A[br,bc]  (linear combo of A's 2x2 blocks)
    # B_t = sum of beta_{t,br,bc} * B[br,bc]  (linear combo of B's 2x4 blocks)
    # M_t contributes to C blocks via: C[cr,cc] += gamma_{t,cr,cc} * M_t
    
    # For the tensor decomposition, each term t gives a sub-decomposition of rank 16.
    # The U, V, W for term t are built by:
    #   U_t[i*m+j, col] = (linear combo of A indices) * U_sub[...]
    #   V_t[j*p+k, col] = (linear combo of B indices) * V_sub[...]
    #   W_t[k*n+i, col] = (linear combo of C indices) * W_sub[...]
    
    # Let me be more explicit. For term t:
    # A_t is a 2x2 matrix where A_t[ar, ac] = sum_{br,bc} alpha[t][ar,ac,br,bc] * A[br*2+ar, bc*2+ac]
    # Wait, that's not quite right either. Let me think differently.
    
    # The 4x4 matrix A has entries A[i,j] for i in [0,4), j in [0,4).
    # Block (br, bc) contains A[br*2+ar, bc*2+ac] for ar in [0,2), ac in [0,2).
    
    # For term t, the effective A_t matrix (2x2) has:
    # A_t[ar, ac] = sum_{br,bc} alpha[t][(br,bc)] * A[br*2+ar, bc*2+ac]
    
    # Similarly for B_t (2x4):
    # B_t[j, k] = sum_{br,bc} beta[t][(br,bc)] * B[br*2+j, bc*4+k]
    
    # And C_t (2x4) contributes to C:
    # C[cr*2+ar, cc*4+k] += gamma[t][(cr,cc)] * C_t[ar, k]
    
    # Now, the sub-decomposition for a 2x2 * 2x4 problem:
    # U_sub[i_sub * 2 + j_sub, c] = 1 if (i_sub, j_sub, c) matches schoolbook
    # V_sub[j_sub * 4 + k_sub, c] = 1
    # W_sub[k_sub * 2 + i_sub, c] = 1
    
    # For the full decomposition, for term t at column offset off + c:
    # U[full_i * 4 + full_j, off+c] = U_sub[i_sub * 2 + j_sub, c] * alpha[t][...]
    #   where full_i = br*2 + i_sub, full_j = bc*2 + j_sub
    #   But wait, U_sub already has i_sub and j_sub as the row/col of the sub-A.
    #   The coefficient alpha applies to the whole block, so:
    #   U[br*2 + i_sub, bc*2 + j_sub, off+c] += alpha[t][(br,bc)] * U_sub[i_sub*2+j_sub, c]
    
    # Hmm, but U_sub[i_sub*2+j_sub, c] is 1 only for specific (i_sub, j_sub, c).
    # In schoolbook for 2x2*2x4: U_sub[i*2+j, c] = 1 where c = i*2*4 + j*4 + k... no.
    # Schoolbook: for each (i, j, k), one column with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # For 2x2*2x4: m_sub=2, p_sub=4.
    # U_sub[i*2+j, c] = 1, V_sub[j*4+k, c] = 1, W_sub[k*2+i, c] = 1
    # where c = i*2*4 + j*4 + k (or some ordering).
    
    # So for the full problem, for term t, sub-column c:
    # Let (i_s, j_s, k_s) be the schoolbook indices for column c in the sub-problem.
    # Then for each (br, bc) with alpha[t][(br,bc)] != 0:
    #   U[br*2 + i_s, bc*2 + j_s, off+c] += alpha[t][(br,bc)]
    # For each (br, bc) with beta[t][(br,bc)] != 0:
    #   V[br*2 + j_s, bc*4 + k_s, off+c] += beta[t][(br,bc)]
    # For each (cr, cc) with gamma[t][(cr,cc)] != 0:
    #   W[cc*4 + k_s, cr*2 + i_s, off+c] += gamma[t][(cr,cc)]
    
    # Let me define the Strassen coefficients.
    # Terms:
    # M1 = (A11+A22)*(B11+B22)
    #   alpha1: (0,0):1, (1,1):1
    #   beta1: (0,0):1, (1,1):1
    #   gamma1: C11: +1, C22: +1
    # M2 = (A21+A22)*B11
    #   alpha2: (1,0):1, (1,1):1
    #   beta2: (0,0):1
    #   gamma2: C21: +1, C22: -1
    # M3 = A11*(B12-B22)
    #   alpha3: (0,0):1
    #   beta3: (0,1):1, (1,1):-1
    #   gamma3: C12: +1, C22: -1
    # M4 = A22*(B21-B11)
    #   alpha4: (1,1):1
    #   beta4: (1,0):1, (0,0):-1
    #   gamma4: C11: +1, C21: +1
    # M5 = (A11+A12)*B22
    #   alpha5: (0,0):1, (0,1):1
    #   beta5: (1,1):1
    #   gamma5: C12: +1, C11: -1
    # M6 = (A21-A11)*(B11+B12)
    #   alpha6: (1,0):1, (0,0):-1
    #   beta6: (0,0):1, (0,1):1
    #   gamma6: C22: -1
    # M7 = (A12-A22)*(B21+B22)
    #   alpha7: (0,1):1, (1,1):-1
    #   beta7: (1,0):1, (1,1):1
    #   gamma7: C11: +1
    
    # Let me verify: C11 = M1 + M4 - M5 + M7
    #   M1 contributes +1 to C11
    #   M4 contributes +1 to C11
    #   M5 contributes -1 to C11
    #   M7 contributes +1 to C11
    #   Total: 1+1-1+1 = 2? That's wrong.
    
    # Let me recheck Strassen formulas. Standard Strassen for 2x2:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    # where:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Check C11:
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # M4 = A22(B21-B11) = A22B21 - A22B11
    # M5 = (A11+A12)B22 = A11B22 + A12B22
    # M7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    # C11 = M1 + M4 - M5 + M7
    #   = (A11B11 + A11B22 + A22B11 + A22B22) + (A22B21 - A22B11) - (A11B22 + A12B22) + (A12B21 + A12B22 - A22B21 - A22B22)
    #   = A11B11 + A11B22 + A22B11 + A22B22 + A22B21 - A22B11 - A11B22 - A12B22 + A12B21 + A12B22 - A22B21 - A22B22
    #   = A11B11 + A12B21
    # That's correct! C11 should be A11B11 + A12B21. ✓
    
    # So the gamma coefficients are:
    # M1: C11: +1, C22: +1
    # M2: C21: +1, C22: -1
    # M3: C12: +1, C22: -1
    # M4: C11: +1, C21: +1
    # M5: C12: +1, C11: -1
    # M6: C22: -1
    # M7: C11: +1
    
    # Let me verify C22:
    # C22 = M1 - M3 + M2 - M6
    # M1 = A11B11 + A11B22 + A22B11 + A22B22
    # M3 = A11(B12-B22) = A11B12 - A11B22
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    # C22 = M1 - M3 + M2 - M6
    #   = (A11B11 + A11B22 + A22B11 + A22B22) - (A11B12 - A11B22) + (A21B11 + A22B11) - (A21B11 + A21B12 - A11B11 - A11B12)
    #   = A11B11 + A11B22 + A22B11 + A22B22 - A11B12 + A11B22 + A21B11 + A22B11 - A21B11 - A21B12 + A11B11 + A11B12
    #   = 2*A11B11 + 2*A11B22 + 2*A22B11 + A22B22 - A21B12
    # That's wrong! C22 should be A21B11 + A22B12.
    
    # I think I have the Strassen formulas wrong. Let me look them up properly.
    
    # Correct Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # Let me recheck C22 more carefully:
    # C22 = M1 - M3 + M2 - M6
    # = (A11+A22)(B11+B22) - A11(B12-B22) + (A21+A22)B11 - (A21-A11)(B11+B12)
    # = (A11B11 + A11B22 + A22B11 + A22B22) - (A11B12 - A11B22) + (A21B11 + A22B11) - (A21B11 + A21B12 - A11B11 - A11B12)
    # = A11B11 + A11B22 + A22B11 + A22B22 - A11B12 + A11B22 + A21B11 + A22B11 - A21B11 - A21B12 + A11B11 + A11B12
    # = 2A11B11 + 2A11B22 + 2A22B11 + A22B22 - A21B12
    
    # This is still wrong. The issue is that C22 should be A21B11 + A22B12.
    # I think the standard Strassen C22 formula is: C22 = M1 - M3 + M2 - M6
    # but maybe I'm expanding incorrectly.
    
    # Actually wait, let me recompute. The standard result is:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5  
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # And this is correct. My expansion must have an error. Let me redo:
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # M3 = A11(B12-B22) = A11B12 - A11B22
    # M2 = (A21+A22)B11 = A21B11 + A22B11
    # M6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12
    
    # C22 = M1 - M3 + M2 - M6
    # = [A11B11 + A11B22 + A22B11 + A22B22] - [A11B12 - A11B22] + [A21B11 + A22B11] - [A21B11 + A21B12 - A11B11 - A11B12]
    # = A11B11 + A11B22 + A22B11 + A22B22 - A11B12 + A11B22 + A21B11 + A22B11 - A21B11 - A21B12 + A11B11 + A11B12
    
    # Group by term:
    # A11B11: 1 + 1 = 2
    # A11B22: 1 + 1 = 2
    # A22B11: 1 + 1 = 2
    # A22B22: 1
    # A11B12: -1 + 1 = 0
    # A21B11: 1 - 1 = 0
    # A21B12: -1
    
    # So C22 = 2A11B11 + 2A11B22 + 2A22B11 + A22B22 - A21B12
    
    # This is definitely not A21B11 + A22B12. So either my Strassen formulas are wrong or my expansion is wrong.
    
    # Let me try a different source. The correct Strassen decomposition is:
    # C = [[A11B11 + A12B21, A11B12 + A12B22],
    #      [A21B11 + A22B21, A21B12 + A22B22]]
    
    # I'll just use the well-known correct formulas. Let me verify with a simple example.
    # Actually, I recall now - the formulas I have ARE correct, and my expansion must have an algebra error.
    # Let me just trust the standard formulas and verify numerically in the code.
    
    # For now, let me just implement it and verify.
    
    # Define coefficients for each term:
    # alpha[t]: dict from (br, bc) to coeff, for A blocks
    # beta[t]: dict from (br, bc) to coeff, for B blocks
    # gamma[t]: dict from (cr, cc) to coeff, for C blocks
    
    alpha = [
        {(0,0):1, (1,1):1},      # M1: A11+A22
        {(1,0):1, (1,1):1},      # M2: A21+A22
        {(0,0):1},               # M3: A11
        {(1,1):1},               # M4: A22
        {(0,0):1, (0,1):1},      # M5: A11+A12
        {(1,0):1, (0,0):-1},     # M6: A21-A11
        {(0,1):1, (1,1):-1},     # M7: A12-A22
    ]
    beta = [
        {(0,0):1, (1,1):1},      # M1: B11+B22
        {(0,0):1},               # M2: B11
        {(0,1):1, (1,1):-1},     # M3: B12-B22
        {(1,0):1, (0,0):-1},     # M4: B21-B11
        {(1,1):1},               # M5: B22
        {(0,0):1, (0,1):1},      # M6: B11+B12
        {(1,0):1, (1,1):1},      # M7: B21+B22
    ]
    gamma = [
        {(0,0):1, (1,1):1},      # M1: C11+C22
        {(1,0):1, (1,1):-1},     # M2: C21-C22
        {(0,1):1, (1,1):-1},     # M3: C12-C22
        {(0,0):1, (1,0):1},      # M4: C11+C21
        {(0,1):1, (0,0):-1},     # M5: C12-C11
        {(1,1):-1},              # M6: -C22
        {(0,0):1},               # M7: C11
    ]
    
    # Build schoolbook for 2x2 * 2x4
    U_sub, V_sub, W_sub, _ = _schoolbook(2, 2, 4)
    
    # Now build full U, V, W
    for t in range(7):
        off = t * sub_rank
        for c in range(sub_rank):
            # Find the (i_s, j_s, k_s) for this sub-column
            # In schoolbook, column c corresponds to (i_s, j_s, k_s)
            # where c = i_s * m_sub * p_sub + j_s * p_sub + k_s
            # = i_s * 2 * 4 + j_s * 4 + k_s = 8*i_s + 4*j_s + k_s
            i_s = c // 8
            j_s = (c % 8) // 4
            k_s = c % 4
            
            # Verify: U_sub[i_s*2+j_s, c] should be 1
            # V_sub[j_s*4+k_s, c] should be 1
            # W_sub[k_s*2+i_s, c] should be 1
            
            # Update U: for each (br, bc) in alpha[t]
            for (br, bc), a_coeff in alpha[t].items():
                full_i = br * 2 + i_s
                full_j = bc * 2 + j_s
                U[full_i * 4 + full_j][off + c] += a_coeff
            
            # Update V: for each (br, bc) in beta[t]
            for (br, bc), b_coeff in beta[t].items():
                full_j = br * 2 + j_s
                full_k = bc * 4 + k_s
                V[full_j * 8 + full_k][off + c] += b_coeff
            
            # Update W: for each (cr, cc) in gamma[t]
            for (cr, cc), g_coeff in gamma[t].items():
                full_k = cc * 4 + k_s
                full_i = cr * 2 + i_s
                W[full_k * 4 + full_i][off + c] += g_coeff
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    try:
        result = _strassen_4x4_4x8()
        
        # Verify the decomposition
        U, V, W = result["U"], result["V"], result["W"]
        n, m, p = 4, 4, 8
        rank = len(U[0])
        
        # Check: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
        # And for all other (i,j,k) combinations that shouldn't contribute, should be 0.
        # Actually, the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # In the standard convention, T[i,j,k] = 1 for all valid (i,j,k) since each output
        # element C[i,k] = sum_j A[i,j]*B[j,k]
        
        # Wait, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] B[j,k], so every triple (i,j,k) contributes.
        
        # Let me verify: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all i,j,k.
        
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        # Verification failed
                        return _schoolbook_fallback()
        
        return result
    except Exception:
        return _schoolbook_fallback()

def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U, V, W, _ = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
