"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen for 6x6 using 2x2 blocks of 3x3.
    # Inner 3x3 multiplications are schoolbook (rank 27).
    # 7 Strassen steps * 27 inner terms = 189 rank.
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Gamma coefficients for Strassen reconstruction
    # gamma[q][u][v] where u, v are block row/col (0 or 1)
    # C_00 = M_0 + M_3 - M_4 + M_6
    # C_01 = M_2 + M_4
    # C_10 = M_1 + M_3
    # C_11 = M_0 - M_1 + M_2 + M_5
    # Note: My previous derivation used M1..M7. Let's map q=0..6 to M1..M7.
    # q=0 (M1): C00(+1), C11(+1)
    # q=1 (M2): C10(+1), C11(-1)
    # q=2 (M3): C01(+1), C11(+1)
    # q=3 (M4): C00(+1), C10(+1)
    # q=4 (M5): C00(-1), C01(+1)
    # q=5 (M6): C11(+1)
    # q=6 (M7): C00(+1)
    
    gammas = [
        { (0,0): 1, (1,1): 1 },      # q=0 (M1)
        { (1,0): 1, (1,1): -1 },     # q=1 (M2)
        { (0,1): 1, (1,1): 1 },      # q=2 (M3)
        { (0,0): 1, (1,0): 1 },      # q=3 (M4)
        { (0,0): -1, (0,1): 1 },     # q=4 (M5)
        { (1,1): 1 },                # q=5 (M6)
        { (0,0): 1 }                 # q=6 (M7)
    ]
    
    # Define U, V contributions for each q
    # U indices: i*6+j
    # V indices: j*6+k
    
    def get_U_V_indices(q, i_in, j_in):
        # Returns list of (u_idx, val), (v_idx, val)
        u_idx_vals = []
        v_idx_vals = []
        # Local indices i_in, j_in in 0..2
        # Global A indices:
        # Block 00: r in 0..2, c in 0..2 -> global r, c
        # Block 01: r in 0..2, c in 3..5 -> global r, c+3
        # Block 10: r in 3..5, c in 0..2 -> global r+3, c
        # Block 11: r in 3..5, c in 3..5 -> global r+3, c+3
        
        if q == 0: # M1: X=A00+A11, Y=B00+B11
            u_idx_vals.append((i_in * 6 + j_in, 1))
            u_idx_vals.append(((i_in+3) * 6 + (j_in+3), 1))
            v_idx_vals.append((j_in * 6 + i_in, 1)) # Note: V index is j*p+k. Here j_in is row of Y, i_in is col?
            # Wait. In M1 = X Y.
            # X[i,j] comes from A. Y[j,k] comes from B.
            # The term is X[i,j] * Y[j,k].
            # In the tensor decomposition, U is coeff of A[i,j], V is coeff of B[j,k].
            # So for the term (i_in, j_in, k_in), U picks A part for X[i_in, j_in].
            # V picks B part for Y[j_in, k_in].
            # My loop variables in construct will be i_in, j_in, k_in.
            # So here, j_in is the inner dimension.
            # X is indexed by i_in, j_in.
            # Y is indexed by j_in, k_in.
            # So for V, the indices depend on j_in and k_in.
            # I passed j_in to this helper, but I need k_in for V.
            # I will restructure the loop in construct to handle U and V separately or pass both.
            pass
        return u_idx_vals, v_idx_vals

    # It is safer to just write it out in the loop.
    
    for q in range(7):
        for i_in in range(3):
            for j_in in range(3):
                for k_in in range(3):
                    r = q * 27 + (i_in * 3 + j_in) * 3 + k_in
                    
                    # U contributions (from A)
                    if q == 0: # M1: X=A00+A11
                        U[i_in * 6 + j_in][r] += 1
                        U[(i_in+3) * 6 + (j_in+3)][r] += 1
                    elif q == 1: # M2: X=A10+A11
                        U[(i_in+3) * 6 + j_in][r] += 1
                        U[(i_in+3) * 6 + (j_in+3)][r] += 1
                    elif q == 2: # M3: X=A00
                        U[i_in * 6 + j_in][r] += 1
                    elif q == 3: # M4: X=A11
                        U[(i_in+3) * 6 + (j_in+3)][r] += 1
                    elif q == 4: # M5: X=A00+A01
                        U[i_in * 6 + j_in][r] += 1
                        U[i_in * 6 + (j_in+3)][r] += 1
                    elif q == 5: # M6: X=A10-A00
                        U[(i_in+3) * 6 + j_in][r] += 1
                        U[i_in * 6 + j_in][r] -= 1
                    elif q == 6: # M7: X=A01-A11
                        U[i_in * 6 + (j_in+3)][r] += 1
                        U[(i_in+3) * 6 + (j_in+3)][r] -= 1

                    # V contributions (from B)
                    # Y is indexed by j_in, k_in
                    if q == 0: # Y=B00+B11
                        V[j_in * 6 + k_in][r] += 1
                        V[(j_in+3) * 6 + (k_in+3)][r] += 1
                    elif q == 1: # Y=B00
                        V[j_in * 6 + k_in][r] += 1
                    elif q == 2: # Y=B01-B11
                        V[j_in * 6 + (k_in+3)][r] += 1
                        V[(j_in+3) * 6 + (k_in+3)][r] -= 1
                    elif q == 3: # Y=B10-B00
                        V[(j_in+3) * 6 + k_in][r] += 1
                        V[j_in * 6 + k_in][r] -= 1
                    elif q == 4: # Y=B11
                        V[(j_in+3) * 6 + (k_in+3)][r] += 1
                    elif q == 5: # Y=B00+B01
                        V[j_in * 6 + k_in][r] += 1
                        V[j_in * 6 + (k_in+3)][r] += 1
                    elif q == 6: # Y=B10+B11
                        V[(j_in+3) * 6 + k_in][r] += 1
                        V[(j_in+3) * 6 + (k_in+3)][r] += 1

                    # W contributions (to C)
                    # M_q[i_in, k_in] contributes to C blocks.
                    for (u_blk, v_blk), val in gammas[q].items():
                        i_global = u_blk * 3 + i_in
                        k_global = v_blk * 3 + k_in
                        # W index is k*n + i
                        W[k_global * 6 + i_global][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
